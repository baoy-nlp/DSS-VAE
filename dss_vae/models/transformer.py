import os

import torch
import torch.nn as nn

from dss_vae.decoder.att_decoder import TransformerDecoder
from dss_vae.encoder import TransformerEncoder
from dss_vae.networks.criterions import SequenceCriterion
from dss_vae.utils.beam_search import beam_search
from dss_vae.utils.nest import map_structure
from dss_vae.utils.nn_funcs import data_to_word
from dss_vae.utils.nn_funcs import to_input_dict
from dss_vae.utils.tensor_ops import tensor_gather_helper
from dss_vae.utils.tensor_ops import tile_batch
from .base_gen import BaseGenerator


class Transformer(nn.Module, BaseGenerator):
    """
    A sequence to sequence model with attention mechanism.
    """

    def __init__(self, args, vocab):
        super(Transformer, self).__init__()
        self.args = args
        self.vocab = vocab
        self.max_steps = self.args.tgt_max_time_step

        n_src_vocab = len(self.vocab.src)
        n_tgt_vocab = len(self.vocab.tgt)
        d_word_vec = args.embed_size
        d_model = args.hidden_size

        self.eos = self.vocab.tgt.eos_id
        self.sos = self.vocab.tgt.sos_id
        self.pad = self.vocab.tgt.pad_id

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        self.encoder = TransformerEncoder(
            vocab_size=n_src_vocab,
            n_layers=args.enc_num_layers,
            n_head=args.enc_head,
            input_size=d_word_vec,
            hidden_size=d_model,
            inner_hidden=args.enc_inner_hidden,
            embed_dropout=args.enc_ed,
            block_dropout=args.enc_rd,
            pad=self.pad
        )

        self.decoder = TransformerDecoder(
            vocab_size=n_tgt_vocab,
            n_layers=args.dec_num_layers,
            n_head=args.dec_head,
            input_size=d_word_vec,
            hidden_size=d_model,
            inner_hidden=args.dec_inner_hidden,
            share_proj_weight=args.share_proj_weight,
            share_embed_weight=self.encoder.embeddings if args.share_embed_weight else None,
            embed_dropout=args.dec_ed,
            block_dropout=args.dec_rd,
            pad=self.pad
        )

        self.dropout = nn.Dropout(args.drop)
        self.generator = self.decoder.generator

        self.normalization = 1.0
        self.norm_by_words = False
        self.critic = SequenceCriterion(padding_idx=self.pad)

    def forward(self, src_seq, tgt_seq, log_probs=True):

        ret = self.encode(src_seq)
        enc_output = ret['ctx']
        enc_mask = ret['ctx_mask']
        dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask)

        return self.generator(dec_output, log_probs=log_probs)

    def encode(self, src_seq):

        enc_out = self.encoder(src_seq)

        return {"ctx": enc_out['out'], "ctx_mask": enc_out['mask']}

    def init_decoder(self, enc_outputs, expand_size=1):

        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        if expand_size > 1:
            ctx = tile_batch(ctx, multiplier=expand_size)
            ctx_mask = tile_batch(ctx_mask, multiplier=expand_size)

        return {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "enc_attn_caches": None,
            "slf_attn_caches": None
        }

    def decode(self, tgt_seq, dec_states, log_probs=True):

        ctx = dec_states["ctx"]
        ctx_mask = dec_states['ctx_mask']
        enc_attn_caches = dec_states['enc_attn_caches']
        slf_attn_caches = dec_states['slf_attn_caches']

        dec_output, slf_attn_caches, enc_attn_caches = self.decoder(
            tgt_seq=tgt_seq,
            enc_output=ctx,
            enc_mask=ctx_mask,
            enc_attn_caches=enc_attn_caches,
            self_attn_caches=slf_attn_caches
        )

        next_scores = self.generator(dec_output[:, -1].contiguous(), log_probs=log_probs)
        dec_states['enc_attn_caches'] = enc_attn_caches
        dec_states['slf_attn_caches'] = slf_attn_caches

        return next_scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        slf_attn_caches = dec_states['slf_attn_caches']

        batch_size = slf_attn_caches[0][0].size(0) // beam_size

        n_head = self.decoder.n_head
        dim_per_head = self.decoder.dim_per_head

        slf_attn_caches = map_structure(
            lambda t: tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=t,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, n_head, -1, dim_per_head],
                                           use_gpu=self.args.cuda),
            slf_attn_caches)

        dec_states['slf_attn_caches'] = slf_attn_caches

        return dec_states

    def score(self, examples, return_enc_state=False, **kwargs):
        """
            Used for teacher-forcing training,
            return the log_probability of <input,output>.
        """
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            max_tgt_len=self.max_steps,
            cuda=args.cuda,
            training=self.training,
            src_append=False,
            use_tgt=True,
        )

        seqs_x = input_dict['src']
        seqs_y = input_dict['tgt']
        y_inp = seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()
        words_norm = y_label.ne(self.pad).float().sum(1)

        log_probs = self.forward(seqs_x, y_inp)

        loss = self.critic(inputs=log_probs, labels=y_label, reduce=False, normalization=self.normalization)

        if self.norm_by_words:
            loss = loss.div(words_norm)
        else:
            loss = loss
        return -loss

    def predict(self, examples, to_word=True, beam_size=1):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            max_tgt_len=-1,
            cuda=args.cuda,
            training=self.training,
            src_append=False,
            tgt_append=True,
            use_tgt=False,
            use_tag=False,
            use_dst=False,
        )
        ret = beam_search(
            model=self,
            beam_size=beam_size,
            max_steps=self.max_steps,
            src_seqs=input_dict['src'],
            alpha=-1.0
        )
        if to_word:
            ret = data_to_word(ret, self.vocab.tgt)

        return ret

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict(),
        }

        torch.save(params, path)

    @classmethod
    def load(cls, load_path):
        params = torch.load(load_path, map_location=lambda storage, loc: storage)
        args = params['args']
        vocab = params['vocab']
        model = cls(args, vocab)
        model.load_state_dict(params['state_dict'])
        if args.cuda:
            model = model.cuda()
        return model
