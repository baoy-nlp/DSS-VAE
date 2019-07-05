# from utils.nn_utils import *
import os

import torch
import torch.nn as nn

from dss_vae.decoder.beam_decoder import TopKDecoder
from dss_vae.decoder.rnn_decoder import RNNDecoder
from dss_vae.encoder import RNNEncoder
from dss_vae.networks.bridger import MLPBridger
from dss_vae.utils.nn_funcs import id2word
from dss_vae.utils.nn_funcs import to_input_dict
from dss_vae.utils.nn_funcs import to_input_variable
from .base_gen import BaseGenerator


class BaseSeq2seq(nn.Module, BaseGenerator):
    def __init__(self, args, vocab, src_embed=None, tgt_embed=None):
        super(BaseSeq2seq, self).__init__()
        self.vocab = vocab
        self.src_vocab = vocab.src
        self.tgt_vocab = vocab.tgt
        self.args = args
        self.encoder = RNNEncoder(
            vocab_size=len(self.src_vocab),
            max_len=args.src_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=args.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=src_embed
        )

        self.enc_factor = 2 if args.bidirectional else 1
        self.enc_dim = args.enc_hidden_dim * self.enc_factor

        if args.mapper_type == "link":
            self.dec_hidden = self.enc_dim
        elif args.use_attention:
            self.dec_hidden = self.enc_dim
        else:
            self.dec_hidden = args.dec_hidden_dim

        self.bridger = MLPBridger(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=self.enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=self.dec_hidden,
            decoder_layer=args.dec_num_layers,
        )

        self.decoder = RNNDecoder(
            vocab=len(self.tgt_vocab),
            max_len=args.tgt_max_time_step,
            input_size=args.dec_embed_dim,
            hidden_size=self.dec_hidden,
            embed_droprate=args.dec_ed,
            rnn_droprate=args.dec_rd,
            n_layers=args.dec_num_layers,
            rnn_cell=args.rnn_type,
            use_attention=args.use_attention,
            embedding=tgt_embed,
            eos_id=self.tgt_vocab.eos_id,
            sos_id=self.tgt_vocab.sos_id,
        )

        self.beam_decoder = TopKDecoder(
            decoder_rnn=self.decoder,
            k=args.sample_size
        )
        print("enc layer: {}, dec layer: {}, type: {}, with attention: {}".format(
            args.enc_num_layers,
            args.dec_num_layers,
            args.rnn_type,
            args.use_attention)
        )

    def get_loss(self, **kwargs):
        return {
            "Loss": -self.score(**kwargs)
        }

    def forward(self, seqs_x, x_length, to_word=False):
        pass

    def init(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def encode(self, src_var, src_length):
        encoder_outputs, encoder_hidden = self.encoder.forward(input_var=src_var, input_lengths=src_length)
        return encoder_outputs, encoder_hidden

    def bridge(self, encoder_hidden):
        # batch_size = encoder_hidden.size(1)
        # convert = encoder_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bridger.forward(encoder_hidden)

    def get_hidden(self, examples):
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
            use_tgt=True,
            use_tag=False,
            use_dst=False,
        )

        src_var = input_dict['src']
        tgt_var = input_dict['tgt']
        src_length = input_dict['src_len']

        encoder_outputs, encoder_hidden = self.encode(src_var=src_var, src_length=src_length)
        encoder_hidden = self.bridge(encoder_hidden)
        return encoder_hidden

    def score(self, examples, return_enc_state=False, **kwargs):
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
            use_tgt=True,
            use_tag=False,
            use_dst=False,
        )

        src_var = input_dict['src']
        tgt_var = input_dict['tgt']
        src_length = input_dict['src_len']

        encoder_outputs, encoder_hidden = self.encode(src_var=src_var, src_length=src_length)
        encoder_hidden = self.bridge(encoder_hidden)
        scores = self.decoder.score(inputs=tgt_var, encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs)

        if return_enc_state:
            return scores, encoder_hidden
        else:
            return scores

    def predict(self, examples, to_word=True):
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

        src_var = input_dict['src']
        src_length = input_dict['src_len']

        encoder_outputs, encoder_hidden = self.encode(src_var=src_var, src_length=src_length)
        encoder_hidden = self.bridge(encoder_hidden)

        decoder_output, decoder_hidden, ret_dict, _ = self.decoder.forward(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            teacher_forcing_ratio=0.0
        )
        result = torch.stack(ret_dict['sequence']).squeeze()
        final_result = []
        if len(result.size()) < 2:
            result = result.view(-1, 1)
        example_nums = result.size(-1)
        if to_word:
            for i in range(example_nums):
                hyp = result[:, i].data.tolist()
                res = id2word(hyp, self.vocab.tgt)
                seems = [[res], [len(res)]]
                final_result.append(seems)
        return final_result

    def beam_search(self, src_sent, beam_size=5, dmts=None):
        if dmts is None:
            dmts = self.args.decode_max_time_step
        src_var = to_input_variable(src_sent, self.src_vocab,
                                    cuda=self.args.cuda, training=False, append_boundary_sym=False, batch_first=True)
        src_length = [len(src_sent)]

        encoder_outputs, encoder_hidden = self.encode(src_var=src_var, src_length=src_length)
        encoder_hidden = self.bridger.forward(input_tensor=encoder_hidden)
        meta_data = self.beam_decoder.beam_search(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            beam_size=beam_size,
            decode_max_time_step=dmts
        )
        topk_sequence = meta_data['sequence']
        topk_score = meta_data['score'].squeeze()

        completed_hypotheses = torch.cat(topk_sequence, dim=-1)

        number_return = completed_hypotheses.size(0)
        final_result = []
        final_scores = []
        for i in range(number_return):
            hyp = completed_hypotheses[i, :].data.tolist()
            res = id2word(hyp, self.tgt_vocab)
            final_result.append(res)
            final_scores.append(topk_score[i].item())
        return final_result, final_scores

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict)

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
