# MIT License

# Copyright (c) 2018 the NJUNLP groups.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author baoyu.nlp   
# Time 2019-01-22 10:21

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from dss_vae.decoder import get_decoder
from dss_vae.encoder import get_encoder
from dss_vae.networks.bridger import get_bridger
from dss_vae.networks.criterions import SequenceCriterion
from dss_vae.networks.embeddings import Embeddings
from dss_vae.networks.position import get_position_select
from dss_vae.utils.input_funcs import to_input_dict
from dss_vae.utils.input_funcs import to_target_word
from dss_vae.utils.loss_funcs import advance_loss, mt_loss
from dss_vae.utils.loss_funcs import batch_bow_loss
from dss_vae.utils.mask_funcs import relative_relation_mask
from dss_vae.utils.nn_funcs import unk_replace
from .base_gen import BaseGenerator

INF = 1e10
TINY = 1e-9


def topk_search(logits, mask_src, beam_size=100):
    # prepare data
    neg_log_prob = -log_softmax(logits).data
    max_len = neg_log_prob.size(-1)
    overmask = torch.cat([mask_src[:, :, None],
                          (1 - mask_src[:, :, None]).expand(*mask_src.size(), max_len - 1) * INF
                          + mask_src[:, :, None]], 2)
    neg_log_prob = neg_log_prob * overmask

    batch_size, src_len, v_size = logits.size()
    _, r_ids = neg_log_prob.sort(-1)

    def get_score(data, index):
        # avoid all zero
        # zero_mask = (index.sum(-2) == 0).float() * INF
        return data.gather(-1, index).sum(-2)

    heap_scores = torch.ones(batch_size, beam_size) * INF
    heap_inx = torch.zeros(batch_size, src_len, beam_size).long()
    heap_scores[:, :1] = get_score(neg_log_prob, r_ids[:, :, :1])
    if neg_log_prob.is_cuda:
        heap_scores = heap_scores.cuda(neg_log_prob.get_device())
        heap_inx = heap_inx.cuda(neg_log_prob.get_device())

    def span(ins):
        inds = torch.eye(ins.size(1)).long()
        if ins.is_cuda:
            inds = inds.cuda(ins.get_device())
        return ins[:, :, None].expand(ins.size(0), ins.size(1), ins.size(1)) + inds[None, :, :]

    # iteration starts
    for k in range(1, beam_size):
        cur_inx = heap_inx[:, :, k - 1]
        I_t = span(cur_inx).clamp(0, v_size - 1)  # B x N x N
        S_t = get_score(neg_log_prob, r_ids.gather(-1, I_t))
        S_t, _inx = torch.cat([heap_scores[:, k:], S_t], 1).sort(1)
        S_t[:, 1:] += ((S_t[:, 1:] - S_t[:, :-1]) == 0).float() * INF  # remove duplicates
        S_t, _inx2 = S_t.sort(1)
        I_t = torch.cat([heap_inx[:, :, k:], I_t], 2).gather(
            2, _inx.gather(1, _inx2)[:, None, :].expand(batch_size, src_len, _inx.size(-1)))
        heap_scores[:, k:] = S_t[:, :beam_size - k]
        heap_inx[:, :, k:] = I_t[:, :, :beam_size - k]

    # get the searched
    output = r_ids.gather(-1, heap_inx)
    output = output.transpose(2, 1).contiguous().view(batch_size * beam_size, src_len)  # (B x N) x Ts
    output = Variable(output)
    mask_src = mask_src[:, None, :].expand(batch_size, beam_size, src_len).contiguous().view(batch_size * beam_size,
                                                                                             src_len)

    return output, mask_src


class NonAutoGenerator(nn.Module, BaseGenerator):
    def __init__(self, args, vocab, dec_type, embed=None, name="Non-Autoregressive Base GEN", **kwargs):
        super(NonAutoGenerator, self).__init__()
        print("Init:\t", name)
        self.args = args
        self.vocab = vocab

        if "encoder" in kwargs and kwargs['encoder'] is not None:
            self.encoder = kwargs['encoder']
        else:
            self.encoder = get_encoder(
                args=args,
                vocab_size=len(self.vocab.src),
                model=args.enc_type,
                embedding=embed,
                pad=self.vocab.src.pad_id,
            )

        model_dim = self.encoder.out_dim

        self.bridger = get_bridger(
            model_cls=args.map_type,
            input_dim=model_dim,
            hidden_dim=model_dim,
            k_dim=args.tgt_max_time_step,
            v_dim=model_dim,
            dropout=args.dropm,
            head_num=args.head_num,
        )
        self.decoder = get_decoder(
            model=dec_type,
            args=args,
            hidden_dim=model_dim,
            vocab_size=len(self.vocab.tgt),
            pad=self.vocab.tgt.pad_id,
        )

        self.max_len = args.tgt_max_time_step
        self.word_drop = args.word_drop

        self.normalization = 1.0
        self.norm_by_words = False
        self.critic = SequenceCriterion(padding_idx=self.vocab.tgt.pad_id)

        if args.pos_initial:
            self.position_block = get_position_select(
                pos_type=args.pos_type,
                pos_feat=args.pos_feat,
                pos_pred=args.pos_pred,
                use_rank=args.pos_rank if 'pos_rank' in args else None,
                use_mse=args.pos_mse if 'pos_mse' in args else None,
                use_dst=args.pos_dst if 'pos_dst' in args else None,
                att_num_layer=args.pos_att_layer,
                rnn_num_layer=args.pos_rnn_layer,
                model_dim=model_dim,
                head_num=args.head_num,
                max_len=self.max_len,
                use_pos_exp=args.use_pos_exp,
                use_pos_pred=args.use_pos_pred,
                use_st_gumbel=args.use_st_gumbel
            )

    def init_decoder_input_with_position(self, **kwargs):
        return self.position_block.forward(**kwargs)

    def encode(self, seqs_x, seqs_length=None):
        """
        Args:
            seqs_x: torch.LongTensor<batch,seq_len(max)>
            seqs_length: truly sequence length

        Returns:

        """
        if self.training and self.word_drop > 0.:
            seqs_x = unk_replace(seqs_x, dropoutr=self.word_drop, vocab=self.vocab.src)

        if self.args.enc_type == "att":
            enc_ret = self.encoder.forward(seqs_x)
        else:
            enc_out, enc_hid = self.encoder.forward(seqs_x, None)
            enc_ret = {
                'out': enc_out,
                'hidden': enc_hid,
                'mask': seqs_x.detach().eq(self.vocab.src.pad_id)
            }
        return enc_ret

    def bridge(self, encoder_outputs, encoder_mask=None):
        return self.bridger.forward(encoder_output=encoder_outputs, encoder_mask=encoder_mask)

    def forward(self, **kwargs):
        """
        including two functionality: predict or prob
        Args:
            **kwargs:

        Returns: dict

        """
        raise NotImplementedError

    def predict(self, **kwargs):
        """
        return the predict output of model
        Args:
            **kwargs:

        Returns: word list

        """
        raise NotImplementedError

    def score(self, examples, to_word=False, **kwargs):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]
        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            training=self.training,
            src_append=False,
            use_tgt=True,
            use_tag=args.use_arc,
            use_dst=args.use_dst
        )
        ret = self.forward(
            seqs_x=input_dict['src'],
            x_length=input_dict['src_len'],
            seqs_y=input_dict['tgt'],
            to_word=to_word,
            log_prob=False
        )

        # pos_word_score = self.decoder.scoring(raw_score=ret['word'], tgt_var=input_dict['tgt'])
        raw_word_score = ret['word']
        neg_log_score = advance_loss(
            logits=raw_word_score,
            tgt_var=input_dict['tgt'],
            log_prob=False,
            pad=self.vocab.tgt.pad_id
        )
        if "layer_bow_probs" in ret:
            loss = 0.0
            layer_bow_probs = ret['layer_bow_probs']
            decay_weights = np.arange(len(layer_bow_probs))[::-1] / len(layer_bow_probs) + 0.1
            for decay_weight, layer_bow_prob in zip(decay_weights, layer_bow_probs):
                neg_log_bow_score = batch_bow_loss(
                    logits=layer_bow_prob,
                    tgt_var=input_dict['tgt'],
                    log_prob=False,
                    pad=self.vocab.tgt.pad_id
                )
                loss += neg_log_bow_score * decay_weight
            neg_log_score += loss
        return neg_log_score.mean()  # [batch_size] without word norm

    def get_loss(self, **kwargs):
        return {
            "Loss": self.score(**kwargs)
        }

    def parameters(self):
        # if self.args.pretrain_exp_dir is not None:
        #     if "fine_tune" in self.args and not self.args.fine_tune:
        #         for name, param in self.named_parameters():
        #             if name.startswith("encoder") and "share_encoder" in self.args:
        #                 param.requires_grad = self.args.share_encoder
        #             elif name.startswith("tgt_embed") and "share_tgt_embed" in self.args:
        #                 param.requires_grad = self.args.share_tgt_embed
        #             else:
        #                 yield param
        #     else:
        #         for name, param in self.named_parameters():
        #             yield param
        # else:
        for name, param in self.named_parameters():
            yield param

    def load_state_dict(self, state_dict, strict=True):
        return super(NonAutoGenerator).load_state_dict(state_dict, strict)

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


class SelfAttNAG(NonAutoGenerator):
    def __init__(self, name="Self-ATT Non-Auto GEN", **kwargs):
        super(SelfAttNAG, self).__init__(dec_type='self-att', name=name, **kwargs)

    def forward(self, seqs_x, seqs_y=None, x_length=None, to_word=False, log_prob=True):
        enc_ret = self.encode(seqs_x, seqs_length=x_length)
        encoder_output = enc_ret['out']
        bridget_ret = self.bridger.forward(encoder_output=encoder_output, encoder_mask=enc_ret['mask'])
        dec_input = bridget_ret['out']
        if self.args.pos_initial:
            dec_input = self.init_decoder_input_with_position(dec_input)['out']
        ret = self.decoder.forward(dec_input0=dec_input)
        if log_prob:
            ret['word'] = F.log_softmax(ret['word'], dim=-1)
        if to_word:
            ret["pred"] = to_target_word(ret['word'], vocab=self.vocab.tgt)
        return ret

    def predict(self, examples, to_word=True):
        if not isinstance(examples, list):
            examples = [examples]

        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            max_tgt_len=self.max_len,
            training=self.training,
            src_append=False,
        )
        predict = self.forward(seqs_x=input_dict['src'], x_length=input_dict['src_len'], to_word=to_word)
        return predict['pred']


class CondAttNAG(NonAutoGenerator):
    def __init__(self, name="Cond-ATT Non-Auto GEN", **kwargs):
        super(CondAttNAG, self).__init__(dec_type='cond-att', name=name, **kwargs)

    def forward(self, seqs_x, x_length=None, to_word=False, log_prob=True, **kwargs):
        enc_ret = self.encode(seqs_x, seqs_length=x_length)
        bridget_ret = self.bridger.forward(encoder_output=enc_ret['out'], encoder_mask=enc_ret['mask'])
        dec_input = bridget_ret['out']
        enc_mask = enc_ret['mask'] if self.args.use_enc_mask else None
        dec_mask = None
        if self.args.pos_initial:
            init_ret = self.init_decoder_input_with_position(
                dec_inputs=dec_input, enc_outputs=enc_ret['out'],
                enc_mask=enc_mask
            )
            dec_input = init_ret['out']
            if "use_dec_mask" in self.args and self.args.use_dec_mask:
                dec_mask = init_ret['mask']

        dec_ret = self.decoder.forward(
            dec_input0=dec_input,
            encoder_output=enc_ret['out'],
            enc_mask=enc_mask,
            dec_mask=dec_mask
        )
        if log_prob:
            dec_ret['word'] = dec_ret['word'].log_softmax(dim=-1)
        if to_word:
            dec_ret["pred"] = to_target_word(dec_ret['word'], vocab=self.vocab.tgt)
        return dec_ret

    def predict(self, examples, to_word=True):
        self.eval()
        if not isinstance(examples, list):
            examples = [examples]

        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            max_tgt_len=-1 if self.args.map_type == "base" else self.args.tgt_max_time_step,
            training=self.training,
            src_append=False,
            tgt_append=False,
            scale_to_tgt=self.args.tgt_scale if self.args.map_type == "base" else 0.0,
        )
        predict_ret = self.forward(
            seqs_x=input_dict['src'],
            x_length=input_dict['src_len'],
            to_word=to_word,
            log_prob=False
        )
        return predict_ret['pred']

    def score(self, examples, to_word=False, **kwargs):
        self.train()
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]
        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            max_tgt_len=-1 if self.args.map_type == "base" else self.args.tgt_max_time_step,
            training=self.training,
            src_append=False,
            tgt_append=False,
            use_tgt=True,
            use_tag=args.use_arc,
            use_dst=args.use_dst,
            scale_to_tgt=self.args.tgt_scale if self.args.map_type == "base" else 0.0,
        )
        ret = self.forward(
            seqs_x=input_dict['src'],
            x_length=input_dict['src_len'],
            to_word=to_word,
            log_prob=False
        )

        word_prob = ret['word']
        # neg_log_score = -self.decoder.scoring(raw_score=word_prob, tgt_var=input_dict['tgt'])
        # neg_log_score = advance_loss(
        #     logits=word_prob,
        #     tgt_var=input_dict['tgt'],
        #     log_prob=False,
        #     pad=self.vocab.tgt.pad_id
        # )
        neg_log_score = mt_loss(
            logits=word_prob,
            tgt_var=input_dict['tgt'],
            log_prob=False,
            critic=self.critic,
            pad=self.vocab.tgt.pad_id,
            norm_by_word=self.norm_by_words,
            normalization=self.normalization
        )
        if "layer_bow_probs" in ret:
            loss = 0.0
            layer_bow_probs = ret['layer_bow_probs']
            decay_weights = np.arange(len(layer_bow_probs))[::-1] / len(layer_bow_probs) + 0.1
            for decay_weight, layer_bow_prob in zip(decay_weights, layer_bow_probs):
                neg_log_bow_score = batch_bow_loss(
                    logits=layer_bow_prob,
                    tgt_var=input_dict['tgt'],
                    log_prob=False,
                    pad=self.vocab.tgt.pad_id
                )
                loss += neg_log_bow_score * decay_weight
            neg_log_score += loss
        return neg_log_score


class NonAutoReorderGEN(NonAutoGenerator):
    def __init__(self, name="Cond-ATT Non-Auto Reorder GEN", **kwargs):
        super(NonAutoReorderGEN, self).__init__(dec_type='cond-att', name=name, **kwargs)
        if "tgt_embed" in kwargs and kwargs['tgt_embed'] is not None:
            self.tgt_embed = kwargs['tgt_embed']
            self.tgt_embed.add_position_embedding = False
        else:
            self.tgt_embed = Embeddings(
                num_embeddings=len(self.vocab.tgt),
                embedding_dim=self.encoder.out_dim,
                dropout=0.1,
                add_position_embedding=False,
                padding_idx=self.vocab.tgt.pad_id
            )

    def forward(self, seqs_x, seqs_y, x_length=None, to_word=False, log_prob=True, position_ref=None,
                position_mask=None):
        """

        Args:
            seqs_x:
            seqs_y:
            x_length:
            to_word:
            log_prob:
            position_ref:
            position_mask:

        Returns:

        """
        enc_ret = self.encode(seqs_x, seqs_length=x_length)
        dec_input = self.tgt_embed(seqs_y)
        dec_mask = None
        init_ret = None
        if self.args.pos_initial:
            encoder_output = enc_ret['out']
            batch_size, seq_len, hidden = encoder_output.size()
            encoder_mask = enc_ret['mask']
            out_len = dec_input.size(1)
            dec_enc_mask = encoder_mask.unsqueeze(1).expand(batch_size, out_len, seq_len).contiguous()
            init_ret = self.init_decoder_input_with_position(dec_inputs=dec_input, enc_outputs=encoder_output,
                                                             enc_mask=dec_enc_mask, pos_ref=position_ref,
                                                             pos_mask=position_mask)
            if "use_dec_mask" in self.args and self.args.use_dec_mask:
                dec_mask = init_ret['mask']

        dec_ret = self.decoder.forward(dec_input0=dec_input, encoder_output=enc_ret['out'],
                                       enc_mask=enc_ret['mask'], dec_mask=dec_mask)
        dec_ret['init'] = init_ret
        if log_prob:
            dec_ret['word'] = F.log_softmax(dec_ret['word'], dim=-1)
        if to_word:
            dec_ret["pred"] = to_target_word(dec_ret['word'], vocab=self.vocab.tgt)

        return dec_ret

    def predict(self, examples, to_word=True):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            max_tgt_len=self.max_len,
            cuda=args.cuda,
            training=self.training,
            src_append=False,
            tgt_append=True,
            use_tgt=True,
            use_tag=False,
            use_dst=False,
            shuffle_tgt=True,
        )
        predict = self.forward(
            seqs_x=input_dict['src'],
            seqs_y=input_dict['s_tgt'],
            x_length=input_dict['src_len'],
            to_word=to_word,
            position_ref=input_dict['s_pos'] if self.args.pos_oracle else None
        )
        return predict['pred']

    def score(self, examples, to_word=False, distance_threshold=3, **kwargs):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            max_tgt_len=self.max_len,
            cuda=args.cuda,
            training=self.training,
            src_append=False,
            tgt_append=True,
            use_tgt=True,
            use_tag=args.use_arc,
            use_dst=args.use_dst,
            shuffle_tgt=True,
            scale_to_tgt=0.3
        )
        pos_ref = input_dict['s_pos']
        pos_mask = torch.gt(input_dict['s_tgt'], self.vocab.tgt.pad_id).long()
        ret = self.forward(
            seqs_x=input_dict['src'],
            x_length=input_dict['src_len'],
            seqs_y=input_dict['s_tgt'],
            to_word=to_word,
            log_prob=False,
            position_ref=input_dict['s_pos'] if self.args.pos_supervised else None,
            position_mask=pos_mask if self.args.pos_supervised else None,
        )
        if args.word_supervised:
            log_word_score = self.decoder.scoring(raw_score=ret['word'], tgt_var=input_dict['tgt'])
        else:
            log_word_score = 0.0

        init_ret = ret['init']
        pos_pred = init_ret['pos']

        correct = torch.eq(pos_pred, pos_ref).long() * pos_mask

        relax_correct = torch.le(torch.abs(pos_pred - pos_ref), distance_threshold).long() * pos_mask

        relative_pred, relative_ref, relative_mask = relative_relation_mask(pos_pred, pos_ref, pos_mask)

        relative_correct = torch.eq(relative_pred, relative_ref).long() * relative_mask
        binary_correct = torch.eq(relative_pred.gt(0), relative_ref.gt(0)).long() * relative_mask

        if self.args.pos_supervised:
            log_pos_score = init_ret['loss']
        else:
            log_pos_score = 0.0
        return {
            'Loss': -log_word_score - log_pos_score,
            'Word Loss': -log_word_score,
            'Pos Loss': -log_pos_score,
            'correct': correct.sum().item(),
            'relax_correct': relax_correct.sum().item(),
            'relative_correct': relative_correct.sum().item(),
            'binary_correct': binary_correct.sum().item(),
            'count': pos_mask.sum().item(),
            'relative_count': relative_mask.sum().item(),
            'Local Acc': 100.0 * correct.sum().float() / pos_mask.sum().float()
        }

    # def pos_loss(self, pos_logits, pos_ref, mask=None):
    #     if self.args.pos_type == "absolute":
    #         batch_size = pos_ref.size(0)
    #         vocab_size = pos_logits.size(-1)
    #         pos_logits = pos_logits.contiguous().transpose(1, 0).contiguous()
    #         pos_ref = pos_ref.contiguous().transpose(1, 0)
    #         flattened_mask = mask.transpose(1, 0).contiguous().view(-1)
    #         log_probs = F.log_softmax(pos_logits.view(-1, vocab_size).contiguous(), dim=-1)
    #         flattened_tgt_pos = pos_ref.contiguous().view(-1)
    #         pos_log_probs = torch.gather(log_probs, 1, flattened_tgt_pos.unsqueeze(1)).squeeze(1)
    #         pos_log_probs = pos_log_probs * flattened_mask.float()
    #         pos_log_probs = pos_log_probs.view(-1, batch_size).sum(dim=0)
    #         return pos_log_probs
    #     elif self.args.pos_type == "relative":
    #         return -pos_loss(inputs=pos_logits, target=pos_ref, mask=mask, use_mse=True, use_rank=False)
    #     else:
    #         return 0.0

    def get_loss(self, **kwargs):
        return self.score(**kwargs)
