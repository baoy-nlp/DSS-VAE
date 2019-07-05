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

import torch.nn.functional as F

from dss_vae.decoder import get_decoder
from dss_vae.utils.loss_funcs import arc_loss
from dss_vae.utils.loss_funcs import rank_loss
from dss_vae.utils.loss_funcs import tag_loss
from dss_vae.utils.nn_funcs import to_input_dict
from dss_vae.utils.nn_funcs import to_target_word
from .nonauto_gen import NonAutoGenerator


class SyntaxSupervisedNAG(NonAutoGenerator):
    def __init__(self, args, vocab, embed=None):
        super(SyntaxSupervisedNAG, self).__init__(args, vocab, embed=embed,
                                                  name="Syntax Supervised Non-Autoregressive GEN")
        self.decoder = get_decoder(
            args=args,
            input_dim=self.encoder.out_dim,
            vocab_size=len(self.vocab.tgt),
            model="syn-sup",
            pad=self.vocab.tgt.pad_id,
            arc_size=len(self.vocab.arc) if args.use_arc else -1
        )

    def decode(self, encoder_output, ret_syn=False):
        return self.decoder.forward(encoder_output=encoder_output, ret_syn=ret_syn)

    def forward(self, seqs_x, seqs_y=None, x_length=None, to_word=False, log_prob=True):
        enc_out = self.encode(seqs_x, seqs_length=x_length)
        ret = self.decode(enc_out, ret_syn=not to_word)
        if log_prob:
            ret['word'] = F.log_softmax(ret['word'], dim=-1)
        if to_word:
            ret["pred"] = to_target_word(ret['word'], vocab=self.vocab.tgt)
        return ret

    def score(self, examples, to_word=False, **kwargs):
        args = self.args
        if not isinstance(examples, list):
            examples = [examples]

        input_dict = to_input_dict(
            examples=examples,
            vocab=self.vocab,
            max_tgt_len=self.max_len,
            cuda=args.cuda,
            training=self.training,
            use_tgt=True,
            use_tag=args.use_arc,
            use_dst=args.use_dst
        )
        enc_out = self.encode(seqs_x=input_dict['src'], seqs_length=input_dict['src_len'])
        ret = self.decode(enc_out, ret_syn=True)

        pos_word_score = self.decoder.scoring(raw_score=ret['word'], tgt_var=input_dict['tgt'])
        pos_syn_score = self.syntax_score(ret, input_dict, use_arc=args.use_arc, use_dst=args.use_dst)
        sum_score = pos_word_score + pos_syn_score
        return sum_score

    def syntax_score(self, ret, input_dict, use_arc=False, use_dst=True):
        sum_loss = 0
        if use_dst:
            dst_mask = (input_dict['dst'] > 0).float()
            pred_dst = ret['dst']
            loss_dst = -rank_loss(inputs=pred_dst, target=input_dict['dst'], mask=dst_mask)

            sum_loss += loss_dst
        if use_arc:
            arc = input_dict['arc']
            tag = input_dict['tag']
            pred_arc = ret['arc']
            pred_tag = ret['tag']
            loss_arc = -arc_loss(pred_arc, arc.view(-1))
            loss_tag = -tag_loss(pred_tag, tag.view(-1))
            sum_loss += loss_arc
            sum_loss += loss_tag

        return sum_loss * self.args.syn_weight

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
            use_tgt=False,
            use_tag=False,
            use_dst=False
        )
        predict = self.forward(seqs_x=input_dict['src'], x_length=input_dict['src_len'], to_word=to_word)
        return predict['pred']
