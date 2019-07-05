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
# Time 2019-01-30 16:20

import torch.nn as nn

from dss_vae.networks.sublayers import ConditionAttBlock
from dss_vae.networks.sublayers import MultiAttGroup
from .nonauto_decoder import NonAutoDecoder


class SelfAttNAD(NonAutoDecoder):
    def __init__(self,
                 vocab_size,
                 max_len,
                 hidden_dim,
                 n_layers,
                 n_head,
                 inner_dim,
                 block_dropout,
                 dim_per_head,
                 out_dropout=0.1,
                 pad=0,
                 use_cuda=True,
                 **kwargs
                 ):
        super(SelfAttNAD, self).__init__(
            vocab_size=vocab_size,
            max_len=max_len,
            hidden_dim=hidden_dim,
            out_dropout=out_dropout,
            pad=pad,
        )
        self.self_att_decoder = MultiAttGroup(
            n_layers=n_layers,
            hidden_size=hidden_dim,
            inner_hidden=inner_dim,
            n_head=n_head,
            block_dropout=block_dropout,
            dim_per_head=dim_per_head,
        )
        self.use_cuda = use_cuda

    def forward(self, decoder_input, **kwargs):
        hidden = self.self_att_decoder.forward(decoder_input)
        is_log_prob = False
        if 'log_prob' in kwargs:
            is_log_prob = kwargs['log_prob']
        word_prob = self.word_predictor.forward(hidden, log_probs=is_log_prob)
        return {"word": word_prob}


class CondAttNAD(NonAutoDecoder):
    """
    with src attention while decoding
    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 hidden_dim,
                 n_layers,
                 n_head,
                 inner_dim,
                 block_dropout=0.1,
                 dim_per_head=None,
                 out_dropout=0.1,
                 pad=0,
                 use_cuda=True,
                 **kwargs
                 ):
        super(CondAttNAD, self).__init__(
            vocab_size=vocab_size,
            max_len=max_len,
            hidden_dim=hidden_dim,
            out_dropout=out_dropout,
            pad=pad,
        )
        self.cond_att_decoder = nn.ModuleList([
            ConditionAttBlock(d_model=hidden_dim, d_inner_hid=inner_dim, n_head=n_head, dropout=block_dropout,
                              dim_per_head=dim_per_head)
            for _ in range(n_layers)])
        self.out_layer_norm = nn.LayerNorm(hidden_dim)
        self.num_layers = n_layers
        self.use_cuda = use_cuda

        self.bow_predictor = None
        if 'bow' in kwargs and kwargs['bow']:
            if 'share_predictor' in kwargs and kwargs['share_predictor']:
                self.bow_predictor = self.word_predictor
            else:
                self.bow_predictor = nn.Sequential(
                    nn.Dropout(out_dropout),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.ReLU(),
                    nn.Dropout(out_dropout),
                    nn.Linear(hidden_dim, vocab_size)
                )

    def forward(self, decoder_input, encoder_output, enc_mask, dec_mask=None, enc_attn_caches=None,
                self_attn_caches=None, **kwargs):
        batch_size, tgt_len, hidden_size = decoder_input.size()
        query_len = tgt_len
        src_len = encoder_output.size(1)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)
        output = decoder_input
        layer_outputs = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache = self.cond_att_decoder[i](
                output,
                encoder_output,
                dec_enc_attn_mask=dec_enc_attn_mask,
                slf_attn_mask=dec_mask,
            )
            layer_outputs.append(output)

        # output = self.out_layer_norm(output)
        # is_log_prob = False
        # if 'log_prob' in kwargs:
        #     is_log_prob = kwargs['log_prob']
        word_prob = self.word_predictor.forward(output)
        if self.bow_predictor is not None and self.training:
            layer_probs = []
            for layer_out in layer_outputs:
                layer_probs.append(self.bow_predictor.forward(layer_out))
            return {
                "word": word_prob,
                # 'layer_out': layer_outputs,
                'layer_bow_probs': layer_probs
            }
        else:
            return {
                "word": word_prob,
                # 'layer_out': layer_outputs,
            }
