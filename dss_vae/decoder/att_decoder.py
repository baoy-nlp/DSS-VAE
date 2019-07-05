# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dss_vae.networks.basic import BottleLinear as Linear
from dss_vae.networks.embeddings import Embeddings
from dss_vae.networks.sublayers import ConditionAttBlock
from .base_decoder import BaseDecoder


def get_attn_causal_mask(seq):
    """ Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    """
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class TransformerDecoder(nn.Module, BaseDecoder):
    """
    A decoder model with self attention mechanism.
    """

    def __init__(self, vocab_size, n_layers=6, n_head=8,
                 input_size=512, hidden_size=512, inner_hidden=1024, dim_per_head=None,
                 share_proj_weight=True,
                 share_embed_weight=None,
                 embed_dropout=0.1, block_dropout=0.1, pad=0
                 ):

        super(TransformerDecoder, self).__init__()
        self.pad = pad
        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = hidden_size
        if share_embed_weight is None:
            self.embeddings = Embeddings(vocab_size, input_size,
                                         dropout=embed_dropout, add_position_embedding=True, padding_idx=pad)
        else:
            self.embeddings = share_embed_weight

        self.block_stack = nn.ModuleList([
            ConditionAttBlock(d_model=hidden_size, d_inner_hid=inner_hidden, n_head=n_head, dropout=block_dropout,
                              dim_per_head=dim_per_head)
            for _ in range(n_layers)])

        self.out_layer_norm = nn.LayerNorm(hidden_size)
        self._dim_per_head = dim_per_head

        if share_proj_weight:
            self.generator = Generator(n_words=vocab_size,
                                       hidden_size=input_size,
                                       shared_weight=self.embeddings.embeddings.weight,
                                       padding_idx=self.pad)
        else:
            self.generator = Generator(n_words=vocab_size, hidden_size=input_size, padding_idx=0)

    @property
    def dim_per_head(self):
        if self._dim_per_head is None:
            return self.d_model // self.n_head
        else:
            return self._dim_per_head

    def forward(self, tgt_seq, enc_output, enc_mask, enc_attn_caches=None, self_attn_caches=None):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len
        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_seq)

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(self.pad).unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = emb
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache = self.block_stack[i](
                output,
                enc_output,
                dec_slf_attn_mask,
                dec_enc_attn_mask,
                enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None
            )

            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        output = self.out_layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches

    def decode(self, **kwargs):
        pass

    def score(self, inputs, encoder_outputs):
        pass


class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):
        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.proj = Linear(self.hidden_size, self.n_words, bias=False)
        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

    def _pad_2d(self, x):

        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, inputs, log_probs=True):
        """
        input == > Linear == > LogSoftmax
        """

        logits = self.proj(inputs)

        logits = self._pad_2d(logits)

        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
