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

import torch.nn as nn

from dss_vae.networks.embeddings import Embeddings
from dss_vae.networks.sublayers import MultiAttGroup


class TransformerEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 n_layers=6,
                 n_head=8,
                 input_size=512,
                 hidden_size=512,
                 inner_hidden=1024,
                 embed_dropout=0.1,
                 block_dropout=0.1,
                 dim_per_head=None,
                 pad=0,
                 **kwargs
                 ):
        super().__init__()
        self.pad_id = pad
        self.embeddings = Embeddings(num_embeddings=vocab_size,
                                     embedding_dim=input_size,
                                     dropout=embed_dropout,
                                     add_position_embedding=True,
                                     padding_idx=self.pad_id
                                     )

        self.self_att_encoder = MultiAttGroup(
            n_layers,
            hidden_size,
            inner_hidden,
            n_head,
            block_dropout,
            dim_per_head
        )
        self.hidden_size = hidden_size

    def reset_embed(self, share_embed):
        self.embeddings = share_embed

    @property
    def out_dim(self):
        return self.hidden_size

    def forward(self, src_seq):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.embeddings(src_seq)

        enc_mask = src_seq.detach().eq(self.pad_id)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = self.self_att_encoder(emb, enc_slf_attn_mask)

        # return out, enc_mask
        return {
            "out": out,
            "mask": enc_mask,
        }
