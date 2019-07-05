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
from dss_vae.networks.rnn_base import RNNBase
from dss_vae.networks.sublayers import MultiAttGroup


class NonAutoATTEncoder(nn.Module):

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
        # batch_size, src_len = src_seq.size()
        emb = self.embeddings(src_seq)
        enc_mask = src_seq.detach().eq(-1)
        out = self.self_att_encoder(emb)
        # return out, enc_mask
        return {
            "out": out,
            "mask": enc_mask,
        }


class NonAutoRNNEncoder(RNNBase):
    def __init__(self,
                 vocab_size,
                 max_len,
                 input_size,
                 hidden_size,
                 embed_droprate=0,
                 rnn_droprate=0,
                 n_layers=1,
                 bidirectional=False,
                 rnn_cell='gru',
                 variable_lengths=False,
                 embedding=None,
                 update_embedding=True,
                 **kwargs,
                 ):
        super(NonAutoRNNEncoder, self).__init__(vocab_size, max_len, input_size, hidden_size,
                                                embed_droprate, rnn_droprate, n_layers, rnn_cell)
        self.variable_lengths = variable_lengths
        if embedding is not None:
            self.embedding = embedding
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional,
                                 dropout=rnn_droprate)
        self.bidirectional = bidirectional

    @property
    def out_dim(self):
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
