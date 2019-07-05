import torch.nn as nn

from dss_vae.networks.rnn_base import RNNBase


class RNNEncoder(RNNBase):
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
        super(RNNEncoder, self).__init__(vocab_size, max_len, input_size, hidden_size,
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
        if self.variable_lengths and input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths and input_lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
