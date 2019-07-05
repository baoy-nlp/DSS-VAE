""" A base class for RNN. """
import torch.nn as nn

from .embeddings import Embeddings


class RNNBase(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        embed_droprate (float): dropout probability for the input sequence
        rnn_droprate (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    module_cell = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(self, vocab_size, max_len, input_size, hidden_size, embed_droprate, rnn_droprate, n_layers, rnn_cell,
                 **kwargs):
        super(RNNBase, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_droprate = rnn_droprate if n_layers > 1 else 0.0
        self.embedding = Embeddings(
            num_embeddings=self.vocab_size,
            embedding_dim=self.input_size,
            dropout=embed_droprate,
            add_position_embedding=False,
            padding_idx=None
        )
        self.rnn_cell_select = rnn_cell

    @property
    def rnn_cell(self):
        if self.rnn_cell_select.lower() in RNNBase.module_cell:
            return RNNBase.module_cell[self.rnn_cell_select.lower()]
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(self.rnn_cell_select))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
