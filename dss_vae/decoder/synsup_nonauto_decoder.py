# MIT License

# Copyright (c) 2018 the NJUNLP.

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

from dss_vae.networks.sublayers import MultiAttGroup
from dss_vae.utils.nn_funcs import positional_encodings_like
from .nonauto_decoder import NonAutoDecoder


class SynSupervisedNAD(NonAutoDecoder):
    """
    set a syntax distance as supervision task.
    decoder is same as AttNADecoder
    with a extra input as syntax prediction.
    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 input_dim,
                 hidden_dim,
                 n_layers,
                 n_head,
                 inner_dim,
                 block_dropout,
                 dim_per_head,
                 mapper_dropout=0.1,
                 out_dropout=0.1,
                 pad=0,
                 use_cuda=True,
                 arc_size=-1,
                 **kwargs
                 ):
        super(SynSupervisedNAD, self).__init__(
            vocab_size,
            max_len,
            input_dim,
            hidden_dim,
            mapper_dropout,
            out_dropout,
            pad,
            map_type=kwargs['map_type'],
            head_num=kwargs['head_num']
        )
        self.use_cuda = use_cuda
        self.window_size = 2
        self.arc_size = arc_size
        hidden_dim = self.enc_map_dec.out_dim
        self.positional_decoder = MultiAttGroup(
            n_layers,
            hidden_dim,
            inner_dim,
            n_head,
            block_dropout,
            dim_per_head,
        )
        self.dst_hid_num = int(n_layers / 2) - 1
        self.tag_hid_num = 0

        self.dst_decoder = nn.Sequential(
            nn.Dropout(block_dropout),
            nn.Conv1d(hidden_dim, hidden_dim, self.window_size),
            nn.ReLU(),
        )
        self.dst_predictor = nn.Sequential(
            nn.Dropout(block_dropout),
            nn.Linear(hidden_dim, 1)
        )

        if arc_size != -1:
            self.terminal = nn.Sequential(
                nn.Dropout(block_dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.non_terminal = nn.Sequential(
                nn.Dropout(block_dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            self.arc_predictor = nn.Sequential(
                nn.Dropout(block_dropout),
                nn.Linear(hidden_dim, arc_size),
            )

    def predict_syntax(self, enc_list):
        """

        Args:
            enc_list: encode output after syntax_layer <batch,max_len,hidden>
        Returns:

        """
        dst_hidden = enc_list[self.dst_hid_num]
        tag_hidden = enc_list[self.tag_hid_num]
        dst_input = self.dst_decoder.forward(dst_hidden.permute(0, 2, 1)).permute(0, 2, 1)

        batch_size = dst_input.size(0)
        dst_score = self.dst_predictor.forward(dst_input).view(batch_size, -1)

        if self.arc_size != -1:
            terminal = self.terminal.forward(tag_hidden)
            non_terminal = self.non_terminal.forward(dst_hidden)
            tag_score = self.arc_predictor.forward(terminal).view(-1, self.arc_size)
            arc_score = self.arc_predictor.forward(non_terminal).view(-1, self.arc_size)
        else:
            tag_score = None
            arc_score = None

        return {
            "dst": dst_score,
            "tag": tag_score,
            "arc": arc_score,
        }

    def computing(self, encoder_hidden, encoder_output, ret_syn=False):
        """

        Args:
            encoder_hidden:
            encoder_output:
            ret_syn: whether or not predict syntax
        Returns:

        """
        dec_inputs = self.get_dec_input(encoder_output)
        position_encoding = positional_encodings_like(dec_inputs, use_cuda=self.use_cuda)

        positional_dec_inputs = dec_inputs + position_encoding
        dec_hidden_list = self.positional_decoder.forward(positional_dec_inputs, ret_list=True)

        word_score = self.word_predictor.forward(dec_hidden_list[-1])
        if ret_syn:
            ret = self.predict_syntax(dec_hidden_list)
            ret['word'] = word_score
            return ret
        return {
            "word": word_score
        }
