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

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonAutoDecoder(nn.Module):
    """
    Mapping and Predicting: including a base mapper and a to-word predictor

    Input:
        z: batch_size, hidden
    Constructing:
        Inv(R): max_len X max_len
        S:  max_len X hidden_dim
    Modules:
        Matrix Mapper: z -> Inv(R)
        Matrix Mapper: z -> S
        Predictor: Inv(R)*S -> Tgt_V
    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 hidden_dim,
                 out_dropout=0.1,
                 pad=0,
                 **kwargs
                 ):
        super(NonAutoDecoder, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.word_predictor = nn.Sequential(
            nn.Dropout(out_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(out_dropout),
            nn.Linear(hidden_dim, vocab_size, bias=False)
        )
        self.pad_id = pad
        # self.word_predictor = Generator(
        #     n_words=vocab_size,
        #     hidden_size=hidden_dim,
        #     padding_idx=self.pad_id
        # )

    def forward(self, **kwargs):
        raise NotImplementedError

    def scoring(self, raw_score, tgt_var, rm_pad=False, **kwargs):
        """
        Args:
            raw_score: [batch_size, seq_len, vocab_size]
            tgt_var: [seq_len, batch_size]
            rm_pad
        """
        batch_size = raw_score.size(0)
        seq_len = raw_score.size(1)
        vocab_size = raw_score.size(-1)

        if tgt_var.size(0) == batch_size:
            tgt_var = tgt_var.contiguous()[:, 0:seq_len]
            tgt_var = tgt_var.contiguous().transpose(1, 0)

        raw_score = raw_score.contiguous().transpose(1, 0).contiguous()
        log_probs = F.log_softmax(raw_score.view(-1, vocab_size).contiguous(), dim=-1)
        flattened_tgt_var = tgt_var.contiguous().view(-1)
        sent_log_probs = torch.gather(log_probs, 1, flattened_tgt_var.unsqueeze(1)).squeeze(1)
        if not rm_pad:
            sent_log_probs = sent_log_probs * (1. - torch.eq(flattened_tgt_var, self.pad_id).float())
        sent_log_probs = sent_log_probs.view(-1, batch_size).sum(dim=0)

        return sent_log_probs
        # batch_size = raw_score.size(0)
        # if tgt_var.size(0) != batch_size:  # convert to [batch_size, seq_len]
        #     tgt_var = tgt_var.contiguous().transpose(1, 0)
        # log_prob = kwargs['log_prob'] if 'log_prob' in kwargs else False
        # pad = self.pad_id if rm_pad else -1
        # return -advance_loss(logits=raw_score, tgt_var=tgt_var, pad=pad, log_prob=log_prob, **kwargs)
