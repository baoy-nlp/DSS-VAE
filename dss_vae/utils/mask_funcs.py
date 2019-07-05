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
# Time 2019-01-23 09:30

import numpy as np
import torch

_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)


def sequence_mask(length_array, max_len=-1, cuda=False):
    max_len = max_len if max_len == -1 else length_array[0]
    batch_size = len(length_array)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def distance_mask(distance, pad=0):
    """
    Args:
        distance: syntax distance with [batch,max_len], just could see which value is lower than me.
    Return:
        value matrix: [batch,max_len,]
    """
    pass


def mask_scores(scores, beam_mask, EOS=2):
    """
    Mask scores of next step according to beam mask.
    Args:
        scores (torch.Tensor): Scores of next tokens with shape [batch_size, beam_size, vocab_size].
            Smaller should be better (usually negative log-probs).
        beam_mask (torch.Tensor): Mask of beam. 1.0 means not closed and vice verse. The shape is
            [batch_size, beam_size]

    Returns:
        Masked scores of next tokens.
    """
    vocab_size = scores.size(-1)

    finished_row = beam_mask.new(vocab_size, ).zero_() + float(_FLOAT32_INF)

    # If beam finished, only PAD could be generated afterwards.
    finished_row[EOS] = 0.0

    scores = scores * beam_mask.unsqueeze(2) + torch.matmul((1.0 - beam_mask).unsqueeze(2), finished_row.unsqueeze(0))

    return scores


def relative_relation_mask(pos_pred, pos_ref, pos_mask=None):
    """

    Args:
        pos_pred: [batch,seq_len]
        pos_ref:  [batch,seq_len]
        pos_mask: [batch,seq_len]

    Returns:

    """
    batch_size, seq_len = pos_mask.size()
    pos_mask = pos_mask.long()
    mask_tensor = pos_mask.view(batch_size, seq_len, 1) * pos_mask.view(batch_size, 1, seq_len)
    # [batch,seq_len,seq_len]
    pred_relative = pos_pred.unsqueeze(-2) - pos_pred.unsqueeze(-1)
    ref_relative = pos_ref.unsqueeze(-2) - pos_ref.unsqueeze(-1)

    return pred_relative, ref_relative, mask_tensor
