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
# Time 2019-01-23 09:28

import numpy as np
import torch


def unk_replace(input_sequence, dropoutr, vocab):
    if dropoutr > 0.:
        prob = torch.rand(input_sequence.size())
        if torch.cuda.is_available():
            prob = prob.cuda()
        prob[(input_sequence.data - vocab.sos_id) * (input_sequence.data - vocab.pad_id) * (
                input_sequence.data - vocab.eos_id) == 0] = 1
        decoder_input_sequence = input_sequence.clone()
        decoder_input_sequence[prob < dropoutr] = vocab.unk_id
        return decoder_input_sequence
    return input_sequence


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == "fixed":
        return 1.0
    elif anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'sigmoid':
        return float(1 / (1 + np.exp(0.001 * (x0 - step))))
    elif anneal_function == 'negative-sigmoid':
        return float(1 / (1 + np.exp(-0.001 * (x0 - step))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


def wd_anneal_function(unk_max, anneal_function, step, k, x0):
    return unk_max * kl_anneal_function(anneal_function, step, k, x0)
