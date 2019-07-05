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
# Time 2019-01-28 15:02
from __future__ import division

import os

from dss_vae.structs.dataset import Dataset
from .evaluation import evaluate


class BaseEvaluator(object):
    """
    Set a evaluator for each corresponding models
    """

    def __init__(self, model, eval_set, out_dir="./out", batch_size=20):
        self.model = model
        self.eval_set = eval_set
        self.batch_size = batch_size
        self.out_dir = out_dir

        # check for data directory
        self.dir_check()

    def dir_check(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=False)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class UniversalBleuEvaluator(BaseEvaluator):
    def __init__(self, model, eval_set, sort_key, eval_tgt, out_dir="./out", batch_size=20, write_down=False, **kwargs):
        super().__init__(model, eval_set, out_dir, batch_size)
        self.write_down = write_down
        self.sort_key = sort_key
        self.eval_tgt = eval_tgt
        self.score_item = "BLEU"

    def __call__(self, eval_desc="bleu"):
        training = self.model.training
        self.model.eval()
        if isinstance(self.eval_set, Dataset):
            examples = self.eval_set.examples
        elif isinstance(self.eval_set, list):
            examples = self.eval_set
        else:
            raise RuntimeError("TYPE ERROR")

        eval_results = evaluate(
            examples=examples, model=self.model, sort_key=self.sort_key,
            eval_tgt=self.eval_tgt,
            batch_size=self.batch_size,
            out_dir=os.path.join(self.out_dir, eval_desc) if self.write_down is not None else None)
        self.model.training = training
        return {
            'BLEU': eval_results['accuracy'],
            'EVAL TIME': eval_results['use_time'],
            "EVAL SPEED": len(self.eval_set) / eval_results['use_time']
        }
