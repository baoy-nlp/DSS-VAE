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
# Time 2019-01-28 18:02
from __future__ import division

import os

from .base_evaluator import BaseEvaluator
from .bleu_scorer import BleuScoreMetric
from .evaluation import new_evaluate as evaluate


class ParaphraseEvaluator(BaseEvaluator):
    def __init__(self, model, eval_set, sort_key, eval_tgt, out_dir="./out", batch_size=20, write_down=False, **kwargs):
        super().__init__(model, eval_set, out_dir, batch_size)
        self.write_down = write_down
        self.sort_key = sort_key
        self.eval_tgt = eval_tgt
        self.score_item = "BLEU"

    def __call__(self, eval_desc="para"):
        """
        Args:
            eval_desc:

        Returns: eval the target bleu and origin bleu for paraphrase model

        """
        training = self.model.training
        self.model.eval()
        eval_results = evaluate(examples=self.eval_set, model=self.model, sort_key=self.sort_key,
                                eval_tgt=self.eval_tgt,
                                batch_size=self.batch_size,
                                out_dir=os.path.join(self.out_dir, eval_desc) if self.write_down is not None else None)
        origin_bleu = BleuScoreMetric.evaluate_file(pred_file=eval_results['pred_file'],
                                                    gold_files=eval_results['input_file'])
        self.model.training = training
        return {
            'BLEU': eval_results['accuracy'],
            'Origin BLEU': origin_bleu,
            'EVAL TIME': eval_results['use_time'],
            "EVAL SPEED": len(self.eval_set) / eval_results['use_time']
        }
