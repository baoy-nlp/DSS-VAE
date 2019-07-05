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

import time

from dss_vae.utils.model_utils import update_tracker
from .base_evaluator import BaseEvaluator
from .tools import batchify_examples


class ReorderEvaluator(BaseEvaluator):
    def __init__(self, model, eval_set, sort_key, eval_tgt, out_dir="./out", batch_size=20, write_down=False, **kwargs):
        super().__init__(model, eval_set, out_dir, batch_size)
        self.write_down = write_down
        self.sort_key = sort_key
        self.eval_tgt = eval_tgt
        # if 'dev_item' in model.args and model.args.dev_item is not None:
        #     self.score_item = model.args.dev_item
        # else:
        if model.args.pos_pred == 1:
            self.score_item = "ACC"  # absolute match
        if model.args.pos_pred == 2:
            if model.args.pos_mse:
                self.score_item = "ACC"
            if model.args.pos_rank:
                self.score_item = "Binary_ACC"
            if model.args.pos_dst:
                self.score_item = "Relative_ACC"

    def __call__(self, eval_desc="reorder"):
        """
        Args:
            eval_desc:

        Returns: eval the target bleu and origin bleu for paraphrase model

        """
        training = self.model.training
        self.model.eval()
        dev_log_dict = {}
        inp_examples, inp_ids = batchify_examples(examples=self.eval_set, sort_key=self.sort_key,
                                                  batch_size=self.batch_size)
        eval_start = time.time()
        for batch_examples in inp_examples:
            batch_ret = self.model.get_loss(examples=batch_examples, return_enc_state=False, train_iter=-1)
            dev_log_dict = update_tracker(batch_ret, dev_log_dict)
        use_time = time.time() - eval_start
        self.model.training = training
        return {
            'ACC': dev_log_dict['Acc'].mean().item(),
            'Relax_ACC': dev_log_dict['relax_correct'] * 100.0 / dev_log_dict['count'],
            'Relative_ACC': dev_log_dict['relative_correct'] * 100.0 / dev_log_dict['relative_count'],
            'Binary_ACC': dev_log_dict['binary_correct'] * 100.0 / dev_log_dict['relative_count'],
            'EVAL TIME': use_time,
            "EVAL SPEED": len(self.eval_set) / use_time
        }
