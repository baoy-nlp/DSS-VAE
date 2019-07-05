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

from dss_vae.utils.model_utils import update_tracker
from .base_evaluator import BaseEvaluator
from .bleu_scorer import BleuScoreMetric
from .evaluation import new_evaluate
from .tools import *

"""
Evaluate the F1 of Parser, BLEU for Text Generation.
"""


def get_dev_item(task_type, bleu=-1):
    if task_type == "EnhanceSyntaxVAE":
        return "ELBO"

    if bleu < 50.0:
        return "TGT BLEU"
    else:
        return "TGT_ORI_RATE"


class SyntaxEvaluator(BaseEvaluator):

    def __init__(self, model, eval_set, eval_lists, sort_key, eval_tgt, out_dir="./out", batch_size=20,
                 write_down=False, use_bpe=False, **kwargs):
        super().__init__(model, eval_set, out_dir, batch_size)
        self.eval_file_dict = eval_lists
        self.write_down = write_down
        self.sort_key = sort_key
        self.eval_tgt = eval_tgt
        if model.args.dev_item is None:
            self.score_item = "BLEU"
        else:
            self.score_item = model.args.dev_item
        self.use_bpe = use_bpe
        self.model = model

    def eval_paraphrase(self, eval_desc="paraphrase"):
        model = self.model
        args = self.model.args
        training = model.training
        model.eval()
        eval_results = new_evaluate(
            examples=self.eval_set,
            model=model,
            sort_key=self.sort_key,
            eval_tgt=self.eval_tgt,
            batch_size=self.batch_size,
            out_dir=os.path.join(self.out_dir, eval_desc) if self.write_down is not None else None
        )
        tgt_bleu = eval_results['accuracy']
        ori_bleu = BleuScoreMetric.evaluate_file(
            pred_file=eval_results['pred_file'],
            gold_files=eval_results['input_file']
        )
        model.training = training
        return {
            'TGT BLEU': tgt_bleu,
            'ORI BLEU': ori_bleu,
            'TGT_ORI_RATE': tgt_bleu / ori_bleu,
            'EVAL TIME': eval_results['use_time'],
            "EVAL SPEED": len(self.eval_set) / eval_results['use_time']
        }

    def eval_reconstruct(self, eval_desc="vae-reconsturct"):
        model = self.model
        args = self.model.args
        training = model.training
        model.eval()
        eval_results = new_evaluate(
            examples=self.eval_set,
            model=model,
            sort_key=self.sort_key,
            eval_tgt=self.eval_tgt,
            batch_size=self.batch_size,
            out_dir=os.path.join(self.out_dir, eval_desc) if self.write_down is not None else None
        )
        tgt_bleu = eval_results['accuracy']
        model.training = training
        return {
            'BLEU': tgt_bleu,
            'EVAL TIME': eval_results['use_time'],
            "EVAL SPEED": len(self.eval_set) / eval_results['use_time']
        }

    def eval_elbo(self, eval_desc='vae-elbo', eval_step=None):
        model = self.model
        args = self.model.args
        training = model.training
        model.eval()
        step = eval_step if eval_step is not None else 2 * args.x0
        ret_track = {}

        batch_examples, _ = batchify_examples(
            examples=self.eval_set,
            batch_size=self.batch_size,
            sort=False
        )
        for batch in batch_examples:
            ret_loss = model.get_loss(batch, step)
            ret_track = update_tracker(ret_loss, ret_track)

        if self.write_down:
            write_result(ret_track, fname=os.path.join(self.out_dir, eval_desc + ".score"))
        model.training = training
        return ret_track

    def __call__(self, eval_desc="syntax-vae", step=None, **kwargs):
        """
        Args:
            eval_desc:

        Returns: eval the multi-bleu for machine translation

        """
        args = self.model.args
        ret_track = {}
        if args.task_type == "SyntaxVAE2":
            ret_track = self.eval_elbo()
            rec_ret = self.eval_reconstruct()
            ret_track.update(**rec_ret)
            if args.dev_item is None:
                self.score_item = get_dev_item(task_type=args.task_type)
            else:
                self.score_item = args.dev_item

        elif args.task_type == "SyntaxPara":
            ret_track = self.eval_paraphrase()
            if args.dev_item is None:
                self.score_item = get_dev_item(task_type=args.task_type, bleu=ret_track['TGT BLEU'])
            else:
                self.score_item = args.dev_item
        return ret_track
