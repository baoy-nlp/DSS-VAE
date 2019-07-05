# coding=utf-8
from __future__ import print_function

import os
import time

from .bleu_scorer import BleuScoreMetric
from .tools import batchify_examples
from .tools import predict_to_plain
from .tools import prepare_input
from .tools import prepare_ref
from .tools import write_result


# sys.path.append(".")
# import sys


def decode(examples, model):
    was_training = model.training
    model.eval()
    decode_results = model.predict(examples)
    if was_training:
        model.train()
    return decode_results


def prediction(
        examples,
        model,
        sort_key='src',
        eval_src="src",
        out_dir=None,
        batch_size=None,
        batch_type="sents",
        use_bpe=False,
):
    cum_oracle_acc = 0.0
    if batch_size is None:
        batch_size = 50 if "eval_bs" not in model.args else model.args.eval_bs
    cache_outputs = []
    inp_examples, inp_ids = batchify_examples(examples, sort_key=sort_key, batch_size=batch_size)
    eval_start = time.time()
    for batch_examples in inp_examples:
        pred_result = decode(batch_examples, model)
        cache_outputs.extend(pred_result)
    use_time = time.time() - eval_start
    pred_examples = [None] * len(cache_outputs)
    for new_id, ori_ids in enumerate(inp_ids):
        pred_examples[ori_ids] = cache_outputs[new_id]
    eval_result = {
        "predict": pred_examples,
        "oracle_accuracy": cum_oracle_acc,
        "use_time": use_time
    }

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        out_dir = "./"

    input_out = os.path.join(out_dir, "input.txt")
    pred_out = os.path.join(out_dir, "pred.txt")

    source = prepare_ref(examples, eval_src, model.vocab)
    inputs = predict_to_plain(source)

    pred = predict_to_plain(eval_result['predict'])
    write_result(inputs, fname=os.path.join(out_dir, 'input.txt'))
    write_result(pred, fname=os.path.join(out_dir, "pred.txt"))

    eval_result['input_file'] = input_out
    eval_result['pred_file'] = pred_out

    return eval_result


def new_evaluate(
        examples,
        model,
        sort_key='src',
        eval_src='src',
        eval_tgt='src',
        ret_dec_result=False,
        batch_size=None,
        out_dir=None
):
    eval_result = prediction(
        examples,
        model,
        sort_key,
        eval_src,
        out_dir,
        batch_size
    )
    references = prepare_ref(examples, eval_tgt, model.vocab)
    eval_result['reference'] = references
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        out_dir = "./"
    gold_out = os.path.join(out_dir, "gold.txt")
    gold = predict_to_plain(eval_result['reference'])
    write_result(gold, fname=os.path.join(out_dir, "gold.txt"))

    eval_result['gold_file'] = gold_out

    eval_result['accuracy'] = BleuScoreMetric.evaluate_file(pred_file=eval_result['pred_file'], gold_files=gold_out,
                                                            etype="corpus")

    if ret_dec_result:
        return eval_result, eval_result['predict']
    else:
        return eval_result


def evaluate(
        examples, model,
        sort_key='src',
        eval_src='src',
        eval_tgt='src',
        return_decode_result=False,
        batch_size=None,
        out_dir=None,
):
    cum_oracle_acc = 0.0
    pred_examples = []

    if batch_size is None:
        batch_size = 50 if "eval_bs" not in model.args else model.args.eval_bs
    inp_examples = prepare_input(
        examples,
        sort_key=sort_key,
        batch_size=batch_size
    )
    ref_examples = []
    eval_start = time.time()
    for batch_examples in inp_examples:
        ref_examples.extend(batch_examples)
        pred_result = decode(batch_examples, model)
        pred_examples.extend(pred_result)
    # references = [[recovery(e.src, model.vocab.src)] for e in inp_examples] if eval_tgt == "src" else \
    #     [[recovery(e.tgt, model.vocab.tgt)] for e in inp_examples]
    use_time = time.time() - eval_start
    references = prepare_ref(ref_examples, eval_tgt, model.vocab)
    # acc = get_bleu_score(references, pred_examples)
    eval_result = {
        'reference': references,
        'predict': pred_examples,
        'oracle_accuracy': cum_oracle_acc,
        'use_time': use_time
    }

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        out_dir = "./"

    input_out = os.path.join(out_dir, "input.txt")
    pred_out = os.path.join(out_dir, "pred.txt")
    gold_out = os.path.join(out_dir, "gold.txt")

    source = prepare_ref(ref_examples, eval_src, model.vocab)
    pred = predict_to_plain(eval_result['predict'])
    inputs = predict_to_plain(source)
    gold = predict_to_plain(eval_result['reference'])
    # gold = con_to_plain(eval_result['reference'])
    write_result(inputs, fname=os.path.join(out_dir, 'input.txt'))
    write_result(pred, fname=os.path.join(out_dir, "pred.txt"))
    write_result(gold, fname=os.path.join(out_dir, "gold.txt"))

    eval_result['input_file'] = input_out
    eval_result['pred_file'] = pred_out
    eval_result['gold_file'] = gold_out

    eval_result['accuracy'] = BleuScoreMetric.evaluate_file(pred_file=pred_out, gold_files=gold_out, etype="corpus")

    if return_decode_result:
        return eval_result, pred_examples
    else:
        return eval_result
