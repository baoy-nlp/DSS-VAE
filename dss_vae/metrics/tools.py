# coding=utf-8
from __future__ import print_function

import numpy as np
import torch
# sys.path.append(".")
from nltk.translate import bleu_score

from dss_vae.utils.nn_funcs import word2id, id2word
from dss_vae.structs import Dataset
from dss_vae.structs import PlainExample


# import sys


def write_result(docs, fname):
    with open(fname, "w") as f:
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, list):
                    doc = " ".join(doc)
                f.write(doc)
                f.write("\n")
        elif isinstance(docs, dict):
            for key, val in docs.items():
                if val is not None:
                    f.write("%s:%f" % (
                        key,
                        torch.mean(val) if isinstance(val, torch.Tensor) else val)
                            )
                    f.write("\n")


def write_append_result(docs, fname):
    with open(fname, "a") as f:
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, list):
                    doc = " ".join(doc)
                f.write(doc)
                f.write("\n")
        elif isinstance(docs, dict):
            for key, val in docs.items():
                if val is not None:
                    f.write("%s:%f" % (
                        key,
                        torch.mean(val) if isinstance(val, torch.Tensor) else val)
                            )
                    f.write("\n")


class TagJudge(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk = vocab.unk_id

    def is_tag(self, item):
        idx = self.vocab[item]
        return idx > self.unk


def to_plain(result):
    return [r[0][0] for r in result]


def con_to_plain(decode_results):
    return [" ".join(hyp[0]) for hyp in decode_results]


def predict_to_plain(decode_results):
    return [" ".join(hyp[0][0]) for hyp in decode_results]


def reference_to_plain(decode_results):
    return [" ".join(hyp[0][0][0]) for hyp in decode_results]


def recovery(seqs, vocab, keep_origin=False):
    if keep_origin:
        return [seqs]
    return id2word(word2id(seqs, vocab), vocab)


def recovery_ref(refs, vocab, keep_origin=False):
    if keep_origin:
        return [[e.src] for e in refs]
    return [[recovery(e.src, vocab)] for e in refs]


def get_bleu_score(references, hypothesis):
    """
    Args:
        references: list(list(list(str)))
        # bin: list(bin)
        hypothesis: list(list(list(str)))
        # hypotheses: list(list(str))
    """

    hypothesis = [hyp[0][0] for hyp in hypothesis]

    return 100.0 * bleu_score.corpus_bleu(list_of_references=references, hypotheses=hypothesis)


def prepare_input(examples, sort_key='src', batch_size=None):
    data_set = Dataset(examples)
    new_examples = []
    if batch_size is None:
        batch_size = 50
    for batch_examples in data_set.batch_iter(batch_size, shuffle=False):
        batch_examples.sort(key=lambda e: -len(getattr(e, sort_key)))
        new_examples.append(batch_examples)
    return new_examples


def batchify_examples(examples, sort_key="src", batch_size=None, batch_type="sents", sort=True):
    new_examples = []
    if batch_size is None:
        if batch_type == "sents":
            batch_size = 50 if len(examples) > 50 else len(examples)
        else:
            batch_size = 1000
    if sort:
        origin_ids = sorted(range(len(examples)), key=lambda i: -len(getattr(examples[i], sort_key)))
    else:
        origin_ids = np.arange(len(examples))
    examples = sorted(examples, key=lambda e: -len(getattr(e, sort_key)))
    if batch_type == "sents":
        index_arr = np.arange(len(examples))
        batch_num = int(np.ceil(len(examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [examples[i] for i in batch_ids]
            new_examples.append(batch_examples)
    else:
        cum_len = 0
        cum_batch = []
        for example in examples:
            cur_len = len(getattr(example, sort_key))
            if cur_len + cum_len > batch_size:
                new_examples.append(cum_batch)
                cum_batch = [example]
                cum_len = cur_len
            else:
                cum_batch.append(example)
                cum_len += cur_len
    return new_examples, origin_ids


def prepare_ref(examples, eval_tgt, vocab, keep_origin=True):
    references = [[recovery(e.src, vocab.src, keep_origin)] for e in examples] if eval_tgt == "src" else \
        [[recovery(e.tgt, vocab.tgt, keep_origin)] for e in examples]
    return references


def split_examples(examples):
    new_examples = []
    for e in examples:
        # if len(e.src) < 30:
        new_examples.append(e)
    examples = new_examples
    sem_examples = [PlainExample(src=e.src, tgt=None) for e in examples]
    syn_examples = [PlainExample(src=e.tgt, tgt=None) for e in examples]
    return sem_examples, syn_examples
