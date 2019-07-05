# coding=utf-8
"""
store the instance from training set or console inputs.
"""
import pickle as pickle

import numpy as np
from nltk.tree import Tree


def to_example(words, token_split=" "):
    return PlainExample(
        src=words.split(token_split),
        tgt=None,
    )


class CachedProperty(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    @staticmethod
    def from_raw_file(file_path, e_type="Plain"):
        example_dict = {
            "Plain": PlainExample,
            "PTB": PTBExample,
            "NAG": PlainExample,
            "SyntaxVAE": PlainExample,
            "SyntaxGEN": SyntaxGenExample,
        }
        with open(file_path, "r") as f:
            examples = [example_dict[e_type].parse_raw(line) for line in f]
        return Dataset(examples)

    @staticmethod
    def from_list(raw_list):
        examples = [PlainExample.parse_raw(line) for line in raw_list]
        return Dataset(examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            batch_examples.sort(key=lambda e: -len(e))

            yield batch_examples

    def __sizeof__(self):
        return len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class PlainExample(object):
    def __init__(self, src, tgt, idx=0, meta=None, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            self.__setattr__(key, item)

            self.entries.append(key)
        self.src = src
        self.tgt = tgt

        self.idx = idx
        self.meta = meta

    @staticmethod
    def parse_raw(raw_line, field_split='\t', token_split=' '):
        line_items = raw_line.strip().split(field_split)
        if len(line_items) <= 1:
            return PlainExample(
                src=line_items[0].split(token_split),
                tgt=None,
            )
        else:
            return PlainExample(
                src=line_items[0].split(token_split),
                tgt=line_items[1].split(token_split)
            )

    @staticmethod
    def parse_for_control(raw_line, field_split="\t", token_split=" "):
        line_items = raw_line.strip().split(field_split)
        return PlainExample(src=line_items[0].split(token_split), tgt=None), PlainExample(
            src=line_items[1].split(token_split), tgt=None
        )

    def __len__(self):
        return len(self.src)


class PTBExample(PlainExample):
    def __init__(self, src, stag, tags, arcs, distance, tgt=None, idx=0, meta=None):
        super().__init__(src, tgt, idx, meta, stag=stag, tag=tags, arc=arcs, dst=distance)
        self.stag = stag
        self.tag = tags
        self.arc = arcs
        self.dst = distance

    @staticmethod
    def parse_raw(raw_line, field_split='\t', token_split=' '):
        from .distance_tree import tree2list, get_distance
        tree = Tree.fromstring(raw_line)
        if tree.label() in ("TOP", "ROOT"):
            assert len(tree) == 1
            tree = tree[0]
        words, stags = zip(*tree.pos())
        linear_trees, arcs, tags = tree2list(tree)

        if type(linear_trees) is str:
            linear_trees = [linear_trees]
        distances_sent, _ = get_distance(linear_trees)
        distances_sent = [0] + distances_sent + [0]

        return PTBExample(
            src=list(words),
            stag=list(stags),
            tags=['<unk>'] + tags + ['<unk>'],
            arcs=['<unk>'] + arcs + ['<unk>'],
            distance=distances_sent
        )

    def __len__(self):
        return super().__len__()


class SyntaxGenExample(PlainExample):
    def __init__(self, src, stag=None, tag=None, arc=None, dst=None, tgt=None, idx=0, meta=None):
        super(SyntaxGenExample, self).__init__(src, tgt, idx, meta, stag=stag, tag=tag, arc=arc, dst=dst)
        self.stag = stag
        self.tag = tag
        self.arc = arc
        self.dst = dst

    @staticmethod
    def parse_raw(raw_line, field_split='\t', token_split=' '):
        from .distance_tree import tree2list, get_distance

        def analysis_tree(raw_tree, need_distance=True):
            out_tree = Tree.fromstring(raw_tree)
            if out_tree.label() in ("TOP", "ROOT"):
                assert len(out_tree) == 1
                out_tree = out_tree[0]
            words, stags = zip(*out_tree.pos())
            if not need_distance:
                return {
                    "word": list(words),
                    "stag": list(stags)
                }
            linear_trees, arcs, tags = tree2list(out_tree)
            if type(linear_trees) is str:
                linear_trees = [linear_trees]
            distances_sent, _ = get_distance(linear_trees)
            distances_sent = [0] + distances_sent + [0]
            return {
                "word": list(words),
                "stag": list(stags),
                "tag": tags,
                "arc": arcs,
                "dst": distances_sent,
            }

        raw_data = raw_line.split(field_split)
        src_tree = analysis_tree(raw_data[0], need_distance=False)
        tgt_tree = analysis_tree(raw_data[1])

        return SyntaxGenExample(
            src=src_tree["word"],
            tgt=tgt_tree["word"],
            stag=tgt_tree["stag"],
            tag=tgt_tree["tag"],
            arc=tgt_tree["arc"],
            dst=tgt_tree["dst"],
        )
