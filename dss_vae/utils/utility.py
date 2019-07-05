from __future__ import print_function

import errno
import sys
from os import makedirs


def make_sure_path_exists(path):
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def eprint(*args, **kwargs):
    print(args)


def combine_files(fids, out, tb):
    print('%d files...' % len(fids))
    total_sentence = 0
    for n, file in enumerate(fids):
        if n % 10 == 0 or n == len(fids) - 1:
            print("%c%.2f%%\r" % (13, (n + 1) / float(len(fids)) * 100), end='')
        sents = tb.parsed_sents(file)
        for s in sents:
            out.write(s.pformat(margin=sys.maxsize))
            out.write(u'\n')
            total_sentence += 1
    print()
    print('%d sentences.' % total_sentence)
    print()


def write_docs(fname, docs):
    with open(fname, 'w') as f:
        for doc in docs:
            f.write(str(doc))
            f.write('\n')


def load_docs(fname):
    res = []
    with open(fname, 'r') as data_file:
        for line in data_file:
            line_res = line.strip("\n")
            res.append(line_res)
    return res


class PostProcess(object):
    def __init__(self, sos, eos, tgt_vocab, src_vocab, src_pad, tgt_pad):
        self.sos = sos
        self.eos = eos
        self.tgt = tgt_vocab
        self.src = src_vocab
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad

    def extract_single_source(self, source):
        process = []
        for tok in source:
            if tok == self.src_pad:
                pass
            else:
                process.append(self.src.itos[tok])
        return " ".join(process)

    def extract_single_target(self, target):
        process = []
        for tok in target:
            if tok == self.sos or tok == self.tgt_pad:
                pass
            elif tok == self.eos:
                break
            else:
                process.append(self.tgt.itos[tok])

        return " ".join(process)
