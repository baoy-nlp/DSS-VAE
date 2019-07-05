from collections import Counter
from itertools import chain

import numpy as np


class Corpus(object):
    def __init__(self, examples):
        self.examples = examples

    @property
    def avg_length(self):
        len_sum = sum(len(e) for e in self.examples)
        sent_number = len(self.examples)
        return len_sum / sent_number

    @property
    def unigram_freq(self):
        word_freq = Counter(chain(*self.examples))
        return word_freq

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
