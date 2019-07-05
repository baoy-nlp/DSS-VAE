"""
Recursive representation of a phrase-structure parse tree 
    for natural language sentences.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict


class PhraseTree(object):
    puncs = [",", ".", ":", "``", "''", "PU"]  ## (COLLINS.prm)

    def __init__(
            self,
            symbol=None,
            children=[],
            sentence=[],
            leaf=None,
    ):
        self.symbol = symbol  # label at top node
        self.children = children  # list of PhraseTree objects
        self.sentence = sentence
        self.leaf = leaf  # word at bottom level else None

        self._str = None

    def __len__(self):
        return len(self.sentence)

    def __str__(self):
        if self._str is None:
            if self.leaf is None:
                childstr = ' '.join(str(c) for c in self.children)
                self._str = '({} {})'.format(self.symbol, childstr)
            else:
                self._str = '({} {})'.format(
                    self.sentence[self.leaf][1],
                    self.sentence[self.leaf][0],
                )
        return self._str

    @property
    def words(self):
        w_list = [item[0] for item in self.sentence]
        return " ".join(w_list)

    def propagate_sentence(self, sentence):
        """
        Recursively assigns sentence (list of (word, POS) pairs)
            to all nodes of a tree.
        """
        self.sentence = sentence
        for child in self.children:
            child.propagate_sentence(sentence)

    def pretty(self, level=0, marker='  '):
        pad = marker * level

        if self.leaf is not None:
            leaf_string = '({} {})'.format(
                self.symbol,
                self.sentence[self.leaf][0],
            )
            return pad + leaf_string

        else:
            result = pad + '(' + self.symbol
            for child in self.children:
                result += '\n' + child.pretty(level + 1)
            result += ')'
            return result

    @staticmethod
    def parse(line):
        """
        Loads a tree from a tree in PTB parenthetical format.
        """
        line += " "
        sentence = []
        _, t = PhraseTree._parse(line, 0, sentence)

        if t.symbol == 'TOP' and len(t.children) == 1:
            t = t.children[0]

        return t

    @staticmethod
    def _parse(line, index, sentence):
        "((...) (...) w/t (...)). returns pos and tree, and carries sent out."

        if not line[index] == '(':
            print("Invalid start tree string {} at {}".format(line, index))
            return None, PhraseTree()
        index += 1
        symbol = None
        children = []
        leaf = None
        while line[index] != ')':
            if line[index] == '(':
                try:
                    index, t = PhraseTree._parse(line, index, sentence)
                    children.append(t)
                except:
                    print(line)


            else:
                if symbol is None:
                    # symbol is here!
                    rpos = min(line.find(' ', index), line.find(')', index))
                    # see above N.B. (find could return -1)

                    symbol = line[index:rpos]  # (word, tag) string pair

                    index = rpos
                else:
                    rpos = line.find(')', index)
                    word = line[index:rpos]
                    sentence.append((word, symbol))
                    leaf = len(sentence) - 1
                    index = rpos

            if line[index] == " ":
                index += 1

        if not line[index] == ')':
            print("Invalid end tree string %s at %d" % (line, index))
            return None, PhraseTree()

        t = PhraseTree(
            symbol=symbol,
            children=children,
            sentence=sentence,
            leaf=leaf,
        )

        return (index + 1), t

    def left_span(self):
        try:
            return self._left_span
        except AttributeError:
            if self.leaf is not None:
                self._left_span = self.leaf
            else:
                self._left_span = self.children[0].left_span()
            return self._left_span

    def right_span(self):
        try:
            return self._right_span
        except AttributeError:
            if self.leaf is not None:
                self._right_span = self.leaf
            else:
                self._right_span = self.children[-1].right_span()
            return self._right_span

    def brackets(self, advp_prt=True, counts=None):

        if counts is None:
            counts = defaultdict(int)

        if self.leaf is not None:
            return {}

        nonterm = self.symbol
        if advp_prt and nonterm == 'PRT':
            nonterm = 'ADVP'

        if len(self.children) <= 0:
            return counts
        left = self.left_span()
        right = self.right_span()

        # ignore punctuation
        while (
                        left < len(self.sentence) and
                        self.sentence[left][1] in PhraseTree.puncs
        ):
            left += 1
        while (
                        right > 0 and self.sentence[right][1] in PhraseTree.puncs
        ):
            right -= 1

        if left <= right and nonterm != 'TOP':
            counts[(nonterm, left, right)] += 1

        for child in self.children:
            child.brackets(advp_prt=advp_prt, counts=counts)

        return counts

    def phrase(self):
        if self.leaf is not None:
            return [(self.leaf, self.symbol)]
        else:
            result = []
            for child in self.children:
                result.extend(child.phrase())
            return result

    @staticmethod
    def load_treefile(fname):
        trees = []
        for i, line in enumerate(open(fname)):
            t = PhraseTree.parse(line)
            if len(t) == 0:
                print("wrong parse:{}\t{}".format(i, line))
            trees.append(t)
        return trees

    def compare(self, gold, advp_prt=True):
        """
        returns (Precision, Recall, F-measure)
        """
        predbracks = self.brackets(advp_prt)
        goldbracks = gold.brackets(advp_prt)

        correct = 0
        for gb in goldbracks:
            if gb in predbracks:
                correct += min(goldbracks[gb], predbracks[gb])

        pred_total = sum(predbracks.values())
        gold_total = sum(goldbracks.values())

        return FScore(correct, pred_total, gold_total)

    def enclosing(self, i, j):
        """
        Returns the left and right indices of the labeled span in the tree
            which is next-larger than (i, j)
            (whether or not (i, j) is itself a labeled span)
        """
        for child in self.children:
            left = child.left_span()
            right = child.right_span()
            if (left <= i) and (right >= j):
                if (left == i) and (right == j):
                    break
                return child.enclosing(i, j)

        return (self.left_span(), self.right_span())

    def span_labels(self, i, j):
        """
        Returns a list of span labels (if any) for (i, j)
        """
        if self.leaf is not None:
            return []

        if (self.left_span() == i) and (self.right_span() == j):
            result = [self.symbol]
        else:
            result = []

        for child in self.children:
            left = child.left_span()
            right = child.right_span()
            if (left <= i) and (right >= j):
                result.extend(child.span_labels(i, j))
                break

        return result

    def grammar_seq(self, g_seq=[]):
        if self.leaf is None:
            grammar = [self.symbol] + [child.symbol for child in self.children]
            g_str = " ".join(grammar)
            g_seq.append(g_str)
            for child in self.children:
                child.grammar_seq(g_seq)
        return g_seq

    def grammar(self, grammar_dict=None):
        if grammar_dict is None:
            grammar_dict = defaultdict(int)
        if self.leaf is not None:
            return grammar_dict
        else:
            grammar = [self.symbol] + [child.symbol for child in self.children]
            g_str = " ".join(grammar)
            grammar_dict[g_str] += 1
            for child in self.children:
                child.grammar(grammar_dict=grammar_dict)
        return grammar_dict

    def binarize(self):
        if self.leaf is not None:
            return
        elif len(self.children) > 2:
            sub_child = self.children[1:]
            if self.symbol.endswith("#"):
                sub_symbol = self.symbol
            else:
                sub_symbol = self.symbol + "#"
            # sub_symbol=self.symbol
            new_right = PhraseTree(symbol=sub_symbol, children=sub_child)
            new_child = [self.children[0], new_right]
            self.children = new_child

        for child in self.children:
            child.binarize()


class FScore(object):
    def __init__(self, correct=0, predcount=0, goldcount=0):
        self.correct = correct  # correct brackets
        self.predcount = predcount  # total predicted brackets
        self.goldcount = goldcount  # total gold brackets

    def precision(self):
        if self.predcount > 0:
            return (100.0 * self.correct) / self.predcount
        else:
            return 0.0

    def recall(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
        else:
            return 0.0

    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0

    def __str__(self):
        precision = self.precision()
        recall = self.recall()
        fscore = self.fscore()
        return '(P= {:0.2f}, R= {:0.2f}, F= {:0.2f})'.format(
            precision,
            recall,
            fscore,
        )

    def __iadd__(self, other):
        self.correct += other.correct
        self.predcount += other.predcount
        self.goldcount += other.goldcount
        return self

    def __add__(self, other):
        return FScore(self.correct + other.correct,
                      self.predcount + other.predcount,
                      self.goldcount + other.goldcount)

    def __cmp__(self, other):
        if self.fscore() < other.fscore():
            return -1
        elif self.fscore() == other.fscore():
            return 0
        else:
            return 1

    def __gt__(self, other):
        if self.fscore() > other.fscore():
            return True
        return False

    def __lt__(self, other):
        if self.fscore() < other.fscore():
            return True
        return False

    @staticmethod
    def eval_tree_file(gfile, tfile):
        gold_trees = PhraseTree.load_treefile(gfile)
        test_trees = PhraseTree.load_treefile(tfile)

        return FScore.eval_tree_list(gold_trees, test_trees)

    @staticmethod
    def eval_tree_list(gold_trees, test_trees):
        cumulative = FScore()

        for gold, test in zip(gold_trees, test_trees):
            if len(test.sentence) == 0:
                acc = FScore()
            else:
                acc = test.compare(gold, advp_prt=True)
            cumulative += acc

        return cumulative

    @staticmethod
    def eval_seq_list(gold_seqs, test_seqs):
        cumulative = FScore()
        gold_trees = [PhraseTree.parse(line) for line in gold_seqs]
        test_trees = [PhraseTree.parse(line) for line in test_seqs]
        for gold, test in zip(gold_trees, test_trees):
            if len(test.sentence) == 0:
                acc = FScore()
            else:
                acc = test.compare(gold, advp_prt=True)
            cumulative += acc

        return cumulative


if __name__ == "__main__":
    gold_file = "../parser_data/dev.tree"
    test_file = "../parser_data/dev.pred.11800.convert"
    print("Fscore:{}".format(FScore.eval_tree_file(gold_file, test_file)))
