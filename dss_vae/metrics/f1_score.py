from .base_metric import BaseMetric


class F1Metric(BaseMetric):
    def __init__(self, ):
        super().__init__(name='F1')

    def get_evaluate(self, **kwargs):
        pass

    def _check_format(self, **kwargs):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        super().evaluate(**kwargs)


class F1(object):
    def __init__(self, correct=0, pred=0, ref=0):
        self.correct = correct
        self.predict = pred
        self.reference = ref

    @property
    def precision(self):
        if self.predict > 0:
            return (100.0 * self.correct) / self.predict
        else:
            return 0.0

    @property
    def recall(self):
        if self.reference > 0:
            return (100.0 * self.correct) / self.reference
        else:
            return 0.0

    @property
    def fscore(self):
        precision = self.precision
        recall = self.recall
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0

    def __str__(self):
        precision = self.precision
        recall = self.recall
        fscore = self.fscore
        return '(P= {:0.2f}, R= {:0.2f}, F= {:0.2f})'.format(
            precision,
            recall,
            fscore,
        )

    def __iadd__(self, other):
        self.correct += other.correct
        self.predict += other.predict
        self.reference += other.reference
        return self

    def __add__(self, other):
        return F1(
            self.correct + other.correct,
            self.predict + other.predict,
            self.reference + other.reference
        )

    def __cmp__(self, other):
        if self.fscore < other.fscore:
            return -1
        elif self.fscore == other.fscore:
            return 0
        else:
            return 1

    def __gt__(self, other):
        if self.fscore > other.fscore():
            return True
        return False

    def __lt__(self, other):
        if self.fscore < other.fscore():
            return True
        return False
