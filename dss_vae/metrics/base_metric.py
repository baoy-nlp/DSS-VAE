class BaseMetric(object):
    def __init__(self, name='Base Metric'):
        self.name = name

    def get_evaluate(self, **kwargs):
        raise NotImplementedError

    def _check_format(self, **kwargs):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        score = self.get_evaluate(**kwargs)
        print("{} Score : ".format(self.name), score)
