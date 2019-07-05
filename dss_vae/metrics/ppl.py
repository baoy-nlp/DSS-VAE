from .base_metric import BaseMetric


class PPLScore(BaseMetric):
    def __init__(self, ):
        super().__init__(name="Perplexity")

    def get_evaluate(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        super().evaluate(**kwargs)
