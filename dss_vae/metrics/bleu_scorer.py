from nltk.translate import bleu_score

from .base_metric import BaseMetric


class BleuScoreMetric(BaseMetric):
    def __init__(self):
        super(BleuScoreMetric, self).__init__(name="BLEU")

    def get_evaluate(self, reference, hypothesis, etype="corpus"):
        """
        :param reference: list[list[list[str]]]
        :param hypothesis: list[list[str]]
        :param etype: "corpus" or "sents"
        :return: BLEU SCORE
        """
        self._check_format(reference, hypothesis)
        eval_select = {
            'corpus': bleu_score.corpus_bleu,
            'sents': bleu_score.sentence_bleu
        }[etype]
        return 100.0 * eval_select(list_of_references=reference, hypotheses=hypothesis)

    def _check_format(self, reference, hypothesis):
        if not isinstance(reference, list):
            raise RuntimeError("reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))
        elif not isinstance(reference[0], list):
            raise RuntimeError("reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))
        elif not isinstance(reference[0][0], list):
            raise RuntimeError("reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))
        elif not isinstance(reference[0][0][0], str):
            raise RuntimeError("reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))

        if not isinstance(hypothesis, list):
            raise RuntimeError("hypothesis type is not true, expect to be list(list(str)) but is {}".format(type(hypothesis)))
        elif not isinstance(hypothesis[0], list):
            raise RuntimeError("hypothesis type is not true, expect to be list(list(str)) but is {}".format(type(hypothesis)))
        elif not isinstance(hypothesis[0][0], str):
            raise RuntimeError("hypothesis type is not true, expect to be list(list(str)) but is {}".format(type(hypothesis)))

    @staticmethod
    def evaluate_file(pred_file, gold_files, etype="corpus"):

        with open(pred_file, "r") as f:
            hypothesis = [line.strip().split(" ") for line in f.readlines()]

        references = [[] for _ in range(len(hypothesis))]
        if not isinstance(gold_files, list):
            gold_files = [gold_files, ]

        for file in gold_files:
            with open(file, "r") as f:
                ref_i = [line.strip().split(" ") for line in f.readlines()]
                for idx, ref in enumerate(ref_i):
                    references[idx].append(ref)

        evaluator = BleuScoreMetric()
        bleu = evaluator.get_evaluate(reference=references, hypothesis=hypothesis, etype=etype)
        return bleu
