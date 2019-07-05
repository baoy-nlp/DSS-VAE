import sys

sys.path.append(".")
from .bleu_scorer import BleuScoreMetric
from dss_vae.utils.utility import write_docs
from dss_vae.preprocess.prepare_raw_set import ptb_to_s2b


def tree_to_s2b_file(ptb_file, s2b_file):
    ptbs = ptb_to_s2b(tree_file=ptb_file, rm_same=False)
    write_docs(fname=s2b_file, docs=ptbs)


def eval_syn(ptb_file, ref_file):
    s2b_file = ptb_file + ".s2b"
    tgt_file = ref_file + ".s2b"
    tree_to_s2b_file(ptb_file, s2b_file)
    tree_to_s2b_file(ref_file, tgt_file)
    print(BleuScoreMetric.evaluate_file(pred_file=s2b_file, gold_files=tgt_file))


pred_file = sys.argv[1]
gold_file = sys.argv[2]

eval_syn(pred_file, gold_file)
