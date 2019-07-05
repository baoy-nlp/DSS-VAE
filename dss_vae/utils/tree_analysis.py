from collections import defaultdict

from dss_vae.structs import GlobalNames
from dss_vae.structs import FScore
from dss_vae.structs import PhraseTree
from dss_vae.utils.utility import write_docs
from dss_vae.preprocess import s2b_to_s2t
from dss_vae.preprocess import s2t_check
from dss_vae.preprocess import s2t_fix
from dss_vae.preprocess import s2t_to_tree


def eval_s2t(preds, golds):
    error_count = 0
    eval_gold = []
    eval_pred = []
    for pred, gold in zip(preds, golds):
        if s2t_check(pred):
            eval_gold.append(s2t_to_tree(s2t_str=gold))
            eval_pred.append(s2t_to_tree(s2t_str=pred))
        else:
            eval_gold.append(s2t_to_tree(s2t_str=gold))
            eval_pred.append("(TOP XX)")
            error_count += 1

    return FScore.eval_seq_list(gold_seqs=eval_gold, test_seqs=eval_pred), error_count


def eval_s2t_robust(preds, golds):
    error_count = 0
    error_sum_fix = 0.0
    error_fix_sents = 0.0
    eval_gold = []
    eval_pred = []
    for pred, gold in zip(preds, golds):
        pred, error_fix = s2t_fix(pred, fm=GlobalNames.get_fm())
        error_sum_fix += error_fix
        if error_fix > 0:
            error_fix_sents += 1
        gold, _ = s2t_fix(gold, fm=GlobalNames.get_fm())
        if s2t_check(pred):
            eval_gold.append(s2t_to_tree(s2t_str=gold))
            eval_pred.append(s2t_to_tree(s2t_str=pred))
        else:
            eval_gold.append(s2t_to_tree(s2t_str=gold))
            eval_pred.append("(TOP XX)")
            error_count += 1

    avg_error = error_sum_fix / error_fix_sents if error_fix_sents > 0 else 0.0

    return FScore.eval_seq_list(gold_seqs=eval_gold, test_seqs=eval_pred), "{},avg_fix:{}".format(error_count, avg_error)


def eval_s2b(preds, golds):
    error_count = 0
    error_sum_fix = 0.0
    error_fix_sents = 0.0
    eval_gold = []
    eval_pred = []
    for pred, gold in zip(preds, golds):
        pred, error_fix = s2b_to_s2t(pred, fm=GlobalNames.get_fm())
        error_sum_fix += error_fix
        if error_fix > 0:
            error_fix_sents += 1
        gold, _ = s2b_to_s2t(gold, fm=GlobalNames.get_fm())
        if s2t_check(pred):
            eval_gold.append(s2t_to_tree(s2t_str=gold))
            eval_pred.append(s2t_to_tree(s2t_str=pred))
        else:
            eval_gold.append(s2t_to_tree(s2t_str=gold))
            eval_pred.append("(TOP XX)")
            error_count += 1

    avg_error = error_sum_fix / error_fix_sents if error_fix_sents > 0 else 0.0

    return FScore.eval_seq_list(gold_seqs=eval_gold, test_seqs=eval_pred), "{},avg_fix:{}".format(error_count, avg_error)


def eval_file(pred_file, gold_file):
    pred = []
    gold = []
    with open(pred_file, 'r') as f:
        for line in f:
            pred.append(line)
    with open(gold_file, 'r') as f:
        for line in f:
            gold.append(line)
    return eval_s2t(pred, gold)


def extract_origin_grammar(tree_file, out_file="grammar.out"):
    grammar_dict = defaultdict(int)
    trees = PhraseTree.load_treefile(tree_file)
    for tree in trees:
        tree.grammar(grammar_dict)
    grammar_list = [grammar for grammar, val in grammar_dict.items()]
    write_docs(fname=out_file, docs=grammar_list)
    return grammar_dict


def extract_binary_grammar(tree_file, out_file="grammar.out"):
    grammar_dict = defaultdict(int)
    trees = PhraseTree.load_treefile(tree_file)
    for tree in trees:
        tree.binarize()
        tree.grammar(grammar_dict)
    grammar_list = [grammar for grammar, val in grammar_dict.items()]
    write_docs(fname=out_file, docs=grammar_list)
    return grammar_dict


def evaluate_coverage(dict_a, dict_b):
    sum_val = 0.0
    count = 0.0

    for item, _ in dict_b.items():
        sum_val += 1.0
        if item in dict_a:
            count += 1.0
    return count * 100.0 / sum_val


def evaluate_using_ratio(dict_a, dict_b):
    sum_val = 0.0
    count = 0.0

    for item, val in dict_b.items():
        sum_val += val
        if item in dict_a:
            count += val
    return count * 100.0 / sum_val


def evaluate_grammar_coverage(train_file, dev_file, test_file, grammar_type='.binary'):
    if grammar_type == '.binary':
        extract_grammar = extract_binary_grammar
    else:
        extract_grammar = extract_origin_grammar
    train_dict = extract_grammar(train_file, train_file + grammar_type)
    dev_dict = extract_grammar(dev_file, dev_file + grammar_type)
    test_dict = extract_grammar(test_file, test_file + grammar_type)

    print("cover dev:{}".format(evaluate_coverage(train_dict, dev_dict)))
    print("cover test:{}".format(evaluate_coverage(train_dict, test_dict)))

    print("ratio dev:{}".format(evaluate_using_ratio(train_dict, dev_dict)))
    print("ratio test:{}".format(evaluate_using_ratio(train_dict, test_dict)))
