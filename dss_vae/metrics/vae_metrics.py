"""
evaluate vae performance:

evaluate on test set:
    nll loss : model output
    kl loss : model output
    reconstruction bleu
    uni-gram kl
    avg_len
    ppl:
"""
import os

from dss_vae.metrics.tools import *
from dss_vae.utils.model_utils import update_tracker
from dss_vae.structs import Corpus
from .bleu_scorer import BleuScoreMetric
from .evaluation import evaluate
from .unigram_kl import UnigramKLMetric


class VaeEvaluator(object):
    def __init__(self, model, out_dir, batch_size=10, train_batch_size=32, **kwargs):
        self.vae = model
        self.out_dir = out_dir
        self.eval_batch_size = batch_size
        self.train_batch_size = train_batch_size


class SyntaxVaeEvaluator(VaeEvaluator):
    """
    set a evaluator
    """

    def __init__(self, model, out_dir, batch_size=10, train_batch_size=32, **kwargs):
        super(SyntaxVaeEvaluator, self).__init__(model, out_dir, batch_size, train_batch_size)

    def __call__(self, *args, **kwargs):
        pass

    def evaluate_reconstruction(self, examples, eval_desc, eval_step=None, write_down=True):
        training = self.vae.training
        self.vae.eval()
        step = eval_step if eval_step is not None else 2 * self.vae.args.x0
        eval_results = evaluate(examples, self.vae, sort_key='src', eval_tgt='src', batch_size=self.eval_batch_size)
        pred = predict_to_plain(eval_results['predict'])
        gold = reference_to_plain(eval_results['reference'])

        dev_set = Dataset(examples)
        dev_track = {}
        for dev_examples in dev_set.batch_iter(batch_size=self.train_batch_size, shuffle=False):
            ret_loss = self.vae.get_loss(dev_examples, step)
            dev_track = update_tracker(ret_loss, dev_track)
        if write_down:
            write_result(pred, fname=os.path.join(self.out_dir, eval_desc + ".pred"))
            write_result(gold, fname=os.path.join(self.out_dir, eval_desc + ".gold"))
            write_result(dev_track, fname=os.path.join(self.out_dir, eval_desc + ".score"))
            with open(os.path.join(self.out_dir, eval_desc + ".score", ), "a") as f:
                f.write("{}:{}".format(self.vae.args.eval_mode, eval_results['accuracy']))
        self.vae.training = training
        return dev_track, eval_results

    def evaluate_generation(self, corpus_examples=None, sample_size=50, eval_desc="gen", write_down=True):
        training = self.vae.training
        self.vae.eval()
        train = [e.src for e in corpus_examples] if corpus_examples is not None else None
        sample_data = self.vae.genewwrating(sample_size, batch_size=self.eval_batch_size)
        sample_corpus = to_plain(sample_data)
        write_result(docs=sample_corpus, fname=os.path.join(self.out_dir, eval_desc + ".text"))

        if write_down:
            t = UnigramKLMetric()
            p = Corpus(sample_corpus)
            g = Corpus(train) if train is not None else None
            eval_results = {
                'UnigramKL': t.get_evaluate(train, pred_source=sample_corpus,
                                            vocab=self.vae.vocab.src) if train is not None else None,
                "Gold AvgLen": g.avg_length if train is not None else None,
                "Gen AvgLen": p.avg_length,
            }
            write_result(docs=eval_results, fname=os.path.join(self.out_dir, eval_desc + ".score"))
        print("write result to {}".format(os.path.join(self.out_dir, eval_desc + ".text")))

        self.vae.training = training

    def evaluate_para(self, eval_dir: str, eval_list: list, eval_desc='para'):
        training = self.vae.training
        data_set_list = [Dataset.from_raw_file(os.path.join(eval_dir, file), e_type='plain') for file in eval_list]

        def eval_para(dev_set, data_name):
            ori_examples = []
            tgt_examples = []
            pred = []
            for dev_examples in dev_set.batch_iter(batch_size=self.eval_batch_size, shuffle=False):
                ori, tgt = split_examples(dev_examples)
                ori_examples.extend(ori)
                tgt_examples.extend(tgt)
                ret = self.vae.conditional_generating(condition='sem', examples=ori)
                pred.extend(ret['res'])

            ori_reference = recovery_ref(ori_examples, self.vae.vocab.src, keep_origin=True)
            tgt_reference = recovery_ref(tgt_examples, self.vae.vocab.src, keep_origin=True)
            ori_bleu = get_bleu_score(references=ori_reference, hypothesis=pred)
            tgt_bleu = get_bleu_score(references=tgt_reference, hypothesis=pred)
            pred_list = predict_to_plain(pred)
            ori_list = con_to_plain(ori_reference)
            tgt_list = con_to_plain(tgt_reference)
            write_result(pred_list, fname=os.path.join(self.out_dir, "{}.{}.pred".format(data_name, eval_desc)))
            write_result(tgt_list, fname=os.path.join(self.out_dir, "{}.{}.tgt".format(data_name, eval_desc)))
            write_result(ori_list, fname=os.path.join(self.out_dir, "{}.{}.ori".format(data_name, eval_desc)))
            bleu_score = "ori bleu:{}, tgt bleu:{} !".format(ori_bleu, tgt_bleu)
            print("finish {} with {}".format(data_name, bleu_score))
            return list([ori_bleu, tgt_bleu])

        s_format = "ori bleu:{}, tgt bleu:{} !"

        scores_history = [eval_para(data_set, fname) for data_set, fname in zip(data_set_list, eval_list)]
        scores_history_str = [s_format.format(*item) for item in scores_history]
        write_append_result(docs=scores_history_str, fname=os.path.join(self.out_dir, eval_desc + ".score"))
        self.vae.training = training
        return scores_history

    def evaluate_pure_para(self, eval_dir: str, eval_list: list, eval_desc='pure-para'):
        training = self.vae.training
        data_set_list = [Dataset.from_raw_file(os.path.join(eval_dir, file), e_type='plain') for file in eval_list]

        def eval_para(dev_set, data_name):
            ori_examples = []
            tgt_examples = []
            pred = []
            for dev_examples in dev_set.batch_iter(batch_size=self.eval_batch_size, shuffle=False):
                ori, tgt = split_examples(dev_examples)
                ori_examples.extend(ori)
                tgt_examples.extend(tgt)
                ret = self.vae.conditional_generating(condition='sem-only', examples=ori)
                pred.extend(ret['res'])

            ori_reference = recovery_ref(ori_examples, self.vae.vocab.src, keep_origin=True)
            tgt_reference = recovery_ref(tgt_examples, self.vae.vocab.src, keep_origin=True)
            ori_bleu = get_bleu_score(references=ori_reference, hypothesis=pred)
            tgt_bleu = get_bleu_score(references=tgt_reference, hypothesis=pred)
            pred_list = predict_to_plain(pred)
            ori_list = con_to_plain(ori_reference)
            tgt_list = con_to_plain(tgt_reference)
            write_result(pred_list, fname=os.path.join(self.out_dir, "{}.{}.pred".format(data_name, eval_desc)))
            write_result(tgt_list, fname=os.path.join(self.out_dir, "{}.{}.tgt".format(data_name, eval_desc)))
            write_result(ori_list, fname=os.path.join(self.out_dir, "{}.{}.ori".format(data_name, eval_desc)))
            bleu_score = "ori bleu:{}, tgt bleu:{} !".format(ori_bleu, tgt_bleu)
            print("finish {} with {}".format(data_name, bleu_score))
            return list([ori_bleu, tgt_bleu])

        s_format = "ori bleu:{}, tgt bleu:{} !"

        scores_history = [eval_para(data_set, fname) for data_set, fname in zip(data_set_list, eval_list)]
        scores_history_str = [s_format.format(*item) for item in scores_history]
        write_append_result(docs=scores_history_str, fname=os.path.join(self.out_dir, eval_desc + ".score"))
        self.vae.training = training
        return scores_history

    def evaluate_control(self, eval_dir: str, eval_list: list, eval_desc='con'):
        training = self.vae.training
        self.vae.eval()
        data_set_list = [Dataset.from_raw_file(os.path.join(eval_dir, file), e_type='plain') for file in eval_list]

        def eval_control(dev_set, data_name):
            sem_examples = []
            syn_examples = []
            pred = []
            for dev_examples in dev_set.batch_iter(batch_size=self.eval_batch_size, shuffle=False):
                sem, syn = split_examples(dev_examples)
                sem_examples.extend(sem)
                syn_examples.extend(syn)
                ret = self.vae.eval_adv(sem, syn)
                pred.extend(ret['res'])
            sem_e = recovery_ref(sem_examples, self.vae.vocab.src, keep_origin=True)
            syn_e = recovery_ref(syn_examples, self.vae.vocab.src, keep_origin=True)
            tgt_bleu = get_bleu_score(references=syn_e, hypothesis=pred)
            pred_list = predict_to_plain(pred)
            sem_list = con_to_plain(sem_e)
            syn_list = con_to_plain(syn_e)
            write_result(pred_list, fname=os.path.join(self.out_dir, "{}.{}.pred".format(data_name, eval_desc)))
            write_result(syn_list, fname=os.path.join(self.out_dir, "{}.{}.gold".format(data_name, eval_desc)))
            write_result(sem_list, fname=os.path.join(self.out_dir, "{}.{}.input".format(data_name, eval_desc)))
            print("finish {} with bleu:{} !".format(data_name, tgt_bleu))
            return str(tgt_bleu)

        scores = [eval_control(data_set, fname) for data_set, fname in zip(data_set_list, eval_list)]
        write_result(docs=scores, fname=os.path.join(self.out_dir, eval_desc + ".score"))
        self.vae.training = training

    def evaluate_style_transfer(self, eval_dir: str, eval_list: list, eval_desc='transfer'):
        """
        read two sentence: sem,syn
        get output: sem->parse,syn->parse,pred->parse
        """
        training = self.vae.training
        self.vae.eval()
        data_set_list = [Dataset.from_raw_file(os.path.join(eval_dir, file), e_type='plain') for file in eval_list]

        def evaluate_data_set(dev_set, data_name):
            pred_output = os.path.join(self.out_dir, "{}.{}.pred".format(data_name, eval_desc))
            tgt_output = os.path.join(self.out_dir, "{}.{}.tgt".format(data_name, eval_desc))
            ori_output = os.path.join(self.out_dir, "{}.{}.ori".format(data_name, eval_desc))
            ss_output = os.path.join(self.out_dir, "{}.{}.ss".format(data_name, eval_desc))
            st_output = os.path.join(self.out_dir, "{}.{}.st".format(data_name, eval_desc))
            sp_output = os.path.join(self.out_dir, "{}.{}.sp".format(data_name, eval_desc))

            sem_examples = []
            syn_examples = []
            pred = []
            syn_tgt = []
            syn_src = []
            syn_pred = []
            for dev_examples in dev_set.batch_iter(batch_size=self.eval_batch_size, shuffle=False):
                sem, syn = split_examples(dev_examples)
                sem_examples.extend(sem)
                syn_examples.extend(syn)
                ret = self.vae.eval_adv(sem, syn)
                pred.extend(ret['res'])
                if not self.vae.args.model_select == "OriginVAE":
                    syn_src.extend(ret['ori syn'])
                    syn_tgt.extend(ret['ref syn'])

            ori_reference = recovery_ref(sem_examples, self.vae.vocab.src)
            tgt_reference = recovery_ref(syn_examples, self.vae.vocab.src)
            ori_list = con_to_plain(ori_reference)
            tgt_list = con_to_plain(tgt_reference)
            write_result(tgt_list, fname=tgt_output)
            write_result(ori_list, fname=ori_output)
            pred_list = predict_to_plain(pred)
            write_result(pred_list, fname=pred_output)

            # for word
            word_ori_bleu = BleuScoreMetric.evaluate_file(pred_file=pred_output, gold_files=ori_output)  # src-pred
            word_tgt_bleu = BleuScoreMetric.evaluate_file(pred_file=pred_output, gold_files=tgt_output)  # tgt-pred
            word_bleu = "{} word : ori bleu:{}, tgt bleu:{} !".format(data_name, word_ori_bleu, word_tgt_bleu)
            print("finish {} with {}".format(data_name, word_bleu))

            # for syn
            if not self.vae.args.model_select == "OriginVAE":
                syn_data_set = Dataset.from_raw_file(pred_output, e_type='plain')

                for dev_examples in syn_data_set.batch_iter(batch_size=self.eval_batch_size, shuffle=False):
                    syn_output = self.vae.eval_syntax(dev_examples)
                    syn_pred.extend(syn_output)
                syn_src_list = predict_to_plain(syn_src)
                write_result(syn_src_list, fname=ss_output)
                syn_tgt_list = predict_to_plain(syn_tgt)
                write_result(syn_tgt_list, fname=st_output)
                syn_pred_list = predict_to_plain(syn_pred)
                write_result(syn_pred_list, fname=sp_output)
                syn_ori_bleu = BleuScoreMetric.evaluate_file(pred_file=sp_output, gold_files=ss_output)  # src-pred
                syn_tgt_bleu = BleuScoreMetric.evaluate_file(pred_file=sp_output, gold_files=st_output)  # tgt-pred
                syn_bleu = "{} syn : ori bleu:{}, tgt bleu:{} !".format(data_name, syn_ori_bleu, syn_tgt_bleu)
                print("finish {} with {}".format(data_name, syn_bleu))

                # oracle
                bound_word_bleu = BleuScoreMetric.evaluate_file(pred_file=ori_output, gold_files=tgt_output)
                bound_syn_bleu = BleuScoreMetric.evaluate_file(pred_file=ss_output, gold_files=st_output)
                bound_bleu = "{} word bleu:{}, syn bleu:{} !".format(data_name, bound_word_bleu, bound_syn_bleu)
                print("finish {} with {}".format(data_name, bound_bleu))
                return "\n".join([word_bleu, syn_bleu, bound_bleu])
            else:
                return word_bleu

        scores = [evaluate_data_set(data_set, fname) for data_set, fname in zip(data_set_list, eval_list)]
        write_result(docs=scores, fname=os.path.join(self.out_dir, eval_desc + ".score"))
        self.vae.training = training
