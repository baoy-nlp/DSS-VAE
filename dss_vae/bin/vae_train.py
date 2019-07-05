from __future__ import absolute_import

import sys
import time

import torch
from tensorboardX import SummaryWriter

from dss_vae.metrics.evaluation import evaluate
from dss_vae.metrics.vae_metrics import SyntaxVaeEvaluator
from dss_vae.utils.model_utils import build_model
from dss_vae.utils.model_utils import get_lr_schedule
from dss_vae.utils.model_utils import get_model_info
from dss_vae.utils.model_utils import load_data
from dss_vae.utils.model_utils import update_tracker


def train_nae(main_args, model_args, model=None):
    if "task_type" in model_args and model_args.task_type is not None:
        main_args.task_type = model_args.task_type
    dir_ret = get_model_info(main_args, model_args)
    model, optimizer, vocab = build_model(main_args, model_args, model)
    model_file = dir_ret['model_file']
    log_dir = dir_ret['log_dir']
    out_dir = dir_ret['out_dir']
    train_set, dev_set = load_data(main_args)

    writer = SummaryWriter(log_dir)
    train_iter = main_args.start_iter
    report_loss = report_examples = 0.
    history_dev_scores = []

    epoch = num_trial = patience = 0

    print("Dev ITEM: ", model_args.dev_item.lower())

    while True:
        epoch += 1
        # train_track = {}
        epoch_begin = time.time()
        for batch_examples in train_set.batch_iter(batch_size=main_args.batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()
            loss = -model.score(batch_examples)
            loss_val = torch.sum(loss).item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)
            loss.backward()

            if main_args.clip_grad > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)

            optimizer.step()

            if train_iter % main_args.log_every == 0:
                print('\r[Iter %d] Train loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr, end=" ")

                writer.add_scalar(
                    tag='Train-Iter/AutoEncoder/Loss',
                    scalar_value=report_loss / report_examples,
                    global_step=train_iter
                )
                writer.add_scalar(
                    tag='Optimize/lr',
                    scalar_value=optimizer.param_groups[0]['lr'],
                    global_step=train_iter,
                )

                report_loss = report_examples = 0.
            if train_iter % main_args.dev_every == 0:
                print()
                print('\r[Iter %d] begin validation' % train_iter, file=sys.stderr)
                eval_results = evaluate(examples=dev_set.examples, model=model, eval_src='src', eval_tgt='src',
                                        out_dir=out_dir)
                dev_acc = eval_results['accuracy']
                print('\r[Iter %d] auto_encoder %s=%.5f took %ds' % (
                    train_iter, model.args.eval_mode, dev_acc, eval_results['use_time']),
                      file=sys.stderr)
                writer.add_scalar(
                    tag='Valid-Iter/AutoEncoder/%s' % model.args.eval_mode,
                    scalar_value=dev_acc,
                    global_step=train_iter
                )

                is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
                history_dev_scores.append(dev_acc)

                writer.add_scalar(
                    tag='Valid-Iter/AutoEncoder/Best %s' % model.args.eval_mode.upper(),
                    scalar_value=max(history_dev_scores),
                    global_step=train_iter
                )

                model, optimizer, num_trial, patience = get_lr_schedule(
                    is_better=is_better,
                    model_file=model_file,
                    main_args=main_args,
                    patience=patience,
                    num_trial=num_trial,
                    model=model,
                    optimizer=optimizer,
                    reload_model=False
                )
        epoch_time = time.time() - epoch_begin
        print('\r[Epoch %d] epoch elapsed %ds' % (epoch, epoch_time), file=sys.stderr)
        writer.add_scalar(
            tag='Train-Epoch/Epoch Time',
            scalar_value=epoch_time,
            global_step=epoch
        )


def train_ae(main_args, model_args, model=None):
    if "task_type" in model_args and model_args.task_type is not None:
        main_args.task_type = model_args.task_type
    dir_ret = get_model_info(main_args, model_args)
    model, optimizer, vocab = build_model(main_args, model_args, model)
    model_file = dir_ret['model_file']
    log_dir = dir_ret['log_dir']
    out_dir = dir_ret['out_dir']
    train_set, dev_set = load_data(main_args)

    writer = SummaryWriter(log_dir)
    train_iter = main_args.start_iter
    report_loss = report_examples = 0.
    history_dev_scores = []

    epoch = num_trial = patience = 0

    while True:
        epoch += 1
        epoch_begin = time.time()
        for batch_examples in train_set.batch_iter(batch_size=main_args.batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()
            loss = -model.score(batch_examples)
            loss_val = torch.sum(loss).item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)
            loss.backward()

            if main_args.clip_grad > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)

            optimizer.step()

            if train_iter % main_args.log_every == 0:
                print('\r[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr, end=" ")

                writer.add_scalar(
                    tag='AutoEncoder/Train/loss',
                    scalar_value=report_loss / report_examples,
                    global_step=train_iter
                )
                writer.add_scalar(
                    tag='Optimize/lr',
                    scalar_value=optimizer.param_groups[0]['lr'],
                    global_step=train_iter,
                )

                report_loss = report_examples = 0.

            if train_iter % main_args.dev_every == 0:
                print()
                eval_results = evaluate(examples=dev_set.examples, model=model, eval_src='src', eval_tgt='src',
                                        out_dir=out_dir)
                dev_acc = eval_results['accuracy']
                print('\r[Iter %d] AutoEncoder %s=%.5f took %ds' % (
                    train_iter, model.args.eval_mode, dev_acc, eval_results['use_time']),
                      file=sys.stderr)
                writer.add_scalar(
                    tag='AutoEncoder/Dev/%s' % model.args.eval_mode,
                    scalar_value=dev_acc,
                    global_step=train_iter
                )

                is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
                history_dev_scores.append(dev_acc)

                writer.add_scalar(
                    tag='AutoEncoder/Dev/Best %s' % model.args.eval_mode,
                    scalar_value=max(history_dev_scores),
                    global_step=train_iter
                )

                model, optimizer, num_trial, patience = get_lr_schedule(
                    is_better=is_better,
                    model_file=model_file,
                    main_args=main_args,
                    patience=patience,
                    num_trial=num_trial,
                    model=model,
                    optimizer=optimizer,
                    reload_model=False
                )

        epoch_time = time.time() - epoch_begin
        print('\r[Epoch %d] epoch elapsed %ds' % (epoch, epoch_time), file=sys.stderr)
        writer.add_scalar(
            tag='AutoEncoder/epoch elapsed',
            scalar_value=epoch_time,
            global_step=epoch
        )


def train_vae(main_args, model_args, model=None):
    para_eval_dir = "/home/user_data/baoy/projects/seq2seq_parser/data/quora-mh/unsupervised"
    para_eval_list = ["dev.para.txt"]
    if "task_type" in model_args and model_args.task_type is not None:
        main_args.task_type = model_args.task_type
    dir_ret = get_model_info(main_args, model_args)
    model, optimizer, vocab = build_model(main_args, model_args, model)
    model_file = dir_ret['model_file']
    log_dir = dir_ret['log_dir']
    out_dir = dir_ret['out_dir']
    train_set, dev_set = load_data(main_args)

    model, optimizer, vocab = build_model(main_args, model_args, model)

    evaluator = SyntaxVaeEvaluator(
        model=model,
        out_dir=out_dir,
        train_batch_size=main_args.batch_size,
        batch_size=model_args.eval_bs,
    )

    writer = SummaryWriter(log_dir)
    writer.add_text("model", str(model))
    writer.add_text("args", str(main_args))

    train_iter = main_args.start_iter
    epoch = num_trial = patience = 0
    history_elbo = []
    history_bleu = []
    max_kl_item = -1
    max_kl_weight = None

    continue_anneal = model_args.peak_anneal

    if model_args.peak_anneal:
        model_args.warm_up = 0

    memory_temp_count = 0

    t_type = torch.Tensor

    adv_training = model_args.dis_train and model_args.adv_train
    if adv_training:
        print("has the adv training process")
    adv_syn = model_args.adv_syn > 0. or model_args.infer_weight * model_args.inf_sem
    adv_sem = model_args.adv_sem > 0. or model_args.infer_weight * model_args.inf_syn

    print(model_args.dev_item.lower())
    while True:
        epoch += 1
        train_track = {}
        for batch_examples in train_set.batch_iter(batch_size=main_args.batch_size, shuffle=True):
            train_iter += 1
            if adv_training:
                ret_loss = model.get_loss(batch_examples, train_iter, is_dis=True)
                if adv_syn:
                    dis_syn_loss = ret_loss['dis syn']
                    optimizer.zero_grad()
                    dis_syn_loss.backward()
                    if main_args.clip_grad > 0.:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)
                        # optimizer.step()
                if adv_sem:
                    ret_loss = model.get_loss(batch_examples, train_iter, is_dis=True)
                    dis_sem_loss = ret_loss['dis sem']
                    optimizer.zero_grad()
                    dis_sem_loss.backward()
                    if main_args.clip_grad > 0.:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)
                        # optimizer.step()

            ret_loss = model.get_loss(batch_examples, train_iter)
            loss = ret_loss['Loss']
            optimizer.zero_grad()
            loss.backward()
            if main_args.clip_grad > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)

            optimizer.step()
            train_iter += 1
            # tracker = update_track(loss, train_avg_kl, train_avg_nll, tracker)
            train_track = update_tracker(ret_loss, train_track)
            if train_iter % main_args.log_every == 0:
                train_avg_nll = ret_loss['NLL Loss']
                train_avg_kl = ret_loss['KL Loss']
                _kl_weight = ret_loss['KL Weight']
                for key, val in ret_loss.items():
                    writer.add_scalar(
                        'Train-Iter/VAE/{}'.format(key),
                        val.item() if isinstance(val, t_type) else val,
                        train_iter
                    )

                print("\rTrain-Iter %04d, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, WD-Drop %6.3f"
                      % (train_iter, loss.item(), train_avg_nll, train_avg_kl, _kl_weight, model.step_unk_rate),
                      end=' ')
                writer.add_scalar(
                    tag='Optimize/lr',
                    scalar_value=optimizer.param_groups[0]['lr'],
                    global_step=train_iter,
                )

            if train_iter % main_args.dev_every == 0 and train_iter > model_args.warm_up:
                # dev_track, eval_results = _test_vae(model, dev_set, main_args, train_iter)
                dev_track, eval_results = evaluator.evaluate_reconstruction(examples=dev_set.examples,
                                                                            eval_desc="dev{}".format(train_iter),
                                                                            eval_step=train_iter, write_down=False)
                _weight = model.get_kl_weight(step=train_iter)
                _kl_item = torch.mean(dev_track['KL Item'])
                # writer.add_scalar("VAE/Valid-Iter/KL Item", _kl_item, train_iter)
                for key, val in dev_track.items():
                    writer.add_scalar(
                        'Valid-Iter/VAE/{}'.format(key),
                        torch.mean(val) if isinstance(val, t_type) else val,
                        train_iter
                    )
                if continue_anneal and model.step_kl_weight is None:
                    if _kl_item > max_kl_item:
                        max_kl_item = _kl_item
                        max_kl_weight = _weight
                    else:
                        if (max_kl_item - _kl_item) > model_args.stop_clip_kl:
                            model.step_kl_weight = max_kl_weight
                            writer.add_text(tag='peak_anneal',
                                            text_string="fixed the kl weight:{} with kl peak:{} at step:{}".format(
                                                max_kl_weight,
                                                max_kl_item,
                                                train_iter
                                            ), global_step=train_iter)
                            continue_anneal = False
                dev_elbo = torch.mean(dev_track['Model Score'])
                writer.add_scalar("Evaluation/VAE/Dev Score", dev_elbo, train_iter)

                # evaluate bleu
                dev_bleu = eval_results['accuracy']
                print()
                print("Valid-Iter %04d, NLL_Loss:%9.4f, KL_Loss: %9.4f, Sum Score:%9.4f BLEU:%9.4f" % (
                    train_iter,
                    torch.mean(dev_track['NLL Loss']),
                    torch.mean(dev_track['KL Loss']),
                    dev_elbo,
                    eval_results['accuracy']), file=sys.stderr
                      )
                writer.add_scalar(
                    tag='Evaluation/VAE/Iter %s' % model.args.eval_mode,
                    scalar_value=dev_bleu,
                    global_step=train_iter
                )
                if model_args.dev_item == "ELBO" or model_args.dev_item.lower() == "para-elbo" or model_args.dev_item.lower() == "gen-elbo":
                    is_better = history_elbo == [] or dev_elbo < min(history_elbo)
                elif model_args.dev_item == "BLEU" or model_args.dev_item.lower() == "para-bleu" or model_args.dev_item.lower() == "gen-bleu":
                    is_better = history_bleu == [] or dev_bleu > max(history_bleu)

                history_elbo.append(dev_elbo)
                writer.add_scalar("Evaluation/VAE/Best ELBO Score", min(history_elbo), train_iter)
                history_bleu.append(dev_bleu)
                writer.add_scalar("Evaluation/VAE/Best BLEU Score", max(history_bleu), train_iter)

                if is_better:
                    writer.add_scalar(
                        tag='Evaluation/VAE/Best %s' % model.args.eval_mode,
                        scalar_value=dev_bleu,
                        global_step=train_iter
                    )
                    writer.add_scalar(
                        tag='Evaluation/VAE/Best NLL-LOSS',
                        scalar_value=torch.mean(dev_track['NLL Loss']),
                        global_step=train_iter
                    )
                    writer.add_scalar(
                        tag='Evaluation/VAE/Best KL-LOSS',
                        scalar_value=torch.mean(dev_track['KL Loss']),
                        global_step=train_iter
                    )
                    if train_iter * 2 > model_args.x0:
                        memory_temp_count = 3

                if model_args.dev_item.lower().startswith("gen") and memory_temp_count > 0:
                    evaluator.evaluate_generation(
                        sample_size=len(dev_set.examples),
                        eval_desc="gen_iter{}".format(train_iter),
                    )
                    memory_temp_count -= 1

                if model_args.dev_item.lower().startswith("para") and memory_temp_count > 0:

                    para_score = evaluator.evaluate_para(
                        eval_dir=para_eval_dir,
                        eval_list=para_eval_list,
                        eval_desc="para_iter{}".format(train_iter)
                    )
                    if memory_temp_count == 3:
                        writer.add_scalar(
                            tag='Evaluation/VAE/Para Dev Ori-BLEU',
                            scalar_value=para_score[0][0],
                            global_step=train_iter
                        )
                        writer.add_scalar(
                            tag='Evaluation/VAE/Para Dev Tgt-BLEU',
                            scalar_value=para_score[0][1],
                            global_step=train_iter
                        )
                        if len(para_score) > 1:
                            writer.add_scalar(
                                tag='Evaluation/VAE/Para Test Ori-BLEU',
                                scalar_value=para_score[1][0],
                                global_step=train_iter
                            )
                            writer.add_scalar(
                                tag='Evaluation/VAE/Para Test Tgt-BLEU',
                                scalar_value=para_score[1][1],
                                global_step=train_iter
                            )
                    memory_temp_count -= 1

                model, optimizer, num_trial, patience = get_lr_schedule(
                    is_better=is_better,
                    model_file=model_file,
                    main_args=main_args,
                    patience=patience,
                    num_trial=num_trial,
                    model=model,
                    optimizer=optimizer,
                    reload_model=model_args.reload_model,
                )
                model.train()
        elbo = torch.mean(train_track['Model Score'])
        print()
        print("Train-Epoch %02d, Score %9.4f" % (epoch, elbo))
        for key, val in train_track.items():
            writer.add_scalar(
                'Train-Epoch/VAE/{}'.format(key),
                torch.mean(val) if isinstance(val, t_type) else val,
                epoch
            )
