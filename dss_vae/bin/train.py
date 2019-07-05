from __future__ import absolute_import

import sys
import time

import torch
from tensorboardX import SummaryWriter

from dss_vae.metrics import get_evaluator
from dss_vae.utils.global_ops import GlobalOps
from dss_vae.utils.model_utils import build_model
from dss_vae.utils.model_utils import get_lr_schedule
from dss_vae.utils.model_utils import get_model_info
from dss_vae.utils.model_utils import load_data
from dss_vae.utils.model_utils import update_tracker

sort_key_dict = {
    'rnn': "src",
    "att": "src",
    'src': "src",
    'tgt': "tgt",
    "syn": "syn",
}
eval_key_dict = {
    'autoencoder': "bleu",
    'paraphrase': "para",
    "vae": "vae",
    "syntax-para": "svae",
    'syntaxvae': "svae",
    "syntaxvae2": "svae2",
    "translation": "mt",
    "reorder": 'reorder'
}


def training(main_args, model_args, model=None):
    if "task_type" in model_args and model_args.task_type is not None:
        main_args.task_type = model_args.task_type
    dir_ret = get_model_info(main_args=main_args, model_args=model_args)
    model, optimizer, vocab = build_model(main_args=main_args, model_args=model_args, model=model)
    train_set, dev_set = load_data(main_args)
    model_file = dir_ret['model_file']
    log_dir = dir_ret['log_dir']
    out_dir = dir_ret['out_dir']

    writer = SummaryWriter(log_dir)
    GlobalOps.writer_ops = writer
    writer.add_text("main_args", str(main_args))
    writer.add_text("model_args", str(model_args))

    print("...... Start Training ......")

    train_iter = main_args.start_iter
    train_nums, train_loss = 0., 0.
    epoch, num_trial, patience, = 0, 0, 0

    history_scores = []
    task_type = main_args.task_type
    eval_select = eval_key_dict[task_type.lower()]
    sort_key = sort_key_dict[model_args.sort_key] if "sort_key" in model_args else sort_key_dict[model_args.enc_type]
    evaluator = get_evaluator(
        eval_choice=eval_select,
        model=model,
        eval_set=dev_set.examples,
        eval_lists=main_args.eval_lists,
        sort_key=sort_key,
        eval_tgt="tgt",
        batch_size=model_args.eval_bs,
        out_dir=out_dir,
        write_down=True
    )
    print("Dev ITEM: ", evaluator.score_item)
    adv_training = "adv_train" in model_args and model_args.adv_train
    adv_syn, adv_sem = False, False

    def hyper_init():
        adv_syn_ = (model_args.adv_syn + model_args.infer_weight * model_args.inf_sem) > 0.
        adv_sem_ = (model_args.adv_sem + model_args.infer_weight * model_args.inf_syn) > 0.
        return adv_syn_, adv_sem_

    if adv_training:
        adv_syn, adv_sem = hyper_init()

    def normal_training():
        optimizer.zero_grad()
        batch_ret_ = model.get_loss(examples=batch_examples, return_enc_state=False, train_iter=train_iter)
        batch_loss_ = batch_ret_['Loss']
        return batch_ret_

    def universal_training():
        if adv_training:
            ret_loss = model.get_loss(examples=batch_examples, train_iter=train_iter, is_dis=True)
            if adv_syn:
                dis_syn_loss = ret_loss['dis syn']
                optimizer.zero_grad()
                dis_syn_loss.backward()
                if main_args.clip_grad > 0.:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)
                optimizer.step()
            if adv_sem:
                ret_loss = model.get_loss(examples=batch_examples, train_iter=train_iter, is_dis=True)
                dis_sem_loss = ret_loss['dis sem']
                optimizer.zero_grad()
                dis_sem_loss.backward()
                if main_args.clip_grad > 0.:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)
                optimizer.step()
        return normal_training()

    while True:
        epoch += 1
        epoch_begin = time.time()
        train_log_dict = {}

        for batch_examples in train_set.batch_iter(batch_size=main_args.batch_size, shuffle=True):
            train_iter += 1
            train_nums += len(batch_examples)
            # batch_ret = model.get_loss(examples=batch_examples, return_enc_state=False, train_iter=train_iter)
            batch_ret = universal_training()
            batch_loss = batch_ret['Loss']
            train_loss += batch_loss.sum().item()
            torch.mean(batch_loss).backward()

            if main_args.clip_grad > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)
            optimizer.step()

            train_log_dict = update_tracker(batch_ret, train_log_dict)

            if train_iter % main_args.log_every == 0:
                print('\r[Iter %d] Train loss=%.5f' % (train_iter, train_loss / train_nums), file=sys.stdout, end=" ")
                for key, val in train_log_dict.items():
                    if isinstance(val, torch.Tensor):
                        writer.add_scalar(
                            tag="{}/Train/{}".format(task_type, key),
                            scalar_value=torch.mean(val).item(),
                            global_step=train_iter
                        )
                writer.add_scalar(
                    tag="Optimize/lr",
                    scalar_value=optimizer.param_groups[0]['lr'],
                    global_step=train_iter
                )
                writer.add_scalar(
                    tag='Optimize/trial',
                    scalar_value=num_trial,
                    global_step=train_iter,
                )
                writer.add_scalar(
                    tag='Optimize/patience',
                    scalar_value=patience,
                    global_step=train_iter,
                )

            if train_iter % main_args.dev_every == 0 and train_iter > model_args.warm_up:
                eval_result_dict = evaluator()
                dev_acc = eval_result_dict[evaluator.score_item]
                if isinstance(dev_acc, torch.Tensor):
                    dev_acc = dev_acc.sum().item()
                print('\r[Iter %d] %s %s=%.3f took %d s' %
                      (
                          train_iter,
                          task_type,
                          evaluator.score_item,
                          dev_acc,
                          eval_result_dict['EVAL TIME']), file=sys.stdout
                      )
                is_better = (history_scores == []) or dev_acc > max(history_scores)
                history_scores.append(dev_acc)

                writer.add_scalar(
                    tag='%s/Valid/Best %s' % (task_type, evaluator.score_item),
                    scalar_value=max(history_scores),
                    global_step=train_iter
                )
                for key, val in eval_result_dict.items():
                    writer.add_scalar(
                        tag="{}/Valid/{}".format(task_type, key),
                        scalar_value=val.sum().item() if isinstance(val, torch.Tensor) else val,
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
                model.train()

        epoch_time = time.time() - epoch_begin
        print('\r[Epoch %d] epoch elapsed %ds' % (epoch, epoch_time), file=sys.stdout)
        # writer.add_scalar(
        #     tag='{}/epoch elapsed'.format(task_type),
        #     scalar_value=epoch_time,
        #     global_step=epoch
        # )
