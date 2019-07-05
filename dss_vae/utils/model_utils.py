import math
import os
import shutil
import sys

import torch

from dss_vae.models import init_create_model
from dss_vae.models import load_static_model
from dss_vae.structs import Dataset
from dss_vae.structs import Vocab
from .config_funcs import dict_to_yaml, args_to_dict
from .config_funcs import yaml_to_dict, dict_to_args


def update_tracker(ret_loss, tracker):
    for key, val in ret_loss.items():
        if isinstance(val, torch.Tensor):
            if key in tracker:
                tracker[key] = torch.cat((tracker[key], val.mean().data.unsqueeze(0)))
            else:
                tracker[key] = val.mean().data.unsqueeze(0)
        else:
            if key in tracker:
                tracker[key] = tracker[key] + val
            else:
                tracker[key] = val
    if "correct" in tracker:
        tracker['Acc'] = torch.Tensor([tracker["correct"] * 100.0 / tracker['count']])
    return tracker


def lr_transformer(args, i, lr0=0.1, ):
    return lr0 * 10 / math.sqrt(args.d_model) * min(1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))


def get_lr_anneal(args, train_iter):
    lr_end = 1e-5
    return max(0, (args.lr - lr_end) * (args.anneal_steps - train_iter) / args.anneal_steps) + lr_end


def update_lr(args, opt, train_iter):
    if args.lr_schedule == "fixed":
        opt.param_groups[0]['lr'] = args.lr
    elif args.lr_schedule == "anneal":
        opt.param_groups[0]['lr'] = get_lr_anneal(train_iter + 1)
    elif args.lr_schedule == "transformer":
        opt.param_groups[0]['lr'] = lr_transformer(train_iter + 1)
    return opt


def get_lr_schedule(is_better, model, optimizer, main_args, patience, num_trial,
                    model_file, reload_model=True):
    if is_better:
        patience = 0
        print('save model to [%s]' % model_file, file=sys.stdout)
        model.save(model_file)
        # also save the optimizers' state
        torch.save(optimizer.state_dict(), model_file + '.optim.bin')
    elif patience < main_args.patience:
        patience += 1
        print('hit patience %d' % patience, file=sys.stdout)

    if patience == main_args.patience:
        num_trial += 1
        print('hit #%d trial' % num_trial, file=sys.stdout)

        # decay lr, and restore from previously best checkpoint
        lr = optimizer.param_groups[0]['lr'] * main_args.lr_decay
        print('decay learning rate to %f' % lr, file=sys.stdout)
        # load model
        if reload_model:
            print('load previously best model', file=sys.stdout)
            params = torch.load(model_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if main_args.cuda:
                model = model.cuda()

        # load optimizers
        if main_args.reset_optimizer:
            print('reset optimizer', file=sys.stdout)
            optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
        else:
            print('restore parameters of the optimizers', file=sys.stdout)
            optimizer.load_state_dict(torch.load(model_file + '.optim.bin'))

        # set new lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # reset patience
        patience = 0
    lr = optimizer.param_groups[0]['lr']
    if lr <= 1e-6:
        print('early stop!', file=sys.stdout)

    return model, optimizer, num_trial, patience


def get_eval_info(main_args, model_args, mode='Train'):
    if main_args.exp_name is not None:
        dir_path = [model_args.model_select, main_args.exp_name]
    else:
        dir_path = [model_args.model_select]
    model_name = ".".join(dir_path)
    model_dir = os.path.join(main_args.model_dir, model_name)
    model_dir = os.path.join(model_dir, mode)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def get_model_info(main_args, model_args, check_dir=True):
    if main_args.exp_name is not None:
        dir_path = [model_args.model_select, main_args.exp_name]
    else:
        dir_path = [model_args.model_select]
    model_name = ".".join(dir_path)
    task_dir = os.path.join(main_args.model_dir, main_args.task_type)
    model_dir = os.path.join(task_dir, model_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        os.makedirs(model_dir)
    elif not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # model_dir = os.path.join(model_dir, "model")
    model_path_list = model_dir.split("/")
    log_path_list = [str(model_path_list[-2]), str(model_path_list[-1])]
    log_file = ".".join(log_path_list)
    # log_file = model_dir.split("/")[-2]
    log_dir = os.path.join(main_args.logdir, log_file)

    if check_dir:
        if os.path.exists(log_dir):
            if log_dir.endswith("debug"):
                shutil.rmtree(log_dir)
                print("Delete Log:\t{}".format(log_dir), file=sys.stderr)
            else:
                print("{} is not empty, del it?".format(log_dir))
                x = input("Y is del, other is not:")
                if x.lower() == "y":
                    shutil.rmtree(log_dir)
                    print("Delete Log:\t{}".format(log_dir), file=sys.stderr)
                else:
                    raise RuntimeError("Target is need redirection")

        if main_args.mode.lower().startswith("train"):
            base_params = {
                'base_configs': args_to_dict(main_args),
            }
            model_params = {
                'model_configs': args_to_dict(model_args),
            }

            dict_to_yaml(yaml_out_file=os.path.join(model_dir, "base_config.yaml"), dicts=base_params)
            dict_to_yaml(yaml_out_file=os.path.join(model_dir, "model_config.yaml"), dicts=model_params)

    print("Model Dir:\t{}\nLog Dir:\t{}".format(
        model_dir,
        log_dir
    ))

    return {
        'model_dir': model_dir,
        'log_dir': log_dir,
        "model_file": os.path.join(model_dir, "model.bin"),
        "out_dir": os.path.join(model_dir, "out"),
    }


def build_model(main_args, model_args, model=None):
    if model is None:
        if "pretrain_exp_dir" in model_args and model_args.pretrain_exp_dir is None:
            _main_args, _model_args = load_configs(model_args.pretrain_exp_dir)
            ptrn_model = load_model(main_args=_main_args, model_args=_model_args, check_dir=False, use_cuda=False)
            if ("share_tgt_embed" in model_args and model_args.share_tgt_embed) or (
                    "share_encoder" in model_args and model_args.share_encoder):
                encoder = ptrn_model.encoder if "share_encoder" in model_args and model_args.share_encoder else None
                tgt_embed = ptrn_model.decoder.embeddings if "share_tgt_embed" in model_args and model_args.share_tgt_embed else None
                vocab = ptrn_model.vocab
                model = init_create_model(model_args.model_select, args=model_args, vocab=vocab, encoder=encoder,
                                          tgt_embed=tgt_embed)
            else:
                model = ptrn_model
        else:
            vocab = Vocab.from_bin_file(main_args.vocab)
            model = init_create_model(model_args.model_select, args=model_args, vocab=vocab)
            for param in model.parameters():
                param.data.uniform_(-0.08, 0.08)
    vocab = model.vocab
    model.train()
    if model_args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=main_args.lr, betas=(0.9, 0.995))
    return model, optimizer, vocab


def load_configs(exp_dir):
    pre_main_args = dict_to_args(yaml_to_dict(yaml_in_file=os.path.join(exp_dir, "base_config.yaml"))[
                                     'base_configs'])
    pre_model_args = dict_to_args(yaml_to_dict(yaml_in_file=os.path.join(exp_dir, "model_config.yaml"))[
                                      'model_configs'])
    return pre_main_args, pre_model_args


def load_data(main_args):
    train_set = Dataset.from_bin_file(main_args.train_file)
    dev_set = Dataset.from_bin_file(main_args.dev_file)
    return train_set, dev_set


def load_model(main_args, model_args, check_dir=True, use_cuda=None):
    use_cuda = model_args.cuda if use_cuda is None else use_cuda
    dir_ret = get_model_info(main_args=main_args, model_args=model_args, check_dir=check_dir)
    model_file = dir_ret['model_file']
    print("Load From:\t{}".format(model_file))
    model = load_static_model(model=model_args.model_select, model_path=model_file)
    if use_cuda:
        model.cuda()
    return model
