from __future__ import absolute_import

import argparse

from dss_vae.bin.test import testing
from dss_vae.bin.train import training
from dss_vae.utils.config_funcs import dict_to_args
from dss_vae.utils.config_funcs import yaml_to_dict


def process_args():
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--base_config', type=str, help='basic configs')
    opt_parser.add_argument('--model_config', type=str, help='models configs')
    opt_parser.add_argument('--exp_name', type=str, help='config_files')
    opt_parser.add_argument('--mode', type=str, default=None)
    opt_parser.add_argument('--load_from', type=str, default=None)
    opt_parser.add_argument('--test_dir', type=str, default=None)

    opt = opt_parser.parse_args()
    main_args, model_args = None, None

    if opt.base_config is not None:
        main_args = dict_to_args(yaml_to_dict(opt.base_config)['base_configs'])
        main_args.mode = opt.mode
        if opt.exp_name is not None:
            main_args.exp_name = opt.exp_name
    if opt.model_config is not None:
        model_args = dict_to_args(yaml_to_dict(opt.model_config)['model_configs'])

    return {
        'base': main_args,
        'model': model_args,
        "opt": opt
    }


if __name__ == "__main__":
    config_args = process_args()
    args = config_args['opt']

    if args.mode.startswith("train"):
        training(config_args['base'], config_args['model'])
    elif args.mode.startswith("test"):
        testing(ptrn_dir=args.load_from, test_dir=args.test_dir)
