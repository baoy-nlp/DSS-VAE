from argparse import Namespace

import yaml


def dict_to_args(dicts):
    return Namespace(**dicts)


def args_to_dict(args):
    y = vars(args)
    return y


def yaml_to_dict(yaml_in_file):
    with open(yaml_in_file, 'r', encoding='utf-8') as f:
        arg_dicts = yaml.load(f.read())
    return arg_dicts


def dict_to_yaml(yaml_out_file, dicts):
    with open(yaml_out_file, "w") as f:
        yaml.dump(dicts, f, default_flow_style=False)


def args_to_yaml(yaml_out_file, args):
    dict_to_yaml(yaml_out_file=yaml_out_file, dicts=args_to_dict(args))


def yaml_to_args(yaml_in_file):
    dicts = yaml_to_dict(yaml_in_file=yaml_in_file)
    args = dict_to_args(dicts)
    return args

# if __name__ == "__main__":
#     """
#     Test dict_to_args
#     """
#
#     x = yaml_load_dict("../configs/ptb.yaml")
#
#     args = dict_to_args(x)
#
#     dicts = args_to_dict(args)
#
#     dict_to_yaml(fname=".test", dicts=dicts)
#     print(args)
#     print(dicts)
