#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}

python3 dss_main.py \
--base_config /mnt/cephfs_wj/common/lab/baoyu.nlp/projects/non_auto_gen/configs/model_configs/${1}.yaml \
--model_config /mnt/cephfs_wj/common/lab/baoyu.nlp/projects/non_auto_gen/configs/model_configs/${2}.yaml \
--mode train \
--exp_name ${3}

