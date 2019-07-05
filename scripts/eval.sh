#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}

DEV_MODEL_DIR=/data00/home/baoyu.nlp/run_models/Autoencoder/Seq2seq.tree-ae-debug
DATA_DIR=/data00/home/baoyu.nlp/ext

python3 main.py --mode test \
--load_from ${DEV_MODEL_DIR} \
--test_dir ${DATA_DIR}/${1}
