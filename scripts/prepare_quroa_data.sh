#!/usr/bin/env bash
cd ..

DATA_DIR=/data00/home/baoyu.nlp/quora-train
OUT_DIR=/mnt/cephfs_wj/common/lab/baoyu.nlp/data/quora-for-syntax-vae
mkdir ${OUT_DIR}

python -m preprocess.prepare_raw_set --src_file ${DATA_DIR}/train.con --out_file ${DATA_DIR}/train.txt --data_mode SyntaxVAE
python -m preprocess.prepare_raw_set --src_file ${DATA_DIR}/dev.con --out_file ${DATA_DIR}/dev.txt --data_mode SyntaxVAE
python -m preprocess.prepare_raw_set --src_file ${DATA_DIR}/test.con --out_file ${DATA_DIR}/test.txt --data_mode SyntaxVAE
python -m structs.generate_dataset \
--train_file ${DATA_DIR}/train.txt \
--dev_file ${DATA_DIR}/dev.txt \
--test_file ${DATA_DIR}/dev.txt \
--out_dir ${OUT_DIR}/ \
--max_src_vocab 100 \
--max_src_len 30