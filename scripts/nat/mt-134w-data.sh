#!/usr/bin/env bash
cd ..
MT_Ori_DIR=/mnt/cephfs_wj/bytetrans/baoyu.nlp/data/134w
MT_Tgt_DIR=/mnt/cephfs_wj/bytetrans/baoyu.nlp/data/zh-en-r80

mkdir ${MT_Tgt_DIR}

# convert to <sentence,sentence>

python preprocess/prepare_raw_set.py --data_mode syntax-gen --src_tree_file ${MT_Ori_DIR}/zh.shuf --para_tree_file ${MT_Ori_DIR}/en.shuf --out_file ${MT_Tgt_DIR}/train.mt
python preprocess/prepare_raw_set.py --data_mode syntax-gen --src_tree_file ${MT_Ori_DIR}/testsets/mt03.src --para_tree_file ${MT_Ori_DIR}/testsets/mt03.ref0 --out_file ${MT_Tgt_DIR}/dev.mt
python preprocess/prepare_raw_set.py --data_mode syntax-gen --src_tree_file ${MT_Ori_DIR}/testsets/mt03.src --para_tree_file ${MT_Ori_DIR}/testsets/mt03.ref1 --out_file ${MT_Tgt_DIR}/test.mt

# generate data

python structs/generate_dataset.py --mode Plain --train_file ${MT_Tgt_DIR}/train.mt --dev_file ${MT_Tgt_DIR}/dev.mt --test_file ${MT_Tgt_DIR}/test.mt --tgt_dir ${MT_Tgt_DIR} --max_src_vocab 30000 --max_tgt_vocab 30000 --vocab_freq_cutoff 1 --max_src_len 80 --max_tgt_len 80 --train_size -1

cp ${MT_Ori_DIR}/testsets/mt03.ref0 ${MT_Tgt_DIR}/mt03.ref0
cp ${MT_Ori_DIR}/testsets/mt03.ref1 ${MT_Tgt_DIR}/mt03.ref1
cp ${MT_Ori_DIR}/testsets/mt03.ref2 ${MT_Tgt_DIR}/mt03.ref2
cp ${MT_Ori_DIR}/testsets/mt03.ref3 ${MT_Tgt_DIR}/mt03.ref3