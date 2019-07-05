#!/usr/bin/env bash
cd ..
MT_Ori_DIR=/mnt/cephfs_wj/common/lab/baoyu.nlp/data/iwslt/en-de
MT_Tgt_DIR=/mnt/cephfs_wj/common/lab/baoyu.nlp/data/iwslt/word_level

mkdir ${MT_Tgt_DIR}

python -m preprocess.my_tokenize --raw_file ${MT_Ori_DIR}/train/train.tags.en-de.bpe.en --token_file ${MT_Tgt_DIR}/train.en.token.lower --for_bpe --language english
python -m preprocess.my_tokenize --raw_file ${MT_Ori_DIR}/train/train.tags.en-de.bpe.de --token_file ${MT_Tgt_DIR}/train.de.token.lower --for_bpe --language german
#python -m preprocess.my_tokenize --raw_file ${MT_Ori_DIR}/dev/valid.en-de.bpe.en --token_file ${MT_Tgt_DIR}/valid.en.token.lower --for_bpe --is_lower --language english
#python -m preprocess.my_tokenize --raw_file ${MT_Ori_DIR}/dev/valid.en-de.bpe.de --token_file ${MT_Tgt_DIR}/valid.de.token.lower --for_bpe --is_lower --language german
#python -m preprocess.my_tokenize --raw_file ${MT_Ori_DIR}/test/test.en --token_file ${MT_Tgt_DIR}/test.en.token.lower --is_lower --language english
#python -m preprocess.my_tokenize --raw_file ${MT_Ori_DIR}/test/test.de --token_file ${MT_Tgt_DIR}/test.de.token.lower --is_lower --language german
#
# convert to <sentence,sentence>
#python -m preprocess.prepare_raw_set --data_mode NAG --src_file ${MT_Tgt_DIR}/train.en.token.lower --tgt_file ${MT_Tgt_DIR}/train.de.token.lower --out_file ${MT_Tgt_DIR}/train.token.lower
#python -m preprocess.prepare_raw_set --data_mode NAG --src_file ${MT_Tgt_DIR}/valid.en.token.lower --tgt_file ${MT_Tgt_DIR}/valid.de.token.lower --out_file ${MT_Tgt_DIR}/dev.token.lower
#python -m preprocess.prepare_raw_set --data_mode NAG --src_file ${MT_Tgt_DIR}/test.en.token.lower --tgt_file ${MT_Tgt_DIR}/test.de.token.lower --out_file ${MT_Tgt_DIR}/test.token.lower
#
#
# generate data
#
#python -m structs.generate_dataset --mode Plain --train_file ${MT_Tgt_DIR}/train.token.lower --dev_file ${MT_Tgt_DIR}/dev.token.lower --test_file ${MT_Tgt_DIR}/test.token.lower --out_dir ${MT_Tgt_DIR} --max_src_vocab 40000 --max_tgt_vocab 40000 --vocab_freq_cutoff 1 --max_src_len -1 --max_tgt_len -1 --train_size -1
