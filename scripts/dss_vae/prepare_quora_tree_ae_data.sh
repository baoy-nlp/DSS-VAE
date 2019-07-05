#!/usr/bin/env bash
cd ..

DATA_DIR=/data00/home/baoyu.nlp/data/Quora/unsupervised
#python -m preprocess.prepare_raw_set --src_file ${DATA_DIR}/quora_corpus_unsupervised.nltk.parse --out_file ${DATA_DIR}/whole.ae.tree --data_mode SyntaxVAE --convert_mode Convert # for tree-ae
#python -m preprocess.prepare_raw_set --src_file ${DATA_DIR}/whole.ae.tree --tgt_file ${DATA_DIR}/whole.ae.tree --out_file ${DATA_DIR}/whole.tree.pair.txt --data_mode NAG
#tail -n+10001 ${DATA_DIR}/whole.tree.pair.txt > ${DATA_DIR}/train.pair.tree
#head -10000 ${DATA_DIR}/whole.tree.pair.txt > ${DATA_DIR}/dev.pair.tree
#python -m structs.generate_dataset --train_file ${DATA_DIR}/train.pair.tree --dev_file ${DATA_DIR}/dev.pair.tree --test_file ${DATA_DIR}/dev.pair.tree --out_dir ${DATA_DIR}/ --max_src_vocab 100 --max_src_len 100