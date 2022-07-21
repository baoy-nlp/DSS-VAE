#!/usr/bin/env bash
cd ..
# tokenize
#mkdir ../Quora-Pro/
#mkdir ../Quora-Pro/Sup
#mkdir ../Quora-Pro/Sup/Para

#QUORA_DIR=../Quora/supervised/
#PRO_DIR=../Quora-Pro/
#SUP_DIR=${PRO_DIR}/Sup
PARA_DIR=/data00/home/baoyu.nlp/data/Quora-Pro/Sup/Para
Paraphrase_dir=/data00/home/baoyu.nlp/data/Paraphrase


#python preprocess/my_tokenize.py --raw_file ${QUORA_DIR}/quora_train_par.tok.lower --token_file ../Quora-Pro/Sup/train.par.token --for_parse
#python preprocess/my_tokenize.py --raw_file ${QUORA_DIR}/quora_train_origin.tok.lower --token_file ../Quora-Pro/Sup/train.origin.token --for_parse
#python preprocess/my_tokenize.py --raw_file ${QUORA_DIR}/quora_test_par.tok.lower --token_file ../Quora-Pro/Sup/test.par.token --for_parse
#python preprocess/my_tokenize.py --raw_file ${QUORA_DIR}/quora_test_origin.tok.lower --token_file ../Quora-Pro/Sup/test.origin.token --for_parse
#python preprocess/my_tokenize.py --raw_file ${QUORA_DIR}/quora_val_par.tok.lower --token_file ../Quora-Pro/Sup/dev.par.token --for_parse
#python preprocess/my_tokenize.py --raw_file ${QUORA_DIR}/quora_val_origin.tok.lower --token_file ../Quora-Pro/Sup/dev.origin.token --for_parse

# convert to <sentence,sentence>

#python preprocess/prepare_raw_set.py --data_mode syntax-gen --src_tree_file ../Quora-Pro/Sup/train.origin.token --para_tree_file ../Quora-Pro/Sup/train.par.token --out_file ../Quora-Pro/Sup/Para/train.para
#python preprocess/prepare_raw_set.py --data_mode syntax-gen --src_tree_file ../Quora-Pro/Sup/dev.origin.token --para_tree_file ../Quora-Pro/Sup/dev.par.token --out_file ../Quora-Pro/Sup/Para/dev.para
#python preprocess/prepare_raw_set.py --data_mode syntax-gen --src_tree_file ../Quora-Pro/Sup/test.origin.token --para_tree_file ../Quora-Pro/Sup/test.par.token --out_file ../Quora-Pro/Sup/Para/test.para

# generate data

python structs/generate_dataset.py --mode Plain --train_file ${PARA_DIR}/train.para --dev_file ${PARA_DIR}/dev.para --test_file ${PARA_DIR}/test.para --tgt_dir ${Paraphrase_dir} --max_src_vocab 30000 --max_tgt_vocab 30000 --vocab_freq_cutoff 1 --max_src_len 50 --max_tgt_len 50 --train_size -1

