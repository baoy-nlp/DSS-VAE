#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}

EXP_PATH=/mnt/cephfs_hl/bytetrans/baoyu.nlp/experiments/iter-ablation # MODEL SAVED PATH
DATA_PATH=/mnt/cephfs_hl/bytetrans/baoyu.nlp/data # DATA LOADED PATH

python3 nonauto_run.py \
--main_path ${EXP_PATH} \
--data_dir ${DATA_PATH} \
--dataset iwslt-deen \
--model_cls PositionalFasterTransformer \
--exp_desc ablation-pos${1}_${2}-attn-learnpos1-EMB_${3} \
--pos_pred_type ${1} --pos_predict_scale ${2} --content_scale ${3} \
--pos_search_type 1 --pos_learn_embed \
--decoder_input_how interpolate --decoder_query_select hidden --use_distillation \
--pos_match_norm --ar_supervised --max_rel_len 4 \
--load_vocab --params my --ffw_block highway --share_pos_repr --share_pos_pred \
--pos_n_layers 2 --pos_predict_layer 1 0 0 --pos_index_out --valid_with_teacher --pos_order_scale 0 \
--keep_grad_for_pos_pred --keep_grad_for_pos_match \
--layer_weight 0 0 --content_loss_type emb --pos_index_state \
--state_bow_loss --state_weight 1 0 0 --state_window_size 4 0 0 --state_share_output \
--pos_map_select f_double --final_pos_map_select f_double \
--pos_attn_type default --lr 0.0003 --pos_rel_pred_scale 0.0 \
--proc_pos_valid_use pred --final_pos_valid_use pred \
--pos_enc_cls attn --share_vocab --share_embed

