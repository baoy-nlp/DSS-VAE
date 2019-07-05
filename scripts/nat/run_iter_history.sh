#!/usr/bin/env bash
# for iwslt-ende iterative model
python3 run.py --dataset iwslt-ende --vocab_size 40000 --load_vocab --ffw_block highway --params small --batch_size 2048 --num_gpus 2 --eval_every 1000 --lr_schedule anneal --fast --valid_repeat_dec 20 --use_argmax --next_dec_input both --denoising_prob 0.5 --layerwise_denoising_weight --use_distillation --save_every 1000 --save_last

# for iwslt-deen iterative model
python3 run.py --dataset iwslt-deen --vocab_size 40000 --load_vocab --ffw_block highway --params small --batch_size 2048 --num_gpus 2 --eval_every 1000 --lr_schedule anneal --fast --valid_repeat_dec 20 --use_argmax --next_dec_input both --denoising_prob 0.5 --layerwise_denoising_weight --use_distillation --save_every 1000 --save_last

# baseline deen repeat = 1
python3 run.py --dataset iwslt-deen --vocab_size 40000 --load_vocab --ffw_block highway --params small --batch_size 2048 --num_gpus 2 --eval_every 1000 --lr_schedule anneal --fast --valid_repeat_dec 1 --train_repeat_dec 2 --use_argmax --next_dec_input both --denoising_prob 0.5 --layerwise_denoising_weight --use_distillation --save_every 1000 --save_last

# baseline ende repeat = 1
python3 run.py --dataset iwslt-ende --vocab_size 40000 --load_vocab --ffw_block highway --params small --batch_size 2048 --num_gpus 2 --eval_every 1000 --lr_schedule anneal --fast --valid_repeat_dec 1 --train_repeat_dec 2 --use_argmax --next_dec_input both --denoising_prob 0.5 --layerwise_denoising_weight --use_distillation --save_every 1000 --save_last

