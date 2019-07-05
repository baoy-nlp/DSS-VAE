#!/usr/bin/env bash
"autoregressive model: split vocab"
sh debug.sh BYTETRANS dialog-inout Transformer " --vocab_size 40000 --save_dataset --ffw_block highway --lr_schedule anneal"

