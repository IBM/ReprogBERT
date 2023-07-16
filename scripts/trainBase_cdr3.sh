#!/usr/bin/env bash
cd ../
python main.py --dataset sabdab3 \
               --model_type base \
               --exp_dir base_cd3 \
               --run train \
               --accelerator gpu \
               --num_nodes 1  \
               --lr 1e-5 \
               --bsize 32 \
               --bsize_eval 32 \
               --bar 10
