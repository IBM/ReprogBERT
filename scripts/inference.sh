#!/usr/bin/env bash
cd ../
python main.py  --run inference \
                --single_input QVQLVESGGGFAQAGGSLRLSCAAS********MGWFRQAPGKEREFVAGISWSGSTKYTDSVKGRFTISRDNAKNTVHLQMNNLTPEDTAVYYCAQSRAIEADDSRGYDYWGQGTQVT \
                --model_type reprog \
                --exp_dir reprog_cdr3 \
                --checkpoint output/reprog_cdr3/checkpoints/<checkpoint_file>.ckpt \
                --progen_dir /path/to/progen/progen2
