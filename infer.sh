#!/bin/bash

# Ensure we are in the script's directory, which should be HunyuanVideo-Avatar
cd "$(dirname "$0")"

JOBS_DIR=$(dirname $(dirname "$0")) # Now $0 correctly refers to this script
export PYTHONPATH=./

export MODEL_BASE=./weights
OUTPUT_BASEPATH=./results-single # Note: not exported, but used by checkpoint_path and save-path
checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt

export DISABLE_SP=1
CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py \
    --input 'assets/test_2.csv' \
    --ckpt "${checkpoint_path}" \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path "${OUTPUT_BASEPATH}" \
    --use-fp8 \
    --infer-min

echo "Inference complete. Results saved to ${OUTPUT_BASEPATH}"