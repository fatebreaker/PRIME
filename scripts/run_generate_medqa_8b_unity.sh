#!/bin/bash
#SBATCH --job-name=medqa_raise
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --partition=gypsum-rtx8000
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=./medqa_raise_8b_%j.out
#SBATCH --error=./medqa_raise_8b_%j.error
python3 run_src/do_raise.py \
    --dataset_name MedQA \
    --test_json_filename test_all \
    --model_ckpt /project/pi_hongyu_umass_edu/shared_llm_checkpoints/Meta-Llama-3.1-8B-Instruct \
    --note default \
    --num_rollouts 1 \
    --mode run \
    --half_precision \
    --disable_a5 \
    --disable_a6 \
    --disable_a7
