#!/bin/bash
#SBATCH --job-name=medmcqa_rstar
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --partition=gypsum-rtx8000
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=./medmcqa_rstar_%j.out
#SBATCH --error=./medmcqa_rstar_%j.error
python3 run_src/do_raise.py \
    --dataset_name MedMCQA \
    --test_json_filename test_all \
    --model_ckpt /project/pi_hongyu_umass_edu/shared_llm_checkpoints/Meta-Llama-3.1-8B-Instruct \
    --note default \
    --num_rollouts 8 \
    --mode run \
    --disable_a7 \
    --disable_a6

