#!/bin/bash
#SBATCH --job-name=render_vqa_retrieval
#SBATCH --output=SLURM_Logs/%j.out
#SBATCH --error=SLURM_Logs/%j.err
#SBATCH --partition=a100,a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

python3 VQAExp/render_retrieval_context_images.py \
    --input_json VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json \
    --output_root VQAExp/data \
    --dataset_name Retrieval \
    --font_size 28 \
    --line_spacing 6 \
    --letter_spacing 0 \
    --padding 20 \
    --dpi 300