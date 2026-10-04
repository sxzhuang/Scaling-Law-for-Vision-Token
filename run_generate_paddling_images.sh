#!/bin/bash
#SBATCH --job-name=generate_paddling_images
#SBATCH --output=SLURM_Logs/%j.out
#SBATCH --error=SLURM_Logs/%j.err
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8GB

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

python3 utils/paddling_shifted_inputs.py \
    --folder build_dataset_from_ebook/pride_dataset_square_large_variance/selected_from640/images \
    --resolution 640
