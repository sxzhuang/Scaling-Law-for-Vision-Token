#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

cd /slurm/shuxin_zhuang/tests1
source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr
export PYTHONPATH=/slurm/shuxin_zhuang/tests1

python aaai_resub/qwen_instability_intervention/evaluate_pride_long_context.py
python aaai_resub/qwen_instability_intervention/plot_qwen_504_644_ed_vs_z.py
