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

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token
mkdir -p SLURM_Logs aaai_resub/lambda_scaling_model/outputs/fixed_original_intercepts

python aaai_resub/lambda_scaling_model/fit_lambda_from_fixed_original_intercepts.py \
    --csv_path data_analysis/data4analysis.csv \
    --original_results_dir data_analysis/em_fiting_res/shared_exceptw \
    --output_dir aaai_resub/lambda_scaling_model/outputs/fixed_original_intercepts \
    --resolutions 512,640,1024,1280
