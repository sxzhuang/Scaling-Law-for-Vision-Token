#!/bin/bash
#SBATCH --job-name=vqa_metric
#SBATCH --output=SLURM_Logs/%j.out
#SBATCH --error=SLURM_Logs/%j.err
#SBATCH --partition=a100,a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

python VQAExp/cal_metric.py \
    --gt_json VQAExp/data/Retrieval/retrieval_cropped_0_25k_600.json \
    --predictions_jsonl VQAExp/vqa_predictions/Qwen/Qwen2.5-VL-7B-Instruct/Retrieval/28_6_0/vqa_predictions_res_640.jsonl \
    --output_dir VQAExp/eval_results/retrieval/qwen25vl7b/28_6_0 \
    --output_prefix res640
