#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

TEST_ROOT="aaai_resub/smoke_tests/novels_longest10_16_60_-1"
for resolution in 512 640 1024 1280; do
    image_dir="${TEST_ROOT}/images/${resolution}"
    metadata_path="${TEST_ROOT}/selected_samples/${resolution}/selected_metadata.json"
    prediction_dir="${TEST_ROOT}/ocr_predictions/${resolution}"
    result_dir="${TEST_ROOT}/eval_results/${resolution}"
    eval_json="${result_dir}/prepare_json_for_eval.json"
    metric_json="${result_dir}/dpsk_eval_metric.json"

    python -m run_dpsk_ocr_eval_batch \
        --base_size "${resolution}" \
        --image_size "${resolution}" \
        --input_path "${image_dir}" \
        --output_path "${prediction_dir}/" \
        --batch_size 10

    python build_dataset_from_ebook/build_eval_json.py \
        --images_dir "${image_dir}" \
        --metadata_path "${metadata_path}" \
        --predictions_dir "${prediction_dir}" \
        --output_path "${eval_json}"

    python evaluation_dpsk_ocr_predictions.py \
        --json_for_eval "${eval_json}" \
        --output_file "${metric_json}"
done
