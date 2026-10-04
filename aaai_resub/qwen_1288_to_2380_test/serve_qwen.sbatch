#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

PORT=50117
SERVICE_ROOT="aaai_resub/qwen_1288_to_2380_test/service"
mkdir -p "$SERVICE_ROOT"
printf 'http://%s:%s/v1\n' "$(hostname)" "$PORT" >"$SERVICE_ROOT/endpoint.txt"
printf '%s\n' "$SLURM_JOB_ID" >"$SERVICE_ROOT/job_id.txt"

source /home/sxzhuangch/vllm_server/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME=/slurm/shuxin_zhuang/.cache/huggingface
export HF_HUB_CACHE=/slurm/shuxin_zhuang/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve --config opensource_model/configs/Qwen2.5_VL_7B_Instruct.yaml --host 0.0.0.0 --port "$PORT"
