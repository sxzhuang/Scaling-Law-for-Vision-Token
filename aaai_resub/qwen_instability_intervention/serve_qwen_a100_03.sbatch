#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

cd /slurm/shuxin_zhuang/tests1

PORT=50120
NODE_NAME=$(hostname)
SERVICE_ROOT="aaai_resub/qwen_instability_intervention/services"
mkdir -p "$SERVICE_ROOT"

if ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; then
    echo "Port ${PORT} is already in use on ${NODE_NAME}" >&2
    exit 1
fi

printf '{\n  "job_id": "%s",\n  "node": "%s",\n  "port": %s,\n  "endpoint": "http://%s:%s/v1",\n  "model": "Qwen/Qwen2.5-VL-7B-Instruct"\n}\n' "$SLURM_JOB_ID" "$NODE_NAME" "$PORT" "$NODE_NAME" "$PORT" >"$SERVICE_ROOT/${SLURM_JOB_ID}.json"

source /home/sxzhuangch/vllm_server/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME=/slurm/shuxin_zhuang/.cache/huggingface
export HF_HUB_CACHE=/slurm/shuxin_zhuang/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve --config opensource_model/configs/Qwen2.5_VL_7B_Instruct.yaml --host 0.0.0.0 --port "$PORT"
