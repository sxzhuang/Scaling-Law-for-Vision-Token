#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%A_%a.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a30
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-1

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

PORT=$((50118 + SLURM_ARRAY_TASK_ID))
NODE_NAME=$(hostname)
SERVICE_ROOT="aaai_resub/qwen_instability_intervention/services"
INSTANCE_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SERVICE_ROOT"

if ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; then
    echo "Port ${PORT} is already in use on ${NODE_NAME}" >&2
    exit 1
fi

printf '{\n  "job_id": "%s",\n  "array_job_id": "%s",\n  "array_task_id": %s,\n  "node": "%s",\n  "port": %s,\n  "endpoint": "http://%s:%s/v1",\n  "model": "Qwen/Qwen2.5-VL-7B-Instruct"\n}\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$NODE_NAME" "$PORT" "$NODE_NAME" "$PORT" >"$SERVICE_ROOT/${INSTANCE_ID}.json"

source /home/sxzhuangch/vllm_server/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME=/slurm/shuxin_zhuang/.cache/huggingface
export HF_HUB_CACHE=/slurm/shuxin_zhuang/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve --config opensource_model/configs/Qwen2.5_VL_7B_Instruct.yaml --host 0.0.0.0 --port "$PORT"
