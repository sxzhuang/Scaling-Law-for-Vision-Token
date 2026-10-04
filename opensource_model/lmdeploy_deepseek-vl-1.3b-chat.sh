#!/bin/bash
#SBATCH --job-name=dpvl  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --gres=gpu:1                                  # 每节点申请1块GPU
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB

cd /home/sxzhuangch/lmdeploy_server
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MEDIA_DIR="/slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token"
HTTP_PORT=18080

# 1) 起静态文件服务（后台）
python -m http.server "${HTTP_PORT}" \
  --bind 127.0.0.1 \
  --directory "${MEDIA_DIR}" \
  >/dev/null 2>&1 &
HTTP_PID=$!

# 2) 健康检查：确认端口真的起来了
for i in {1..50}; do
  if curl -fsS "http://127.0.0.1:${HTTP_PORT}/" >/dev/null 2>&1; then
    echo "[INFO] http.server is up on 127.0.0.1:${HTTP_PORT}"
    break
  fi
  sleep 0.1
done

if ! curl -fsS "http://127.0.0.1:${HTTP_PORT}/" >/dev/null 2>&1; then
  echo "[ERROR] http.server failed to start. Log:"
  tail -n 200 "SLURM_Logs/${SLURM_JOB_ID}_http.out" || true
  exit 1
fi

# 3) 作业退出时清理
trap 'kill ${HTTP_PID} 2>/dev/null || true' EXIT


export CUDA_VISIBLE_DEVICES=0

lmdeploy serve api_server deepseek-ai/deepseek-vl-1.3b-chat \
  --server-name 0.0.0.0 \
  --server-port 50018 \
  --session-len 16384

wait