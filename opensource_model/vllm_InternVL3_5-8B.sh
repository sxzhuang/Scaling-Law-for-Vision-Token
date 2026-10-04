#!/bin/bash
#SBATCH --job-name=25vl  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100                          
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --gres=gpu:1                                  # 每节点申请1块GPU
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=6GB

cd /home/sxzhuangch/vllm_server
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export HF_HOME=/slurm/shuxin_zhuang/.cache/huggingface
export HF_HUB_CACHE=/slurm/shuxin_zhuang/.cache/huggingface/hub

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve \
    --config /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token/opensource_model/configs/InternVL3_5-8B.yaml

wait
