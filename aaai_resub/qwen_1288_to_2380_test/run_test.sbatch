#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.out
#SBATCH --error=/slurm/shuxin_zhuang/tests1/SLURM_Logs/%j.err
#SBATCH --chdir=/slurm/shuxin_zhuang/tests1
#SBATCH --partition=a100,a30
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8GB

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

TEST_ROOT="aaai_resub/qwen_1288_to_2380_test"
DATA_ROOT="$TEST_ROOT/data"
PREDICTIONS_ROOT="$TEST_ROOT/predictions"
EVALUATION_ROOT="$TEST_ROOT/evaluation"
SERVER_LOG="$TEST_ROOT/vllm_server_${SLURM_JOB_ID}.log"
PORT=50017

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

python "$TEST_ROOT/prepare_test_samples.py"
rm -rf "$DATA_ROOT/images_2380"
python utils/image_resize.py --input_path "$DATA_ROOT/raw_images" --output_path "$DATA_ROOT/images_2380" --resolution 2380

source /home/sxzhuangch/vllm_server/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HOME=/slurm/shuxin_zhuang/.cache/huggingface
export HF_HUB_CACHE=/slurm/shuxin_zhuang/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve --config opensource_model/configs/Qwen2.5_VL_7B_Instruct.yaml --host 127.0.0.1 --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM server exited before becoming ready; inspect $SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr
rm -rf "$PREDICTIONS_ROOT/uniform" "$EVALUATION_ROOT/uniform"
python VQAExp/run_ocr.py --input_path "$DATA_ROOT/images_2380" --output_path "$PREDICTIONS_ROOT/uniform/res_2380" --url "http://127.0.0.1:${PORT}/v1" --model Qwen/Qwen2.5-VL-7B-Instruct --metadata_json "$DATA_ROOT/metadata.json" --max_tokens 8192 --max_tokens_ratio 1.2 --max_tokens_margin 32 --min_max_tokens 64 --image_url_mode file --concurrency 4
python VQAExp/cal_ocr_edit_distance.py --gt_json "$DATA_ROOT/metadata.json" --predictions_root "$PREDICTIONS_ROOT" --output_root "$EVALUATION_ROOT" --resolutions 2380
python "$TEST_ROOT/summarize_test.py"
