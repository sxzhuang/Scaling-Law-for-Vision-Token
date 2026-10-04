#!/bin/bash

set -euo pipefail

cd /slurm/shuxin_zhuang/workspace/Scaling-Law-for-Vision-Token

SBATCH_SCRIPT="aaai_resub/qwen_instability_intervention/render_pride_long_context_shard.sbatch"
SHARD_ROOT="aaai_resub/qwen_instability_intervention/pride_long_context/data/shards"

for index in 0 1 2; do
    shard="${SHARD_ROOT}/pride_25k_34k_part_${index}.json"
    job_id=$(sbatch --parsable --export=ALL,INPUT_JSON="$shard" "$SBATCH_SCRIPT")
    printf 'shared_part_%s=%s\n' "$index" "$job_id"
done

for index in 0 1 2 3 4; do
    shard="${SHARD_ROOT}/pride_34k_48k_part_${index}.json"
    job_id=$(sbatch --parsable --export=ALL,INPUT_JSON="$shard" "$SBATCH_SCRIPT")
    printf 'extra_part_%s=%s\n' "$index" "$job_id"
done
