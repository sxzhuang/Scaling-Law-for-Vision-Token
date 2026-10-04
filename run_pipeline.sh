#!/bin/bash
# Sequential SLURM pipeline: each job starts only after the previous one succeeds.

set -e

JOB1=$(sbatch run_generate_dataset_square.sh | awk '{print $NF}')
echo "Submitted run_generate_dataset_square.sh  -> Job ID: $JOB1"

JOB2=$(sbatch --dependency=afterok:$JOB1 run_ds_ocr_eval_batch.sh | awk '{print $NF}')
echo "Submitted run_ds_ocr_eval_batch.sh        -> Job ID: $JOB2 (depends on $JOB1)"

JOB3=$(sbatch --dependency=afterok:$JOB2 run_prepare_eval_json.sh | awk '{print $NF}')
echo "Submitted run_prepare_eval_json.sh        -> Job ID: $JOB3 (depends on $JOB2)"

JOB4=$(sbatch --dependency=afterok:$JOB3 run_cal_eval_metric.sh | awk '{print $NF}')
echo "Submitted run_cal_eval_metric.sh          -> Job ID: $JOB4 (depends on $JOB3)"

echo ""
echo "Pipeline submitted: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4"
echo "Use 'squeue -u \$USER' to monitor progress."
