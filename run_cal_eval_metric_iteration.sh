#!/bin/bash
#SBATCH --job-name=cal_metric
#SBATCH --output=SLURM_Logs/%j.out
#SBATCH --error=SLURM_Logs/%j.err
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

# Resolution parameter (default 1024, override via $1)
RESOLUTION=1280

# Define parameter arrays for density experiment
FONT_SIZES=(20 28 36)
LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)

echo "Starting metric calculation for density experiment folders with resolution: ${RESOLUTION}"

# Triple-nested loop to iterate over all combinations
for font_size in "${FONT_SIZES[@]}"; do
    for line_spacing in "${LINE_SPACINGS[@]}"; do
        for letter_spacing in "${LETTER_SPACINGS[@]}"; do
            # Construct dataset name from parameters
            DATASET_NAME="${font_size}_${line_spacing}_${letter_spacing}"
            
            # Build paths dynamically
            json_for_eval="eval_results/pride_square_density_experiment/${DATASET_NAME}/${RESOLUTION}/prepare_json_for_eval.json"
            output_file="eval_results/pride_square_density_experiment/${DATASET_NAME}/${RESOLUTION}/dpsk_eval_metric.json"
            
            echo "Processing: ${DATASET_NAME} @ resolution ${RESOLUTION}"
            
            # Calculate evaluation metrics
            python3 evaluation_dpsk_ocr_predictions.py \
                --json_for_eval "${json_for_eval}" \
                --output_file "${output_file}"
            
            echo "Completed: ${DATASET_NAME}"
        done
    done
done

echo "All metric calculations completed for resolution ${RESOLUTION}"