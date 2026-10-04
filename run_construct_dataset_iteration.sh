#!/bin/bash
#SBATCH --job-name=construct_features
#SBATCH --output=SLURM_Logs/%j.out
#SBATCH --error=SLURM_Logs/%j.err
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

# Resolution parameter (default 1024, override via $1)
RESOLUTION=${1:-1024}

# Define parameter arrays for density experiment
FONT_SIZES=(20 28 36)
LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)

# Output CSV path

echo "Starting feature construction for density experiment folders with resolution: ${RESOLUTION}"

# Triple-nested loop to iterate over all combinations
for font_size in "${FONT_SIZES[@]}"; do
    for line_spacing in "${LINE_SPACINGS[@]}"; do
        for letter_spacing in "${LETTER_SPACINGS[@]}"; do
            # Construct dataset name from parameters
            DATASET_NAME="${font_size}_${line_spacing}_${letter_spacing}"
            
            # Build paths dynamically
            density_json="eval_results/pride_square_density_experiment/${DATASET_NAME}/${RESOLUTION}/ed_pd_len.json"
            meta_data="build_dataset_from_ebook/pride_dataset_square/pride_square_gt.json"
            edit_distance_json="eval_results/pride_square_density_experiment/${DATASET_NAME}/${RESOLUTION}/dpsk_eval_metric.json"
            image_dir="build_dataset_from_ebook/pride_dataset_square_density_experiment/${DATASET_NAME}/images"
            
            echo "Processing: ${DATASET_NAME} @ resolution ${RESOLUTION}"
            
            # Construct features CSV
            python3 data_analysis/construct_features.py \
                --density_json "${density_json}" \
                --meta_data "${meta_data}" \
                --image_dir "${image_dir}" \
                --edit_distance_json "${edit_distance_json}" \
                --font_size ${font_size} \
                --line_spacing ${line_spacing} \
                --letter_spacing ${letter_spacing} \
                --resolution ${RESOLUTION}
            
            echo "Completed: ${DATASET_NAME}"
        done
    done
done

echo "All feature constructions completed for resolution ${RESOLUTION}."
