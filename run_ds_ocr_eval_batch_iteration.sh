#!/bin/bash
#SBATCH --job-name=ds_density_sweep
#SBATCH --output=SLURM_Logs/%j.out
#SBATCH --error=SLURM_Logs/%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB

source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: Allocated GPU not accessible."
    exit 1
fi

echo "GPU check passed."

BASE_SIZE=1024
IMAGE_SIZE=640
FONT_SIZE=36

LINE_SPACINGS=(0 6 24 42)
LETTER_SPACINGS=(-1 0 7)

for line_spacing in "${LINE_SPACINGS[@]}"; do
    for letter_spacing in "${LETTER_SPACINGS[@]}"; do
        DATASET_NAME="${FONT_SIZE}_${line_spacing}_${letter_spacing}"
        INPUT_PATH="build_dataset_from_ebook/pride_dataset_square_density_experiment/${DATASET_NAME}/images/"
        OUTPUT_PATH="ocr_predictions/pride_dataset_square_density_experiment/${DATASET_NAME}/Gaudum_${BASE_SIZE}/"
        
        # Determine batch size based on config combination
        BATCH_SIZE=500
        if [[ ("${FONT_SIZE}" -eq 36 && "${line_spacing}" -eq 42 && "${letter_spacing}" -eq 7 && "${BASE_SIZE}" -eq 1024) || \
              ("${FONT_SIZE}" -eq 36 && "${line_spacing}" -eq 24 && "${letter_spacing}" -eq 7 && "${BASE_SIZE}" -eq 1280) || \
              ("${FONT_SIZE}" -eq 36 && "${line_spacing}" -eq 42 && "${letter_spacing}" -eq -1 && "${BASE_SIZE}" -eq 1280) || \
              ("${FONT_SIZE}" -eq 36 && "${line_spacing}" -eq 42 && "${letter_spacing}" -eq 0 && "${BASE_SIZE}" -eq 1280) || \
              ("${FONT_SIZE}" -eq 36 && "${line_spacing}" -eq 42 && "${letter_spacing}" -eq 7 && "${BASE_SIZE}" -eq 1280) || \
              ("${FONT_SIZE}" -eq 28 && "${line_spacing}" -eq 42 && "${letter_spacing}" -eq 7 && "${BASE_SIZE}" -eq 1280) ]]; then
            BATCH_SIZE=200
        fi
        
        echo "Processing: font_size=${FONT_SIZE}, line_spacing=${line_spacing}, letter_spacing=${letter_spacing}, batch_size=${BATCH_SIZE}"
        
        python3 -m run_dpsk_ocr_eval_batch \
            --base_size ${BASE_SIZE} \
            --image_size ${IMAGE_SIZE} \
            --input_path ${INPUT_PATH} \
            --output_path ${OUTPUT_PATH} \
            --batch_size ${BATCH_SIZE}
        
        echo "Completed: ${DATASET_NAME}"
    done
done

echo "All density experiments completed."