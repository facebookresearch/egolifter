#!/bin/bash
#SBATCH --job-name=process_aea
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition lowpri
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.out

echo `date`: "Job $SLURM_JOB_ID is allocated resource"

set -x
set -e

source ${HOME}/miniconda3/bin/activate gs

module unload cuda
module load cuda/12.1

EGOEXO_RAW_ROOT=$1
EGOEXO_PROCESSED_ROOT=$2
SCENE_NAME=$3

echo "Processing scene: ${SCENE_NAME}"

mkdir -p ${EGOEXO_PROCESSED_ROOT}/${SCENE_NAME}/

cp assets/vignette_imx577.png ${EGOEXO_PROCESSED_ROOT}/vignette_imx577.png
cp assets/vignette_ov7251.png ${EGOEXO_PROCESSED_ROOT}/vignette_ov7251.png

python ./scripts/process_project_aria_3dgs.py \
    --vrs_file ${EGOEXO_RAW_ROOT}/${SCENE_NAME}/videos/aria01.vrs \
    --mps_data_dir ${EGOEXO_RAW_ROOT}/${SCENE_NAME}/trajectory \
    --output_dir ${EGOEXO_PROCESSED_ROOT}/${SCENE_NAME}/

echo "Finished extracted all frames from vrs"

ln -sf ${EGOEXO_RAW_ROOT}/${SCENE_NAME}/trajectory/semidense_points.csv.gz ${EGOEXO_PROCESSED_ROOT}/${SCENE_NAME}/semidense_points.csv.gz

python ./scripts/rectify_aria.py \
    -i ${EGOEXO_PROCESSED_ROOT} \
    -o ${EGOEXO_PROCESSED_ROOT} \
    -s ${SCENE_NAME}

echo "Finished rectification for aria frames"

python scripts/generate_gsa_results.py \
    -i ${EGOEXO_PROCESSED_ROOT}/${SCENE_NAME} \
    --class_set none \
    --sam_variant sam \
    --max_longer_side 512 \
    --no_clip \
    --stride 5

echo "Done processing scene: ${SCENE_NAME}"

# Write the preprocessing status. Will rely on this to check the processor status
echo "Preprocessing succeed!" > ${EGOEXO_PROCESSED_ROOT}/${SCENE_NAME}/status.txt