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
module load cuda/12.1

AEA_RAW_ROOT=$1
AEA_PROCESSED_ROOT=$2
SCENE_NAME=$3

cp assets/vignette_imx577.png ${AEA_PROCESSED_ROOT}/vignette_imx577.png
cp assets/vignette_ov7251.png ${AEA_PROCESSED_ROOT}/vignette_ov7251.png

echo "Processing scene: ${SCENE_NAME}"

mkdir -p ${AEA_PROCESSED_ROOT}/${SCENE_NAME}/

python ./scripts/process_project_aria_3dgs.py \
    --vrs_file ${AEA_RAW_ROOT}/${SCENE_NAME}_main_data/recording.vrs \
    --mps_data_dir ${AEA_RAW_ROOT}/${SCENE_NAME}_mps_slam_trajectories/ \
    --output_dir ${AEA_PROCESSED_ROOT}/${SCENE_NAME}/

cp ${AEA_RAW_ROOT}/${SCENE_NAME}_mps_slam_points/semidense_points.csv.gz ${AEA_PROCESSED_ROOT}/${SCENE_NAME}/

python ./scripts/rectify_aria.py \
    -i ${AEA_PROCESSED_ROOT} \
    -o ${AEA_PROCESSED_ROOT} \
    -s ${SCENE_NAME}

python scripts/generate_gsa_results.py \
    -i ${AEA_PROCESSED_ROOT}/${SCENE_NAME} \
    --class_set none \
    --sam_variant sam \
    --max_longer_side 512 \
    --no_clip \
    --stride 5

echo "Done processing scene: ${SCENE_NAME}"