#!/bin/bash
#SBATCH --job-name=gauss_splat
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition lowpri
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.out

echo `date`: "Job $SLURM_JOB_ID is allocated resource"

set -x
set -e

source ${HOME}/miniconda3/bin/activate gs

module load cuda/12.1

DATA_ROOT=$1
SCENE_NAME=$2
OUTPUT_ROOT=$3

echo "Execute scene reconstruction on sequence data root $1, scene $2, output root $3"

python train_lightning.py \
    scene.data_root=${DATA_ROOT} \
    scene.scene_name=${SCENE_NAME} \
    exp_name=${SCENE_NAME} \
    output_root=${OUTPUT_ROOT} \
    wandb.entity=surreal_gs \
    wandb.project=3dgs_pilot_v1

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"
