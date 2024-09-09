#!/bin/bash
#SBATCH --account=rrg-florian7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gg_train_adt
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64000M
#SBATCH --cpus-per-task=8
#SBATCH --exclude=cdr2614

echo `date`: "Job $SLURM_JOB_ID is allocated resource"

export ENV_ROOT=/home/guqiao/envs/gg
export SRC_ROOT=/home/guqiao/src/gaussian-grouping

echo 'Copy and extract dataset'
SCENE_NAME=$1
shift # Shift the argument to treat the second to the first

time cp ~/datasets/adt_processed/${SCENE_NAME}.zip $SLURM_TMPDIR/
time unzip $SLURM_TMPDIR/${SCENE_NAME}.zip -d $SLURM_TMPDIR/

echo 'set up the environment'
module load StdEnv/2023 gcc/12.3 python/3.10 scipy-stack opencv cuda/12.2
source $ENV_ROOT/bin/activate

cd $SRC_ROOT

$* -s ${SLURM_TMPDIR}/${SCENE_NAME}

