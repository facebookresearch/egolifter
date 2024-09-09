#!/bin/bash
#SBATCH --job-name=processing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition lowpri
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.out

echo `date`: "Job $SLURM_JOB_ID is allocated resource"

set -x
set -e

source ${HOME}/miniconda3/bin/activate gs
source setup_aws.bash

# The following line put all arguments after the script name into one command and run it. 
# It allows push up various experiments without changing this script. 
$*

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"