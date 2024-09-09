#!/bin/bash
#SBATCH --job-name=gs_viewer_web
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00   # run for one day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition lowpri
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.out

set -x
set -e

cat /etc/hosts

source ${HOME}/miniconda3/bin/activate gs

module unload cuda
module load cuda/12.1

data_path=$1

echo "launch web viewer for results in $data_path"

python gaussian-splatting-lightning/viewer.py $data_path --background_color white  --reorient disable
