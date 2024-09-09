#!/bin/bash
set -u
set -e

mkdir -p ${GS_OUTPUT_ROOT}

declare -a SCENE_NAMES=(
    "loc1"
    "loc2"
    "loc3"
    "loc4"
    "loc5"
)

for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    sbatch ./scripts/run_aws.bash python train_lightning.py \
        scene.data_root=${AEA_PROCESSED_ROOT} \
        scene.scene_name=${SCENE_NAME} \
        model=unc_2d_unet \
        model.unet_acti=sigmoid \
        model.dim_extra=16 \
        lift.use_contr=True \
        opt=vanilla100k \
        exp_name=sigmoid_contr16_stride5_opt100k \
        scene.num_workers=8 \
        scene.stride=5 \
        output_root=${GS_OUTPUT_ROOT}/aea_v2/ \
        wandb.entity=surreal_gs \
        wandb.project=aea_v2
done

