#!/bin/bash
set -u
set -e

mkdir -p ${GS_OUTPUT_ROOT}

# declare -a SCENE_NAMES=(
#     "cmu_bike16" # finished
#     "cmu_bike18" # finished
#     "cmu_bike20" # finished
#     "cmu_soccer09" # finished
#     "georgiatech_cooking_10_01" # finished
#     "georgiatech_covid_02" # finished
#     "iiith_cooking_04" # finished
#     "iiith_cooking_10" # finished
#     "iiith_cooking_14" # finished
#     "iiith_cooking_16" # finished
#     "iiith_cooking_19" # finished
#     "iiith_cooking_23" # finished
#     "iiith_cooking_40" # finished
#     "iiith_cooking_43" # finished
#     "iiith_cooking_46" # finished
#     "iiith_cooking_58" # finished
#     "iiith_cooking_68" # finished
#     "iiith_cooking_77" # finished
#     "iiith_cooking_78" # finished
#     "iiith_cooking_82" # finished
#     "iiith_cooking_90" # finished
#     "iiith_cooking_92" # finished
#     "iiith_cooking_93" # finished
#     "iiith_cooking_97" # finished
#     "iiith_cooking_112" # finished
#     "iiith_cooking_126" # finished
#     "iiith_cooking_135" # finished
#     "iiith_cooking_142" # finished
#     "iiith_guitar_002" # finished
#     "iiith_soccer_008" # finished
#     "iiith_soccer_011" # finished
#     "iiith_soccer_031" # finished
#     "iiith_soccer_042" # finished
#     "iiith_soccer_051" # finished
#     "iiith_soccer_053" # finished
#     "indiana_bike_11" # finished
#     "indiana_music_06" # finished
#     "nus_covidtest_13" # finished
#     "nus_covidtest_21" # finished
#     "nus_covidtest_31" # finished
#     "nus_covidtest_32" # finished
#     "nus_covidtest_33" # finished
#     "nus_covidtest_35" # finished
#     "nus_covidtest_37" # finished
#     "nus_covidtest_46" # finished
#     "nus_covidtest_48" # finished
#     "nus_cpr_29" # finished
#     "sfu_basketball_01" # finished
#     "sfu_basketball_03" # finished
#     "sfu_cooking_004" # finished
#     "sfu_cooking025" # finished
#     "sfu_cooking026" # finished
#     "sfu_covid_008" # finished
#     "sfu_covid_009" # finished
#     "sfu_covid_013" # finished
#     "unc_basketball_03-16-23_01" # finished
#     "unc_basketball_03-17-23_01" # finished
#     "uniandes_basketball_002" # finished
#     "uniandes_bouldering_004"
#     "uniandes_bouldering_013"
#     "uniandes_bouldering_025"
#     "uniandes_dance_008"
#     "upenn_0317_Violin_3"
#     "upenn_0407_Violin_1"
#     "upenn_0707_Dance_3"
#     "upenn_0710_Cooking_1"
#     "upenn_0710_Cooking_4"
#     "upenn_0710_Violin_1"
#     "upenn_0711_Cooking_2"
#     "upenn_0711_Cooking_3"
#     "upenn_0712_Cooking_2"
#     "upenn_0712_Cooking_4"
#     "upenn_0713_Cooking_1"
#     "upenn_0714_Cooking_1"
#     "upenn_0714_Cooking_7"
#     "upenn_0717_Piano_1"
#     "upenn_0718_Violin_1"
#     "upenn_0720_Dance_1"
#     "upenn_0727_Partner_Dance_3_1"
#     "upenn_0727_Partner_Dance_3_2"
#     "upenn_0727_Partner_Dance_4_1"
#     "upenn_0727_Partner_Dance_4_2"
#     "utokyo_cpr_2005_26"
#     "utokyo_pcr_2001_32"    
#     "utokyo_pcr_2001_35"
#     "utokyo_salad_4_1018"
# )

declare -a SCENE_NAMES=(
    "cmu_bike16"
    "georgiatech_cooking_10_01"
    "iiith_cooking_92"
    "sfu_basketball_01"
    "sfu_cooking025"
)


for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    sbatch --job-name="gs-egoexo-$SCENE_NAME" ./scripts/run_aws.bash python train_lightning.py \
        scene.data_root=${EGOEXO_PROCESSED_ROOT} \
        scene.scene_name=${SCENE_NAME} \
        model=unc_2d_unet \
        model.unet_acti=baseline \
        model.dim_extra=16 \
        lift.use_contr=True \
        opt=vanilla100k \
        exp_name=${SCENE_NAME}_baseline_contr16_stride5_opt100k \
        scene.num_workers=8 \
        scene.stride=5 \
        output_root=${GS_OUTPUT_ROOT}/egoexo/ \
        wandb.entity=surreal_gs \
        wandb.project=egoexo
done

