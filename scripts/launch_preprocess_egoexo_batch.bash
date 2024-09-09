#!/bin/bash
set -u
set -e

# figured out via python sorted(os.listdir($EGOEXO_RAW_PATH))
declare -a SCENE_NAMES=(
	"cmu_bike01"
	"cmu_bike16"
	"cmu_bike18"
	"cmu_bike20"
	"cmu_soccer09"
	"georgiatech_bike_08"
	"georgiatech_cooking_10_01"
	"georgiatech_covid_02"
	"iiith_cooking_04"
	"iiith_cooking_10"
	"iiith_cooking_109"
	"iiith_cooking_112"
	"iiith_cooking_126"
	"iiith_cooking_135"
	"iiith_cooking_14"
	"iiith_cooking_142"
	"iiith_cooking_16"
	"iiith_cooking_19"
	"iiith_cooking_23"
	"iiith_cooking_40"
	"iiith_cooking_43"
	"iiith_cooking_46"
	"iiith_cooking_58"
	"iiith_cooking_63"
	"iiith_cooking_68"
	"iiith_cooking_77"
	"iiith_cooking_78"
	"iiith_cooking_82"
	"iiith_cooking_90"
	"iiith_cooking_92"
	"iiith_cooking_93"
	"iiith_cooking_97"
	"iiith_guitar_002"
	"iiith_soccer_008"
	"iiith_soccer_011"
	"iiith_soccer_031"
	"iiith_soccer_042"
	"iiith_soccer_051"
	"iiith_soccer_053"
	"indiana_bike_11"
	"indiana_cooking_22"
	"indiana_music_06"
	"nus_covidtest_13"
	"nus_covidtest_21"
	"nus_covidtest_31"
	"nus_covidtest_32"
	"nus_covidtest_33"
	"nus_covidtest_35"
	"nus_covidtest_37"
	"nus_covidtest_46"
	"nus_covidtest_48"
	"nus_cpr_29"
	"nus_cpr_47"
	"sfu_basketball_01"
	"sfu_basketball_03"
	"sfu_cooking025"
	"sfu_cooking026"
	"sfu_cooking_004"
	"sfu_covid_008"
	"sfu_covid_009"
	"sfu_covid_013"
	"unc_basketball_03-16-23_01"
	"unc_basketball_03-17-23_01"
	"uniandes_basketball_002"
	"uniandes_bouldering_004"
	"uniandes_bouldering_013"
	"uniandes_bouldering_025"
	"uniandes_dance_008"
	"upenn_0317_Violin_3"
	"upenn_0407_Violin_1"
	"upenn_0707_Dance_3"
	"upenn_0710_Cooking_1"
	"upenn_0710_Cooking_4"
	"upenn_0710_Violin_1"
	"upenn_0711_Cooking_2"
	"upenn_0711_Cooking_3"
	"upenn_0712_Cooking_2"
	"upenn_0712_Cooking_4"
	"upenn_0713_Cooking_1"
	"upenn_0714_Cooking_1"
	"upenn_0714_Cooking_7"
	"upenn_0717_Piano_1"
	"upenn_0718_Violin_1"
	"upenn_0720_Dance_1"
	"upenn_0727_Partner_Dance_3_1"
	"upenn_0727_Partner_Dance_3_2"
	"upenn_0727_Partner_Dance_4_1"
	"upenn_0727_Partner_Dance_4_2"
	"utokyo_cpr_2005_26"
	"utokyo_cpr_2005_27"
	"utokyo_pcr_2001_32"
	"utokyo_pcr_2001_35"
	"utokyo_salad_4_1018"
)

# The following two lines should be set up in setup_aws.bash

mkdir -p ${EGOEXO_PROCESSED_ROOT}

cp assets/vignette_imx577.png ${EGOEXO_PROCESSED_ROOT}/vignette_imx577.png
cp assets/vignette_ov7251.png ${EGOEXO_PROCESSED_ROOT}/vignette_ov7251.png

# Print the folder names
echo "List of sequence:"
total_num=${#SCENE_NAMES[@]}
echo "There are a total of individual scene names: $total_num"

batch_size=32
last_dependency=""

submit_jobs() {
    local dependency=$1
    shift
    local folders=("$@")
    local jobids=()

    for folder in "${folders[@]}"; do
        if [ -n "$dependency" ]; then
            jobid=$(sbatch --dependency=afterok:$dependency --job-name="prep-$folder" --parsable ./scripts/run_preprocess_egoexo.bash "$EGOEXO_RAW_ROOT" "$EGOEXO_PROCESSED_ROOT" "$folder")
        else
            jobid=$(sbatch --job-name="prep-$folder" --parsable ./scripts/run_preprocess_egoexo.bash "$EGOEXO_RAW_ROOT" "$EGOEXO_PROCESSED_ROOT" "$folder")
        fi
        jobids+=($jobid)
    done
    # Return the last job ID for dependency chaining
    echo "${jobids[-1]}"
}

jobids=()
for (( i=0; i<total_num; i+=batch_size )); do
    batch=("${SCENE_NAMES[@]:i:batch_size}")

    last_dependency=$(submit_jobs "$last_dependency" "${batch[@]}") 
done