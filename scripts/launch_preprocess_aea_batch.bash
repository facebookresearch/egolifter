#!/bin/bash
set -u
set -e

declare -a SCENE_NAMES=(
    "loc1_script1_seq1_rec1"
    "loc1_script1_seq3_rec1"
    "loc1_script1_seq5_rec1"
    "loc1_script1_seq6_rec1"
    "loc1_script1_seq7_rec1"
    "loc1_script2_seq1_rec1"
    "loc1_script2_seq1_rec2"
    "loc1_script2_seq3_rec1"
    "loc1_script2_seq3_rec2"
    "loc1_script2_seq4_rec1"
    "loc1_script2_seq4_rec2"
    "loc1_script2_seq6_rec1"
    "loc1_script2_seq6_rec2"
    "loc1_script2_seq7_rec1"
    "loc1_script2_seq7_rec2"
    "loc1_script2_seq8_rec1"
    "loc1_script2_seq8_rec2"
    "loc1_script3_seq1_rec1"
    "loc1_script3_seq2_rec1"
    "loc1_script3_seq5_rec1"
    "loc1_script4_seq2_rec1"
    "loc1_script4_seq3_rec1"
    "loc1_script4_seq4_rec1"
    "loc1_script4_seq5_rec1"
    "loc1_script5_seq1_rec1"
    "loc1_script5_seq2_rec1"
    "loc1_script5_seq3_rec1"
    "loc1_script5_seq5_rec1"
    "loc1_script5_seq6_rec1"
    "loc2_script1_seq1_rec1"
    "loc2_script1_seq2_rec1"
    "loc2_script1_seq3_rec1"
    "loc2_script1_seq4_rec1"
    "loc2_script1_seq5_rec1"
    "loc2_script1_seq6_rec1"
    "loc2_script1_seq7_rec1"
    "loc2_script2_seq1_rec1"
    "loc2_script2_seq1_rec2"
    "loc2_script2_seq2_rec1"
    "loc2_script2_seq2_rec2"
    "loc2_script2_seq3_rec1"
    "loc2_script2_seq3_rec2"
    "loc2_script2_seq4_rec1"
    "loc2_script2_seq4_rec2"
    "loc2_script2_seq5_rec1"
    "loc2_script2_seq5_rec2"
    "loc2_script2_seq6_rec1"
    "loc2_script2_seq6_rec2"
    "loc2_script2_seq8_rec1"
    "loc2_script2_seq8_rec2"
    "loc2_script3_seq1_rec1"
    "loc2_script3_seq1_rec2"
    "loc2_script3_seq2_rec1"
    "loc2_script3_seq2_rec2"
    "loc2_script3_seq3_rec1"
    "loc2_script3_seq3_rec2"
    "loc2_script3_seq4_rec1"
    "loc2_script3_seq4_rec2"
    "loc2_script3_seq5_rec1"
    "loc2_script3_seq5_rec2"
    "loc2_script4_seq2_rec1"
    "loc2_script4_seq3_rec1"
    "loc2_script4_seq4_rec1"
    "loc2_script4_seq5_rec1"
    "loc2_script4_seq7_rec1"
    "loc2_script5_seq1_rec1"
    "loc2_script5_seq2_rec1"
    "loc2_script5_seq3_rec1"
    "loc2_script5_seq4_rec1"
    "loc2_script5_seq5_rec1"
    "loc2_script5_seq6_rec1"
    "loc2_script5_seq7_rec1"
    "loc3_script1_seq1_rec1"
    "loc3_script1_seq2_rec1"
    "loc3_script1_seq3_rec1"
    "loc3_script1_seq4_rec1"
    "loc3_script1_seq5_rec1"
    "loc3_script1_seq6_rec1"
    "loc3_script1_seq7_rec1"
    "loc3_script2_seq1_rec1"
    "loc3_script2_seq1_rec2"
    "loc3_script2_seq2_rec1"
    "loc3_script2_seq3_rec1"
    "loc3_script2_seq3_rec2"
    "loc3_script2_seq4_rec1"
    "loc3_script2_seq4_rec2"
    "loc3_script2_seq5_rec1"
    "loc3_script2_seq5_rec2"
    "loc3_script2_seq7_rec1"
    "loc3_script2_seq7_rec2"
    "loc3_script3_seq1_rec1"
    "loc3_script3_seq1_rec2"
    "loc3_script3_seq2_rec1"
    "loc3_script3_seq2_rec2"
    "loc3_script3_seq4_rec1"
    "loc3_script3_seq4_rec2"
    "loc3_script3_seq5_rec1"
    "loc3_script3_seq5_rec2"
    "loc3_script4_seq2_rec1"
    "loc3_script4_seq3_rec1"
    "loc3_script4_seq4_rec1"
    "loc3_script4_seq5_rec1"
    "loc3_script4_seq7_rec1"
    "loc3_script5_seq1_rec1"
    "loc3_script5_seq2_rec1"
    "loc3_script5_seq3_rec1"
    "loc3_script5_seq4_rec1"
    "loc3_script5_seq5_rec1"
    "loc3_script5_seq6_rec1"
    "loc3_script5_seq7_rec1"
    "loc4_script1_seq1_rec1"
    "loc4_script1_seq3_rec1"
    "loc4_script1_seq5_rec1"
    "loc4_script1_seq6_rec1"
    "loc4_script2_seq1_rec2"
    "loc4_script2_seq2_rec1"
    "loc4_script2_seq3_rec2"
    "loc4_script2_seq4_rec1"
    "loc4_script2_seq6_rec1"
    "loc4_script2_seq7_rec1"
    "loc4_script2_seq8_rec2"
    "loc4_script3_seq1_rec2"
    "loc4_script3_seq2_rec2"
    "loc4_script3_seq3_rec1"
    "loc4_script3_seq4_rec1"
    "loc4_script4_seq2_rec1"
    "loc4_script5_seq1_rec1"
    "loc4_script5_seq3_rec1"
    "loc4_script5_seq7_rec1"
    "loc5_script4_seq1_rec1"
    "loc5_script4_seq2_rec1"
    "loc5_script4_seq3_rec1"
    "loc5_script4_seq4_rec1"
    "loc5_script4_seq5_rec1"
    "loc5_script4_seq6_rec1"
    "loc5_script4_seq7_rec1"
    "loc5_script5_seq1_rec1"
    "loc5_script5_seq2_rec1"
    "loc5_script5_seq3_rec1"
    "loc5_script5_seq4_rec1"
    "loc5_script5_seq5_rec1"
    "loc5_script5_seq6_rec1"
    "loc5_script5_seq7_rec1"
)

# The following two lines should be set up in setup_aws.bash
# export AEA_RAW_ROOT=/source_1a/data/aria_pilot_dataset/2024_release_v1_beta/
# export AEA_PROCESSED_ROOT=/source_1a/data/qgu/aria_processed/pilot_2024_v1-1/

mkdir -p ${AEA_PROCESSED_ROOT}

cp assets/vignette_imx577.png ${AEA_PROCESSED_ROOT}/vignette_imx577.png
cp assets/vignette_ov7251.png ${AEA_PROCESSED_ROOT}/vignette_ov7251.png

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
            jobid=$(sbatch --dependency=afterok:$dependency --job-name="prep-$folder" --parsable ./scripts/run_preprocess_aea.bash "$AEA_RAW_ROOT" "$AEA_PROCESSED_ROOT" "$folder")
        else
            jobid=$(sbatch --job-name="prep-$folder" --parsable ./scripts/run_preprocess_aea.bash "$AEA_RAW_ROOT" "$AEA_PROCESSED_ROOT" "$folder")
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