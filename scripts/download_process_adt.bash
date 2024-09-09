#!/bin/bash

set -u
set -e

# Copy the vignette images to the dataset
cp assets/vignette_imx577.png ${DATA_ROOT} # Vignette image for the RGB camera
cp assets/vignette_ov7251.png ${DATA_ROOT} # Vignette image for the SLAM camera

# # Names of the scenes used in ADT benchmark in EgoLifter
# declare -a SCENE_NAMES=(
#     "Apartment_release_multiskeleton_party_seq121"
#     "Apartment_release_multiskeleton_party_seq122"
#     "Apartment_release_multiskeleton_party_seq123"
#     "Apartment_release_multiskeleton_party_seq125"
#     "Apartment_release_multiskeleton_party_seq126"
#     "Apartment_release_multiskeleton_party_seq127"
#     "Apartment_release_decoration_skeleton_seq137"
#     "Apartment_release_meal_skeleton_seq136"
#     "Apartment_release_multiuser_clean_seq116"
#     "Apartment_release_multiuser_cook_seq114"
#     "Apartment_release_multiuser_cook_seq143"
#     "Apartment_release_multiuser_meal_seq132"
#     "Apartment_release_multiuser_meal_seq140"
#     "Apartment_release_multiuser_party_seq140"
#     "Apartment_release_work_skeleton_seq131"
#     "Apartment_release_work_skeleton_seq140"
# )

# *TODO*: For debugging only - remove before formal release
declare -a SCENE_NAMES=(
    "Apartment_release_multiskeleton_party_seq121"
)


# Download
for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    adt_benchmark_dataset_downloader \
        -c ${DATA_ROOT}/aria_digital_twin_dataset_download_urls.json \
        -o ${DATA_ROOT}/ \
        -d 0 1 5 6 \
        -l ${SCENE_NAME}
done

# Rectify images and format the folder structure
for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/process_adt_3dgs.py \
        --data_root ${DATA_ROOT} \
        --output_root ${PROCESSED_ROOT} \
        --sequence_name ${SCENE_NAME} &
done

wait

# Generate SAM segmentation results
for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/generate_gsa_results.py \
        -i ${PROCESSED_ROOT}/${SCENE_NAME} \
        --class_set none \
        --sam_variant sam \
        --max_longer_side 512 \
        --no_clip
done

# Generate evaluation target for query-based segmentation
for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/generate_2dseg_query.py \
        --data_root ${PROCESSED_ROOT} \
        --scene_name ${SCENE_NAME} &
done

wait

for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/generate_2dseg_query_sample.py \
        --data_root ${PROCESSED_ROOT} \
        --scene_name ${SCENE_NAME}
done

for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/generate_3dbox_query.py \
        --raw_root ${DATA_ROOT} \
        --data_root ${PROCESSED_ROOT} \
        --scene_name ${SCENE_NAME}
done
