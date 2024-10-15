#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -u
set -e

# Copy the vignette images to the dataset
cp assets/vignette_imx577.png ${ADT_DATA_ROOT} # Vignette image for the RGB camera
cp assets/vignette_ov7251.png ${ADT_DATA_ROOT} # Vignette image for the SLAM camera

# The data format for ADT dataset has recently changed. See this link:
# https://github.com/facebookresearch/projectaria_tools/blob/main/projects/AriaDigitalTwinDatasetTools/data_provider/AriaDigitalTwinDataPathsProvider.h#L38

# scene names using the old data format
declare -a SCENE_NAMES=(
    "Apartment_release_multiskeleton_party_seq121"
    "Apartment_release_multiskeleton_party_seq122"
    "Apartment_release_multiskeleton_party_seq123"
    "Apartment_release_multiskeleton_party_seq125"
    "Apartment_release_multiskeleton_party_seq126"
    "Apartment_release_multiskeleton_party_seq127"
    "Apartment_release_decoration_skeleton_seq137"
    "Apartment_release_meal_skeleton_seq136"
    "Apartment_release_multiuser_clean_seq116"
    "Apartment_release_multiuser_cook_seq114"
    "Apartment_release_multiuser_cook_seq143"
    "Apartment_release_multiuser_meal_seq132"
    "Apartment_release_multiuser_meal_seq140"
    "Apartment_release_multiuser_party_seq140"
    "Apartment_release_work_skeleton_seq131"
    "Apartment_release_work_skeleton_seq140"
)

# scene names using the new format, only used for downloading script
declare -a SCENE_NAMES_NEW=(
    "Apartment_release_multiskeleton_party_seq121_71292"
    "Apartment_release_multiskeleton_party_seq121_M1292"
    "Apartment_release_multiskeleton_party_seq122_71292"
    "Apartment_release_multiskeleton_party_seq122_M1292"
    "Apartment_release_multiskeleton_party_seq123_71292"
    "Apartment_release_multiskeleton_party_seq123_M1292"
    "Apartment_release_multiskeleton_party_seq125_71292"
    "Apartment_release_multiskeleton_party_seq125_M1292"
    "Apartment_release_multiskeleton_party_seq126_71292"
    "Apartment_release_multiskeleton_party_seq126_M1292"
    "Apartment_release_multiskeleton_party_seq127_71292"
    "Apartment_release_multiskeleton_party_seq127_M1292"
    "Apartment_release_decoration_skeleton_seq137_M1292"
    "Apartment_release_meal_skeleton_seq136_M1292"
    "Apartment_release_multiuser_clean_seq116_M1292"
    "Apartment_release_multiuser_cook_seq114_M1292"
    "Apartment_release_multiuser_cook_seq143_M1292"
    "Apartment_release_multiuser_meal_seq132_M1292"
    "Apartment_release_multiuser_meal_seq140_M1292"
    "Apartment_release_multiuser_party_seq140_M1292"
    "Apartment_release_work_skeleton_seq131_M1292"
    "Apartment_release_work_skeleton_seq140_M1292"
)

# # Only for debugging
# declare -a SCENE_NAMES_NEW=(
#     "Apartment_release_multiskeleton_party_seq121_71292"
#     "Apartment_release_multiskeleton_party_seq121_M1292"
# )
# declare -a SCENE_NAMES=(
#     "Apartment_release_multiskeleton_party_seq121"
# )

# Download
for SCENE_NAME in "${SCENE_NAMES_NEW[@]}"; do
    aria_dataset_downloader \
        -c ${ADT_DATA_ROOT}/ADT_download_urls.json \
        -o ${ADT_DATA_ROOT}/ \
        -d 0 1 2 3 6 7 \
        -l ${SCENE_NAME}
done

# Rectify images and format the folder structure
for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/process_adt_3dgs.py \
        --data_root ${ADT_DATA_ROOT} \
        --output_root ${ADT_PROCESSED_ROOT} \
        --sequence_name ${SCENE_NAME} &
done

wait

# Generate SAM segmentation results
for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/generate_gsa_results.py \
        -i ${ADT_PROCESSED_ROOT}/${SCENE_NAME} \
        --class_set none \
        --sam_variant sam \
        --max_longer_side 512 \
        --no_clip
done

# Generate evaluation target for query-based segmentation
for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/generate_2dseg_query.py \
        --data_root ${ADT_PROCESSED_ROOT} \
        --scene_name ${SCENE_NAME} &
done

wait

for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/generate_2dseg_query_sample.py \
        --data_root ${ADT_PROCESSED_ROOT} \
        --scene_name ${SCENE_NAME}
done

for SCENE_NAME in "${SCENE_NAMES[@]}"; do
    python scripts/generate_3dbox_query.py \
        --raw_root ${ADT_DATA_ROOT} \
        --data_root ${ADT_PROCESSED_ROOT} \
        --scene_name ${SCENE_NAME}
done
