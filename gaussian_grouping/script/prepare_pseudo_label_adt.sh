#!/bin/bash

set -e


# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_root> <scene_name>"
    exit 1
fi


data_root="$1"
scene_name="$2"

dataset_folder="${data_root}/${scene_name}"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# 1. DEVA anything mask
cd Tracking-Anything-with-DEVA/

img_path="${data_root}/${scene_name}/images"

# # colored mask for visualization check
# python demo/demo_automatic_adt.py \
#   --chunk_size 4 \
#   --img_path "$img_path" \
#   --amp \
#   --temporal_setting semionline \
#   --size 480 \
#   --output "./example/output_gaussian_dataset_adt/${scene_name}" \
#   --suppress_small_objects  \
#   --max_num_objects 255 \
#   --SAM_PRED_IOU_THRESHOLD 0.7
#
# mv ./example/output_gaussian_dataset_adt/${scene_name}/Annotations ./example/output_gaussian_dataset_adt/${scene_name}/Annotations_color

# gray mask for training
python demo/demo_automatic_adt.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset_adt/${scene_name}" \
  --use_short_id  \
  --suppress_small_objects  \
  --max_num_objects 255 \
  --SAM_PRED_IOU_THRESHOLD 0.7 \
  --SAM_NUM_POINTS_PER_BATCH 64 \
  --SAM_NUM_POINTS_PER_SIDE 48
  
# 2. copy gray mask to the correponding data path
# If the target folder already exists, remove it
if [ -d "${data_root}/${scene_name}/deva_object_mask" ]; then
    rm -r "${data_root}/${scene_name}/deva_object_mask"
fi

cp -r ./example/output_gaussian_dataset_adt/${scene_name}/Annotations ${data_root}/${scene_name}/deva_object_mask

cd ..
