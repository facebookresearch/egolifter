# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import os
import argparse
from natsort import natsorted

def images_to_video(images_folder, output_video_path, fps=20):
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)

    # Read the first image to get the frame size
    first_image_path = os.path.join(images_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    size = (width, height)
    
    # Create a video writer object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    # Iterate through images and write them to the video
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)
    
    out.release()
    print(f'Video saved to {output_video_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images in a folder to a video.')
    parser.add_argument('images_folder', type=str, help='Path to the folder containing image files')
    parser.add_argument('output_video_path', type=str, help='Path to save the output video')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for the output video')
    args = parser.parse_args()
    
    images_to_video(args.images_folder, args.output_video_path, args.fps)