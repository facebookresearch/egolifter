# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import cv2
import os
import argparse

def extract_frames(video_path):
    # Extract the directory and video name from the path
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Create a directory with the same name as the video file in the same directory
    output_dir = os.path.join(video_dir, video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as JPEG file
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f'Extracted {frame_count} frames from {video_name} into {output_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    args = parser.parse_args()
    
    extract_frames(args.video_path)