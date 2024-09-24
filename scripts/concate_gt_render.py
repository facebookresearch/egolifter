# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from utils.render import concate_to_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts images from a directory into an MP4 video.")
    parser.add_argument("-i", "--input_folder", help="Path to the folder of the training log.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for the output video. Default is 20.")
    parser.add_argument("--camera_type", type=str, default="rgb", help="Camera type. Default is rgb.")
    parser.add_argument("--no_rot", action="store_true", help="Do not rotate the images. Otherwise rotate by -90 degrees.")

    args = parser.parse_args()
    
    concate_to_video(args.input_folder, args.fps, args.camera_type, args.no_rot)