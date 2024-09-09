import os, glob
from pathlib import Path
import argparse

from natsort import natsorted

import moviepy
import moviepy.editor


def main(args):
    input_folders = glob.glob(str(args.input_root / "*_c*"))
    input_folders = natsorted(input_folders)
    if not args.output_folder.exists():
        args.output_folder.mkdir(parents=True)

    max_cols = 5

    for i in range(0, len(input_folders), max_cols):
        video_clips = [[], []]
        for j in range(max_cols):
            if i + j >= len(input_folders):
                break

            input_folder = input_folders[i + j]
            input_folder = Path(input_folder)
            print(f"Processing {input_folder}...")

            video_files = glob.glob(str(input_folder / "render_rotate_iter*.mp4"))
            video_files = natsorted(video_files)

            video_clips[0].append(moviepy.editor.VideoFileClip(video_files[0]).resize(width=512))
            video_clips[1].append(moviepy.editor.VideoFileClip(video_files[-1]).resize(width=512))

        stacked_clip = moviepy.editor.clips_array(video_clips)
        stacked_clip.write_videofile(str(args.output_folder / f"stacked_{i}.mp4"), fps=30)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_root", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, default=Path("./concated_videos"))

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    main(args)