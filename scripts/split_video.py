from dataclasses import dataclass
import os

import tyro
import numpy as np
from moviepy.editor import VideoFileClip

@dataclass
class SplitVideo:
    video_path: str 
    output_path: str
    n_col: int
    n_row: int

    def main(self):
        # Load the video file
        video = VideoFileClip(self.video_path)

        # Video dimensions and properties
        video_width, video_height = video.size
        fps = video.fps

        # Define grid size
        grid_size = (self.n_col, self.n_row)  # columns x rows
        crop_width = video_width // grid_size[0]
        crop_height = video_height // grid_size[1]

        # Prepare directory for cropped videos
        os.makedirs(self.output_path, exist_ok=True)

        # Function to crop and save each part of the video
        def crop_and_save_video(part, idx):
            x1 = (idx % grid_size[0]) * crop_width
            y1 = (idx // grid_size[0]) * crop_height
            cropped_video = video.crop(x1=x1, y1=y1, width=crop_width, height=crop_height)
            output_path = os.path.join(self.output_path, f"part_{part}.mp4")
            cropped_video.write_videofile(output_path, fps=fps, codec="libx264")

        # Crop and save videos
        for i in range(grid_size[0] * grid_size[1]):
            crop_and_save_video(i + 1, i)
            

if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(SplitVideo).main()