'''
This script is used to stack images into videos 
for the rendering results using render_path.py
'''

from glob import glob
import os
from dataclasses import dataclass
from pathlib import Path
from natsort import natsorted

import tyro

from moviepy.editor import ImageSequenceClip, TextClip, CompositeVideoClip, ColorClip, clips_array


@dataclass
class StackVideo:
    our_folder: Path
    baseline_folder: Path
    
    image_ext: str = "jpg"
    fps: int = 20
    
    def main(self):
        ours_render_clip = self.folder_to_clip(self.our_folder / "render")
        ours_render_clip = self.add_text_to_clip(ours_render_clip, "Render by EgoLifter")
        ours_feat_clip = self.folder_to_clip(self.our_folder / "feature")
        ours_feat_clip = self.add_text_to_clip(ours_feat_clip, "Feature PCA by EgoLifter")
        
        baseline_render_clip = self.folder_to_clip(self.baseline_folder / "render")
        baseline_render_clip = self.add_text_to_clip(baseline_render_clip, "Render by EgoLifter-Static")
        baseline_feat_clip = self.folder_to_clip(self.baseline_folder / "feature")
        baseline_feat_clip = self.add_text_to_clip(baseline_feat_clip, "Feature PCA by EgoLifter-Static")
        
        combined_clip = clips_array([
            [ours_render_clip, baseline_render_clip],
            [ours_feat_clip, baseline_feat_clip]
        ])
        
        save_path = self.our_folder / "compare_path.mp4"
        
        combined_clip.write_videofile(str(save_path), fps=self.fps)

    def folder_to_clip(self, folder:Path):
        image_paths = natsorted(glob(str(folder / f"*.{self.image_ext}")))
        clip = ImageSequenceClip(image_paths, fps=self.fps)
        return clip

    def add_text_to_clip(self, clip:ImageSequenceClip, text:str):
        text_clip = TextClip(text, fontsize=36, font='Arial', color='white')
        text_clip = text_clip.set_position(('left','top')).set_duration(clip.duration)
        clip = CompositeVideoClip([clip, text_clip])
        return clip


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(StackVideo).main()