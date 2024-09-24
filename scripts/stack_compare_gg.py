# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
This script is used to stack images into videos and compose them into one video for comparison
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
    gs_folder: Path
    gg_folder: Path
    
    image_ext: str = "jpg"
    fps: int = 20
    
    def main(self):
        gt_clip = self.folder_to_clip(self.gs_folder / "gt")
        gt_clip = self.add_text_to_clip(gt_clip, "GT")
        gs_render_clip = self.folder_to_clip(self.gs_folder / "render")
        gs_render_clip = self.add_text_to_clip(gs_render_clip, "Render by EgoLifter")
        gs_feat_clip = self.folder_to_clip(self.gs_folder / "feature")
        gs_feat_clip = self.add_text_to_clip(gs_feat_clip, "Feature PCA by EgoLifter")
        gs_mask_clip = self.folder_to_clip(self.gs_folder / "unc_mask")
        gs_mask_clip = self.add_text_to_clip(gs_mask_clip, "Transient map by EgoLifter")
        
        gg_render_clip = self.folder_to_clip(self.gg_folder / "render")
        gg_render_clip = self.add_text_to_clip(gg_render_clip, "Render by Gaussian Grouping")
        gg_feat_clip = self.folder_to_clip(self.gg_folder / "feature")
        gg_feat_clip = self.add_text_to_clip(gg_feat_clip, "Feature PCA by Gaussian Grouping")
        gg_pred_clip = self.folder_to_clip(self.gg_folder / "pred_obj")
        gg_pred_clip = self.add_text_to_clip(gg_pred_clip, "Predicted object masks by Gaussian Grouping")

        black_clip = ColorClip(size=gt_clip.size, color=(0, 0, 0), duration=gt_clip.duration)

        # combined_clip = clips_array([
        #     [gt_clip, gs_render_clip, gs_feat_clip, gs_mask_clip],
        #     [black_clip, gg_render_clip, gg_feat_clip, gg_pred_clip]
        # ])
        
        combined_clip = clips_array([
            [gt_clip, gs_render_clip, gg_render_clip],
            [black_clip, gs_feat_clip, gg_feat_clip]
        ])
        
        save_path = self.gs_folder / "compare_gg_less.mp4"
        
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