import os, glob
from dataclasses import dataclass
from pathlib import Path

from natsort import natsorted

import tyro
import numpy as np
import imageio

@dataclass
class ExtractQualImages:
    folder_root: Path = Path("./scripts/qual_images")
    subfolder: str = "adt"

    def main(self) -> None:
        image_temp_path = self.folder_root / self.subfolder / "*_baseline.png"
        baseline_paths = glob.glob(str(image_temp_path))
        
        baseline_paths = natsorted(baseline_paths)
        
        save_path = self.folder_root / self.subfolder / "crops"
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, baseline_path in enumerate(baseline_paths):
            input_name = os.path.basename(baseline_path).split(".")[0]
            ours_path = glob.glob(str(self.folder_root / self.subfolder / f"{input_name}*_ours.png"))[0]
            baseline_all = imageio.imread(baseline_path)
            ours_all = imageio.imread(ours_path)
            
            try:
                def_path = glob.glob(str(self.folder_root / self.subfolder / f"{input_name}*_def.png"))[0]
                def_all = imageio.imread(def_path)
            except IndexError:
                pass
            
            # Split the baseline_all into 2x3 image grid
            w = baseline_all.shape[1] // 3
            h = baseline_all.shape[0] // 2

            crops = {
                "gt": baseline_all[:h, :w],
                "baseline_rgb": baseline_all[:h, w:2*w],
                "baseline_feat": baseline_all[h:, w:2*w],
                # "ours_transient": ours_all[h:2*h, :w],
                "ours_transient": ours_all[int(1.104*h):int(1.9*h), int(0):int(0.796*w)],
                "ours_rgb": ours_all[:h, w:2*w],
                "ours_feat": ours_all[h:, w:2*w],
            }
            
            for key, crop in crops.items():
                crop = np.rot90(crop, 3)
                imageio.imwrite(str(save_path / f"{i}_{key}.jpg"), crop)
            
    
    
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ExtractQualImages).main()