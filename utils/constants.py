# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

UP_VEC = np.array([0, 0, 1])
FOV_IN_DEG = 90 # Open3D at most supports 90 degrees
FOV_IN_RAD = FOV_IN_DEG / 180.0 * np.pi

MIMSAVE_ARGS = {
    "fps": 60,
    "macro_block_size": 8
}

BAD_ARIA_PILOT_SCENES = [
    "loc1_script5_seq5_rec1",
    "loc1_script1_seq5_rec1",
    "loc2_script5_seq5_rec1",
    "loc4_script1_seq5_rec1",
]