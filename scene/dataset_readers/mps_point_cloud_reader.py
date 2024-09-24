# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
from typing import List

# Assuming GlobalPointCloud is a list-like container for GlobalPointPosition objects
class GlobalPointCloud(List):
    pass

# Assuming GlobalPointPosition is a class with the specified attributes
class GlobalPointPosition:
    def __init__(self, uid, graph_uid, px_world, py_world, pz_world, inverse_distance_std, distance_std):
        self.uid = int(uid)
        self.graph_uid = graph_uid
        self.px_world = float(px_world)
        self.py_world = float(py_world)
        self.pz_world = float(pz_world)
        self.inverse_distance_std = float(inverse_distance_std)
        self.distance_std = float(distance_std)
        
    @property
    def position_world(self):
        return [self.px_world, self.py_world, self.pz_world]

# Assuming StreamCompressionMode is an Enum with NONE and GZIP as possible values
class StreamCompressionMode:
    NONE = 'none'
    GZIP = 'gzip'

# You may need to implement or adapt CompressedIStream for Python, handling different compression modes
def CompressedIStream(path, compression_mode):
    if compression_mode == StreamCompressionMode.GZIP:
        import gzip
        return gzip.open(path, mode='rt')
    else:
        return open(path, mode='r', newline='')

def read_global_point_cloud(path):
    if path.endswith('.csv'):
        return read_global_point_cloud_with_compression(path, StreamCompressionMode.NONE)
    elif path.endswith('.gz'):
        return read_global_point_cloud_with_compression(path, StreamCompressionMode.GZIP)
    else:
        raise ValueError(f"Unsupported file format: {path}")

def read_global_point_cloud_with_compression(path, compression):
    cloud = GlobalPointCloud()
    try:
        with CompressedIStream(path, compression) as file:
            reader = csv.reader(file)
            headers = next(reader)  # Assuming the first row contains headers
            for row in reader:
                point = GlobalPointPosition(*row)
                cloud.append(point)
        print(f"Loaded #3dPoints: {len(cloud)}")
    except Exception as e:
        print(f"Failed to parse global point cloud file: {e}")
    return cloud