# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates
       

egopose_camera_params = {
    'id': None,
    'res_w': None, # Pulled from metadata
    'res_h': None, # Pulled from metadata
    
    # Dummy camera parameters (taken from Human3.6M), need to change
    'azimuth': 70, # Only used for visualization
    'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
}

# 这个skelton已经和hm3.6 17个点的对齐了
egopose_skeleton = Skeleton(parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8, 9, 8, 11, 12, 8, 14, 15],
       joints_left=[4, 5, 6, 11, 12, 13],
       joints_right=[1, 2, 3, 14, 15, 16])


class EgoposeDataset(MocapDataset):
    def __init__(self, detections_path, remove_static_joints=True):
        super().__init__(fps=20, skeleton=egopose_skeleton)        
        
        # Load serialized dataset
        data = np.load(detections_path, allow_pickle=True)["egopose"].item()
        '''
        egopose
            male_008_a_a
                 env01
                      positions
                      positions_3d
                 env02
        '''
        cam = {}
        cam.update(egopose_camera_params)
        cam['orientation'] = np.array(cam['orientation'], dtype='float32')
        cam['translation'] = np.array(cam['translation'], dtype='float32')
        cam['translation'] = cam['translation']/1000 # mm to meters
        cam['res_w'] = 256
        cam['res_h'] = 256
        self._cameras = cam
        
        self._data = data

        self._skeleton = egopose_skeleton
            
    def supports_semi_supervised(self):
        return False
   