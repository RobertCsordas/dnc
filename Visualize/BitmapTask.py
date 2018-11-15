# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import numpy as np
try:
    import cv2
except:
    cv2=None

from Utils.Helpers import as_numpy

def visualize_bitmap_task(i_data, o_data, zoom=8):
    if not isinstance(o_data, list):
        o_data = [o_data]
        
    imgs = []
    for d in [i_data]+o_data:
        if d is None:
            continue
        
        d=as_numpy(d)
        if d.ndim>2:
            d=d[0]            
        
        imgs.append(np.expand_dims(d.T*255, -1).astype(np.uint8))
        
    img = np.concatenate(imgs, 0)
    return nearest_zoom(img, zoom)

def visualize_01(t, zoom=8):
    return nearest_zoom(np.expand_dims(t*255,-1).astype(np.uint8), zoom)

def nearest_zoom(img, zoom=1):
    if zoom>1 and cv2 is not None:
        return cv2.resize(img, (img.shape[1] * zoom, img.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
    else:
        return img

def concatenate_tensors(tensors):
    max_size = None
    dtype = None

    for t in tensors:
        s = t.shape
        if max_size is None:
            max_size = list(s)
            dtype = t.dtype
            continue

        assert t.ndim ==len(max_size), "Can only concatenate tensors with same ndim."
        assert t.dtype == dtype, "Tensors must have the same type"
        max_size = [max(max_size[i], s[i]) for i in range(len(max_size))]

    res = np.zeros([len(tensors)] + max_size, dtype=dtype)
    for i, t in enumerate(tensors):
        res[i][tuple([slice(0,t.shape[i]) for i in range(t.ndim)])] = t
    return res



