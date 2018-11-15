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
from .BitmapTask import BitmapTask
from Utils.Seed import get_randstate

class AssociativeRecall(BitmapTask):
    def __init__(self, length=None, bit_w=8, block_w=3, transform=lambda x: x):
        super(AssociativeRecall, self).__init__()
        self.length = length
        self.bit_w = bit_w
        self.block_w = block_w
        self.transform = transform
        self.seed = None

    def __getitem__(self, key):
        if self.seed is None:
            self.seed = get_randstate()

        length = self.length() if callable(self.length) else self.length
        if length is None:
            # Random length batch hack.
            length = key

        stride = self.block_w+1

        d = self.seed.randint(0, 2, [length * (self.block_w+1), self.bit_w + 2]).astype(np.float32)
        d[:,-2:] = 0

        # Terminate input block
        for i in range(1,length,1):
            d[i * stride - 1, :] = 0
            d[i * stride - 1, -2] = 1

        # Terminate input sequence
        d[-1, :] = 0
        d[-1, -1] = 1

        # Add and terminate query
        ti = self.seed.randint(0, length-1)
        d = np.concatenate((d, d[ti * stride: (ti+1) * stride-1], np.zeros([self.block_w+1, self.bit_w+2], np.float32)), axis=0)
        d[-(1+self.block_w),-1] = 1

        # Target
        target = np.zeros_like(d)
        target[-self.block_w:] = d[(ti+1) * stride: (ti+2) * stride-1]

        return self.transform({
            "input": d,
            "output": target
        })

