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

import math
import numpy as np
from .BitmapTask import BitmapTask
from Utils.Seed import get_randstate


class KeyValue(BitmapTask):
    def __init__(self, length=None, bit_w=8, transform=lambda x: x):
        assert bit_w % 2 == 0, "bit_w must be even"
        super(KeyValue, self).__init__()
        self.length = length
        self.bit_w = bit_w
        self.transform = transform
        self.seed = None
        self.key_w = self.bit_w//2
        self.max_key = 2**self.key_w - 1

    def __getitem__(self, key):
        if self.seed is None:
            self.seed = get_randstate()

        if self.length is None:
            # Random length batch hack.
            length = key
        else:
            length = self.length() if callable(self.length) else self.length

        # keys must be unique
        keys = None
        last_size = 0
        while last_size!=length:
            res = self.seed.random_integers(0, self.max_key, size=(length - last_size))
            if keys is not None:
                keys = np.concatenate((res, keys))
            else:
                keys = res

            keys = np.unique(keys)
            last_size = keys.size

        # view as bunch of uint8s, convert them to bit patterns, then cut the correct amount from it
        keys = keys.view(np.uint8).reshape(length, -1)
        keys = keys[:, :math.ceil(self.key_w/8)]
        keys = np.unpackbits(np.expand_dims(keys,-1), axis=-1)
        keys = np.flip(keys, axis=-1).reshape(keys.shape[0],-1)[:, :self.key_w]
        keys = keys.astype(np.float32)

        values = self.seed.randint(0,2, keys.shape).astype(np.float32)

        perm = self.seed.permutation(length)
        keys_perm = keys[perm,:]
        values_perm = values[perm,:]

        i_p = np.zeros((2*length+2, self.bit_w+1), dtype=np.float32)
        i_p[:length,:self.key_w] = keys
        i_p[:length,self.key_w:-1] = values
        i_p[length+1:-1, :self.key_w] = keys_perm

        i_p[length,-1] = 1
        i_p[-1, -1] = 1

        o_p = np.zeros((2*length+2, self.key_w), dtype=np.float32)
        o_p[length+1:-1] = values_perm

        return self.transform({
            "input": i_p,
            "output": o_p
        })
