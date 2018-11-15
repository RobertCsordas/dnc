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
import random
from .BitmapTask import BitmapTask

class BitmapTaskRepeater(BitmapTask):
    def __init__(self, dataset):
        super(BitmapTaskRepeater, self).__init__()
        self.dataset = dataset

    def __getitem__(self, key):
        r = [self.dataset[k] for k in key]
        if len(r)==1:
            return r[0]
        else:
            return {
                "input": np.concatenate([a["input"] for a in r], axis=0),
                "output": np.concatenate([a["output"] for a in r], axis=0)
            }

    @staticmethod
    def key_sampler(length, repeat):
        def call_sampler(s):
            if callable(s):
                return s()
            elif isinstance(s, list):
                if len(s) == 2:
                    return random.randint(*s)
                elif len(s) == 1:
                    return s[0]
                else:
                    assert False, "Invalid sample parameter: %s" % s
            else:
                return s

        def s():
            r = call_sampler(repeat)
            return [call_sampler(length) for i in range(r)]

        return s
