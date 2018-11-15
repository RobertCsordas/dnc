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
import torch
from operator import mul
from functools import reduce

class VarLengthCollate:
    def __init__(self, ignore_symbol=0):
        self.ignore_symbol = ignore_symbol

    def _measure_array_max_dim(self, batch):
        s=list(batch[0].size())
        different=[False] * len(s)
        for i in range(1, len(batch)):
            ns = batch[i].size()
            different = [different[j] or s[j]!=ns[j] for j in range(len(s))]
            s=[max(s[j], ns[j]) for j in range(len(s))]
        return s, different

    def _merge_var_len_array(self, batch):
        max_size, different = self._measure_array_max_dim(batch)
        s=[len(batch)] + max_size
        storage = batch[0].storage()._new_shared(reduce(mul, s, 1))
        out = batch[0].new(storage).view(s).fill_(self.ignore_symbol)
        for i, d in enumerate(batch):
            this_o = out[i]
            for j, diff in enumerate(different):
                if different[j]:
                    this_o = this_o.narrow(j,0,d.size(j))
            this_o.copy_(d)
        return out


    def __call__(self, batch):
        if isinstance(batch[0], dict):
            return {k: self([b[k] for b in batch]) for k in batch[0].keys()}
        elif isinstance(batch[0], np.ndarray):
            return self([torch.from_numpy(a) for a in batch])
        elif torch.is_tensor(batch[0]):
            return self._merge_var_len_array(batch)
        else:
            assert False, "Unknown type: %s" % type(batch[0])

class MetaCollate:
    def __init__(self, meta_name="meta", collate=VarLengthCollate()):
        self.meta_name = meta_name
        self.collate = collate

    def __call__(self, batch):
        if isinstance(batch[0], dict):
            meta = [b[self.meta_name] for b in batch]
            batch = [{k: v for k,v in b.items() if k!=self.meta_name} for b in batch]
        else:
            meta = None

        res = self.collate(batch)
        if meta is not None:
            res[self.meta_name] = meta

        return res