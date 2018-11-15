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

import torch
import torch.nn.functional as F
import numpy as np

float32 = [np.float32, torch.float32]
float64 = [np.float64, torch.float64]
uint8 = [np.uint8, torch.uint8]

_all_types = [float32, float64, uint8]

_dtype_numpy_map = {v[0]().dtype.name:v for v in _all_types}
_dtype_pytorch_map = {v[1]:v for v in _all_types}


def dtype(t):
    if torch.is_tensor(t):
        return _dtype_pytorch_map[t.dtype]
    else:
        return _dtype_numpy_map[t.dtype.name]


def cast(t, type):
    if torch.is_tensor(t):
        return t.type(type[1])
    else:
        return t.astype(type[0])


def to_numpy(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    else:
        return t


def to_list(t):
    t = to_numpy(t)
    if isinstance(t, np.ndarray):
        t = t.tolist()
    return t


def is_tensor(t):
    return torch.is_tensor(t) or isinstance(t, np.ndarray)


def first_batch(t):
    if is_tensor(t):
        return t[0]
    else:
        return t


def ndim(t):
    if torch.is_tensor(t):
        return t.dim()
    else:
        return t.ndim


def shape(t):
    return list(t.shape)


def transpose(t, axis):
    if torch.is_tensor(t):
        return t.permute(axis)
    else:
        return np.transpose(t, axis)


def apply_recursive(d, fn, filter=None):
    if isinstance(d, list):
        return [apply_recursive(da, fn) for da in d]
    elif isinstance(d, tuple):
        return tuple(apply_recursive(list(d), fn))
    elif isinstance(d, dict):
        return {k: apply_recursive(v, fn) for k, v in d.items()}
    else:
        if filter is None or filter(d):
            return fn(d)
        else:
            return d


def apply_to_tensors(d, fn):
    return apply_recursive(d, fn, torch.is_tensor)


def recursive_decorator(apply_this_fn):
    def decorator(func):
        def wrapped_funct(*args, **kwargs):
            args = apply_recursive(args, apply_this_fn)
            kwargs = apply_recursive(kwargs, apply_this_fn)

            return func(*args, **kwargs)

        return wrapped_funct

    return decorator

untensor = recursive_decorator(to_numpy)
unnumpy = recursive_decorator(to_list)

def unbatch(only_if_dim_equal=None):
    if only_if_dim_equal is not None and not isinstance(only_if_dim_equal, list):
        only_if_dim_equal = [only_if_dim_equal]

    def get_first_batch(t):
        if is_tensor(t) and (only_if_dim_equal is None or ndim(t) in only_if_dim_equal):
            return t[0]
        else:
            return t

    return recursive_decorator(get_first_batch)


def sigmoid(t):
    if torch.is_tensor(t):
        return torch.sigmoid(t)
    else:
        return 1.0 / (1.0 + np.exp(-t))


def argmax(t, dim):
    if torch.is_tensor(t):
        _, res = t.max(dim)
    else:
        res = np.argmax(t, axis=dim)

    return res


def flip(t, axis):
    if torch.is_tensor(t):
        return t.flip(axis)
    else:
        return np.flip(t, axis)


def transpose(t, axes):
    if torch.is_tensor(t):
        return t.permute(*axes)
    else:
        return np.transpose(t, axes)


def split_n(t, axis):
    if torch.is_tensor(t):
        return t.split(1, dim=axis)
    else:
        return np.split(t, t.shape[axis], axis=axis)


def cat(array_of_tensors, axis):
    if torch.is_tensor(array_of_tensors[0]):
        return torch.cat(array_of_tensors, axis)
    else:
        return np.concatenate(array_of_tensors, axis)


def clamp(t, min=None, max=None):
    if torch.is_tensor(t):
        return t.clamp(min, max)
    else:
        if min is not None:
            t = np.maximum(t, min)

        if max is not None:
            t = np.minimum(t, max)

        return t


def power(t, p):
    if torch.is_tensor(t) or torch.is_tensor(p):
        return torch.pow(t, p)
    else:
        return np.power(t, p)


def random_normal_as(a, mean, std, seed=None):
    if torch.is_tensor(a):
        return torch.randn_like(a) * std + mean
    else:
        if seed is None:
            seed = np.random
        return seed.normal(loc=mean, scale=std, size=shape(a))


def pad(t, pad):
    assert ndim(t) == 4

    if torch.is_tensor(t):
        return F.pad(t, pad)
    else:
        assert np.pad(t, ([0,0], [0,0], pad[0:2], pad[2:]))


def dx(img):
    lsh = img[:, :, :, 2:]
    orig = img[:, :, :, :-2]

    return pad(0.5 * (lsh - orig), (1, 1, 0, 0))


def dy(img):
    ush = img[:, :, 2:, :]
    orig = img[:, :, :-2, :]

    return pad(0.5 * (ush - orig), (0, 0, 1, 1))


def reshape(t, shape):
    if torch.is_tensor(t):
        return t.view(*shape)
    else:
        return t.reshape(*shape)


def broadcast_to_beginning(t, target):
    if torch.is_tensor(t):
        nd_target = target.dim()
        t_shape = list(t.shape)
        return t.view(*t_shape, *([1]*(nd_target-len(t_shape))))
    else:
        nd_target = target.ndim
        t_shape = list(t.shape)
        return t.reshape(*t_shape, *([1] * (nd_target - len(t_shape))))


def lin_combine(d1,w1, d2,w2, bcast_begin=False):
    if isinstance(d1, (list, tuple)):
        assert len(d1) == len(d2)
        res = [lin_combine(d1[i], w1, d2[i], w2) for i in range(len(d1))]
        if isinstance(d1, tuple):
            res = tuple(d1)
    elif isinstance(d1, dict):
        res = {k: lin_combine(v, w1, d2[k], w2) for k, v in d1.items()}
    else:
        if bcast_begin:
            w1 = broadcast_to_beginning(w1, d1)
            w2 = broadcast_to_beginning(w2, d2)

        res = d1 * w1 + d2 * w2

    return res
