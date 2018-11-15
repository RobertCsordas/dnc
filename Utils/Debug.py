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
import sys
import traceback
import torch

enableDebug = False

def nan_check(arg, name=None, force=False):
    if not enableDebug and not force:
        return arg
    is_nan = False
    curr_nan = False
    if isinstance(arg, torch.autograd.Variable):
        curr_nan = not np.isfinite(arg.sum().cpu().data.numpy())
    elif isinstance(arg, torch.nn.parameter.Parameter):
        curr_nan = (not np.isfinite(arg.sum().cpu().data.numpy())) or (not np.isfinite(arg.grad.sum().cpu().data.numpy()))
    elif isinstance(arg, float):
        curr_nan = not np.isfinite(arg)
    elif isinstance(arg, (list, tuple)):
        for a in arg:
            nan_check(a)
    else:
        assert False, "Unsupported type %s" % type(arg)

    if curr_nan:
        if sys.exc_info()[0] is not None:
            trace = str(traceback.format_exc())
        else:
            trace = "".join(traceback.format_stack())

        print(arg)
        if name is not None:
            print("NaN found in %s." % name)
        else:
            print("NaN found.")
        if isinstance(arg, torch.autograd.Variable):
            print("     Argument is a torch tensor. Shape: %s" % list(arg.size()))

        print(trace)
        sys.exit(-1)

    return arg


def assert_range(t, min=0.0, max=1.0):
    if not enableDebug:
        return

    if t.min().cpu().data.numpy()<min or t.max().cpu().data.numpy()>max:
        print(t)
        assert False


def assert_dist(t, use_lower_limit=True):
    if not enableDebug:
        return

    assert_range(t)

    if t.sum(-1).max().cpu().data.numpy()>1.001:
        print("MAT:", t)
        print("SUM:", t.sum(-1))
        assert False

    if use_lower_limit and t.sum(-1).max().cpu().data.numpy()<0.999:
        print(t)
        print("SUM:", t.sum(-1))
        assert False


def print_stat(name, t):
    if not enableDebug:
        return

    min = t.min().cpu().data.numpy()
    max = t.max().cpu().data.numpy()
    mean = t.mean().cpu().data.numpy()

    print("%s: min: %g, mean: %g, max: %g" % (name, min, mean, max))


def dbg_print(*things):
    if not enableDebug:
        return
    print(*things)

class GradPrinter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        return a

    @staticmethod
    def backward(ctx, g):
        print("Grad (print_grad): ", g[0])
        return g

def print_grad(t):
    return GradPrinter.apply(t)

def assert_equal(t1, ref, limit=1e-5, force=True):
    if not (enableDebug or force):
        return

    assert t1.shape==ref.shape, "Tensor shapes differ: got %s, ref %s" % (t1.shape, ref.shape)
    norm = ref.abs().sum() / ref.nonzero().sum().float()
    threshold = norm * limit

    errcnt = ((t1 - ref).abs() > threshold).sum()
    if errcnt > 0:
        print("Tensors differ. (max difference: %g, norm %f). No of errors: %d of %d" %
              ((t1 - ref).abs().max().item(), norm, errcnt, t1.numel()))
        print("---------------------------------------------Out-----------------------------------------------")
        print(t1)
        print("---------------------------------------------Ref-----------------------------------------------")
        print(ref)
        print("-----------------------------------------------------------------------------------------------")
        assert False