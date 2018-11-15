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

import subprocess
import os
import torch
from Utils.lockfile import LockFile

def get_memory_usage():
    try:
        proc = subprocess.Popen("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits".split(" "),
                         stdout=subprocess.PIPE)
        lines = [s.strip().split(" ") for s in proc.communicate()[0].decode().split("\n") if s]
        return {int(g[0][:-1]): int(g[1]) for g in lines}
    except:
        return None


def get_free_gpus():
    try:
        free = []
        proc = subprocess.Popen("nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits".split(" "),
                                stdout=subprocess.PIPE)
        uuids = [s.strip() for s in proc.communicate()[0].decode().split("\n") if s]

        proc = subprocess.Popen("nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits".split(" "),
                                stdout=subprocess.PIPE)

        id_uid_pair = [s.strip().split(", ") for s in proc.communicate()[0].decode().split("\n") if s]
        for i in id_uid_pair:
            id, uid = i

            if uid not in uuids:
                free.append(int(id))

        return free
    except:
        return None

def _fix_order():
    os.environ["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

def allocate(n:int = 1):
    _fix_order()
    with LockFile("/tmp/gpu_allocation_lock"):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print("WARNING: trying to allocate %d GPUs, but CUDA_VISIBLE_DEVICES already set to %s" %
                  (n, os.environ["CUDA_VISIBLE_DEVICES"]))
            return

        allocated = get_free_gpus()
        if allocated is None:
            print("WARNING: failed to allocate %d GPUs" % n)
            return
        allocated = allocated[:n]

        if len(allocated) < n:
            print("There is no more free GPUs. Allocating the one with least memory usage.")
            usage = get_memory_usage()
            if usage is None:
                print("WARNING: failed to allocate %d GPUs" % n)
                return

            inv_usages = {}

            for k, v in usage.items():
                if v not in inv_usages:
                    inv_usages[v] = []

                inv_usages[v].append(k)

            min_usage = list(sorted(inv_usages.keys()))
            min_usage_devs = []
            for u in min_usage:
                min_usage_devs += inv_usages[u]

            min_usage_devs = [m for m in min_usage_devs if m not in allocated]

            n2 = n - len(allocated)
            if n2>len(min_usage_devs):
                print("WARNING: trying to allocate %d GPUs but only %d available" % (n, len(min_usage_devs)+len(allocated)))
                n2 = len(min_usage_devs)

            allocated += min_usage_devs[:n2]

        os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(a) for a in allocated])
        for i in range(len(allocated)):
            a = torch.FloatTensor([0.0])
            a.cuda(i)

def use_gpu(gpu="auto", n_autoalloc=1):
    _fix_order()

    gpu = gpu.lower()
    if gpu in ["auto", ""]:
        allocate(n_autoalloc)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
