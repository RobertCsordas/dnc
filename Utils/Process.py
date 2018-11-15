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

import sys
import ctypes
import subprocess
import os

def run(cmd, hide_stderr = True):
    libc_search_dirs = ["/lib", "/lib/x86_64-linux-gnu", "/lib/powerpc64le-linux-gnu"]

    if sys.platform=="linux" :
        found = None
        for d in libc_search_dirs:
            file = os.path.join(d, "libc.so.6")
            if os.path.isfile(file):
                found = file
                break

        if not found:
            print("WARNING: Cannot find libc.so.6. Cannot kill process when parent dies.")
            killer = None
        else:
            libc = ctypes.CDLL(found)
            PR_SET_PDEATHSIG = 1
            KILL = 9
            killer = lambda: libc.prctl(PR_SET_PDEATHSIG, KILL)
    else:
        print("WARNING: OS not linux. Cannot kill process when parent dies.")
        killer = None


    if hide_stderr:
        stderr = open(os.devnull,'w')
    else:
        stderr = None

    return subprocess.Popen(cmd.split(" "), stderr=stderr, preexec_fn=killer)
