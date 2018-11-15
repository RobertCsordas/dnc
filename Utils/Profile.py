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

import atexit

ENABLED=False

_profiler = None


def construct():
    global _profiler
    if not ENABLED:
        return

    if _profiler is None:
        from line_profiler import LineProfiler
        _profiler = LineProfiler()


def do_profile(follow=[]):
    construct()
    def inner(func):
        if _profiler is not None:
            _profiler.add_function(func)
            for f in follow:
                _profiler.add_function(f)
            _profiler.enable_by_count()
        return func
    return inner

@atexit.register
def print_prof():
    if _profiler is not None:
        _profiler.print_stats()
