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

import time


class OnceEvery:
    def __init__(self, interval):
        self._interval = interval
        self._last_check = 0

    def __call__(self):
        now = time.time()
        if now - self._last_check >= self._interval:
            self._last_check = now
            return True
        else:
            return False


class Measure:
    def __init__(self, average=1):
        self._start = None
        self._average = average
        self._accu_value = 0.0
        self._history_list = []

    def start(self):
        self._start = time.time()

    def passed(self):
        if self._start is None:
            return None

        p = time.time() - self._start
        self._history_list.append(p)
        self._accu_value += p
        if len(self._history_list) > self._average:
            self._accu_value -= self._history_list.pop(0)

        return self._accu_value / len(self._history_list)
