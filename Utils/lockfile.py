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

import os
import fcntl


class LockFile:
    def __init__(self, fname):
        self._fname = fname
        self._fd = None

    def acquire(self):
        self._fd=open(self._fname, "w")
        os.chmod(self._fname, 0o777)

        fcntl.lockf(self._fd, fcntl.LOCK_EX)

    def release(self):
        fcntl.lockf(self._fd, fcntl.LOCK_UN)
        self._fd.close()
        self._fd = None

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
