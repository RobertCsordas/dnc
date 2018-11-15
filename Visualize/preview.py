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
import threading
import traceback
import sys
from Utils import universal as U


def preview(vis_interval=None, to_numpy=True, debatch=False):
    def decorator(func):
        state = {
            "last_vis_time": 0,
            "thread_running": False
        }

        def wrapper_function(*args, **kwargs):
            if state["thread_running"]:
                return

            if vis_interval is not None:
                now = time.time()
                if now - state["last_vis_time"] < vis_interval:
                    return
                state["last_vis_time"] = now

            state["thread_running"] = True

            if debatch:
                args = U.apply_recursive(args, U.first_batch)
                kwargs = U.apply_recursive(kwargs, U.first_batch)

            if to_numpy:
                args = U.apply_recursive(args, U.to_numpy)
                kwargs = U.apply_recursive(kwargs, U.to_numpy)

            def run():
                try:
                    func(*args, **kwargs)
                except Exception:
                    traceback.print_exc()
                    sys.exit(-1)

                state["thread_running"] = False

            download_thread = threading.Thread(target=run)
            download_thread.start()

        return wrapper_function
    return decorator
