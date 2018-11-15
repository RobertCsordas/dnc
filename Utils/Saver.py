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
import os
import inspect
import time


class SaverElement:
    def save(self):
        raise NotImplementedError

    def load(self, saved_state):
        raise NotImplementedError


class CallbackSaver(SaverElement):
    def __init__(self, save_fn, load_fn):
        super().__init__()
        self.save = save_fn
        self.load = load_fn


class StateSaver(SaverElement):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def load(self, state):
        try:
            self._model.load_state_dict(state)
        except Exception as e:
            if hasattr(self._model, "named_parameters"):
                names = set([n for n, _ in self._model.named_parameters()])
                loaded = set(self._model.keys())
                if names!=loaded:
                    d = loaded.difference(names)
                    if d:
                        print("Loaded, but not in model: %s" % list(d))
                    d = names.difference(loaded)
                    if d:
                        print("In model, but not loaded: %s" % list(d))
            if isinstance(self._model, torch.optim.Optimizer):
                print("WARNING: optimizer parameters not loaded!")
            else:
                raise e

    def save(self):
        return self._model.state_dict()


class GlobalVarSaver(SaverElement):
    def __init__(self, name):
        caller_frame = inspect.getouterframes(inspect.currentframe())[1]
        self._vars = caller_frame.frame.f_globals
        self._name = name

    def load(self, state):
        self._vars.update({self._name: state})

    def save(self):
        return self._vars[self._name]


class PyObjectSaver(SaverElement):
    def __init__(self, obj):
        self._obj = obj

    def load(self, state):
        def _load(target, state):
            if isinstance(target, dict):
                for k, v in state.items():
                    target[k] = _load(target.get(k), v)
            elif isinstance(target, list):
                if len(target)!=len(state):
                    target.clear()
                    for v in state:
                        target.append(v)
                else:
                    for i, v in enumerate(state):
                        target[i] = _load(target[i], v)

            elif hasattr(target, "__dict__"):
                _load(target.__dict__, state)
            else:
                return state
            return target

        _load(self._obj, state)

    def save(self):
        def _save(target):
            if isinstance(target, dict):
                res = {k: _save(v) for k, v in target.items()}
            elif isinstance(target, list):
                res = [_save(v) for v in target]
            elif hasattr(target, "__dict__"):
                res = {k: _save(v) for k, v in target.__dict__.items()}
            else:
                res = target

            return res

        return _save(self._obj)

    @staticmethod
    def obj_supported(obj):
        return isinstance(obj, (list, dict)) or hasattr(obj, "__dict__")


class Saver:
    def __init__(self, dir, short_interval, keep_every_n_hours=4):
        self.savers = {}
        self.short_interval = short_interval
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self._keep_every_n_seconds = keep_every_n_hours * 3600

    def register(self, name, saver):
        assert name not in self.savers, "Saver %s already registered" % name

        if isinstance(saver, SaverElement):
            self.savers[name] = saver
        elif hasattr(saver, "state_dict") and callable(saver.state_dict):
            self.savers[name] = StateSaver(saver)
        elif PyObjectSaver.obj_supported(saver):
            self.savers[name] = PyObjectSaver(saver)
        else:
            assert "Unsupported thing to save: %s" % type(saver)

    def __setitem__(self, key, value):
        self.register(key, value)

    def write(self, iter):
        fname = os.path.join(self.dir, self.model_name_from_index(iter))
        print("Saving %s" % fname)

        state = {}
        for name, fns in self.savers.items():
            state[name] = fns.save()

        torch.save(state, fname)
        print("Saved.")

        self._cleanup()

    def tick(self, iter):
        if iter % self.short_interval != 0:
            return

        self.write(iter)

    @staticmethod
    def model_name_from_index(index):
        return "model-%d.pth" % index

    @staticmethod
    def get_checkpoint_index_list(dir):
        return list(reversed(sorted(
            [int(fn.split(".")[0].split("-")[-1]) for fn in os.listdir(dir) if fn.split(".")[-1] == "pth"])))

    @staticmethod
    def get_ckpts_in_time_window(dir, time_window_s, index_list=None):
        if index_list is None:
            index_list = Saver.get_checkpoint_index_list(dir)


        now = time.time()

        res = []
        for i in index_list:
            name = Saver.model_name_from_index(i)
            mtime = os.path.getmtime(os.path.join(dir, name))
            if now - mtime > time_window_s:
                break

            res.append(name)

        return res

    @staticmethod
    def load_last_checkpoint(dir):
        last_checkpoint = Saver.get_checkpoint_index_list(dir)

        if last_checkpoint:
            for index in last_checkpoint:
                fname = Saver.model_name_from_index(index)
                try:
                    print("Loading %s" % fname)
                    data = torch.load(os.path.join(dir, fname))
                except:
                    print("WARNING: Loading %s failed. Maybe file is corrupted?" % fname)
                    continue
                return data
        return None

    def _cleanup(self):
        index_list = self.get_checkpoint_index_list(self.dir)
        new_files = self.get_ckpts_in_time_window(self.dir, self._keep_every_n_seconds, index_list[2:])
        new_files = new_files[:-1]

        for f in new_files:
            os.remove(os.path.join(self.dir, f))

    def load(self, fname=None):
        if fname is None:
            state = self.load_last_checkpoint(self.dir)
            if not state:
                return False
        else:
            state = torch.load(fname)

        for k,s in state.items():
            if k not in self.savers:
                print("WARNING: failed to load state of %s. It doesn't exists." % k)
                continue
            self.savers[k].load(s)

        return True
