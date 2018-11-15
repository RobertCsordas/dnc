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
import json
import argparse


class ArgumentParser:
    class Parsed:
        pass

    _type = type

    @staticmethod
    def str_or_none(none_string="none"):
        def parse(s):
            return None if s.lower()==none_string else s
        return parse

    @staticmethod
    def list_or_none(none_string="none", type=int):
        def parse(s):
            return None if s.lower() == none_string else [type(a) for a in s.split(",") if a]
        return parse

    @staticmethod
    def _merge_args(args, new_args, arg_schemas):
        for name, val in new_args.items():
            old = args.get(name)
            if old is None:
                args[name] = val
            else:
                args[name] = arg_schemas[name]["updater"](old, val)

    class Profile:
        def __init__(self, name, args=None, include=[]):
            assert not (args is None and not include), "One of args or include must be defined"
            self.name = name
            self.args = args
            if not isinstance(include, list):
                include=[include]
            self.include = include

        def get_args(self, arg_schemas, profile_by_name):
            res = {}

            for n in self.include:
                p = profile_by_name.get(n)
                assert p is not None, "Included profile %s doesn't exists" % n

                ArgumentParser._merge_args(res, p.get_args(arg_schemas, profile_by_name), arg_schemas)

            ArgumentParser._merge_args(res, self.args, arg_schemas)
            return res


    def __init__(self, description=None):
        self.parser = argparse.ArgumentParser(description=description)
        self.loaded = {}
        self.profiles = {}
        self.args = {}
        self.raw = None
        self.parsed = None
        self.parser.add_argument("-profile", type=str, help="Pre-defined profiles.")

    def add_argument(self, name, type=None, default=None, help="", save=True, parser=lambda x:x, updater=lambda old, new:new):
        assert name not in ["profile"], "Argument name %s is reserved" % name
        assert not (type is None and default is None), "Either type or default must be given"

        if type is None:
            type = ArgumentParser._type(default)

        self.parser.add_argument(name, type=int if type==bool else type, default=None, help=help)
        if name[0] == '-':
            name = name[1:]

        self.args[name] = {
            "type": type,
            "default": int(default) if type==bool else default,
            "save": save,
            "parser": parser,
            "updater": updater
        }

    def add_profile(self, prof):
        if isinstance(prof, list):
            for p in prof:
                self.add_profile(p)
        else:
            self.profiles[prof.name] = prof

    def do_parse_args(self, loaded={}):
        self.raw = self.parser.parse_args()

        profile = {}
        if self.raw.profile:
            assert not loaded, "Loading arguments from file, but profile given."
            for pr in self.raw.profile.split(","):
                p = self.profiles.get(pr)
                assert p is not None, "Invalid profile: %s. Valid profiles: %s" % (pr, self.profiles.keys())
                p = p.get_args(self.args, self.profiles)
                self._merge_args(profile, p, self.args)

        for k, v in self.raw.__dict__.items():
            if k in ["profile"]:
                continue

            if v is None:
                if k in loaded and self.args[k]["save"]:
                    self.raw.__dict__[k] = loaded[k]
                else:
                    self.raw.__dict__[k] = profile.get(k, self.args[k]["default"])

        self.parsed = ArgumentParser.Parsed()
        self.parsed.__dict__.update({k: self.args[k]["parser"](self.args[k]["type"](v)) if v is not None else None
                                     for k,v in self.raw.__dict__.items() if k in self.args})

        return self.parsed

    def parse_or_cache(self):
        if self.parsed is None:
            self.do_parse_args()

    def parse(self):
        self.parse_or_cache()
        return self.parsed

    def save(self, fname):
        self.parse_or_cache()
        with open(fname, 'w') as outfile:
            json.dump(self.raw.__dict__, outfile, indent=4)
            return True

    def load(self, fname):
        if os.path.isfile(fname):
            map = {}
            with open(fname, "r") as data_file:
                map = json.load(data_file)

            self.do_parse_args(map)
        return self.parsed

    def sync(self, fname, save=True):
        if os.path.isfile(fname):
            self.load(fname)

        if save:
            dir = os.path.dirname(fname)
            if not os.path.isdir(dir):
                os.makedirs(dir)

            self.save(fname)
        return self.parsed
