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
import os
from .Vocabulary import Vocabulary
from Utils import Visdom
from Utils import universal as U

class NLPTask(torch.utils.data.Dataset):
    def __init__(self):
        super(NLPTask, self).__init__()

        self.my_dir = os.path.abspath(os.path.dirname(__file__))
        self.cache_dir = os.path.join(self.my_dir, "cache")

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.vocabulary = self._load_vocabulary()

        self._preview = None

    def _load_vocabulary(self):
        cache_file = os.path.join(self.cache_dir, "vocabulary.pth")
        if not os.path.isfile(cache_file):
            print("WARNING: Vocabulary not found. Removing cached files.")
            for f in os.listdir(self.cache_dir):
                f = os.path.join(self.cache_dir, f)
                if f.endswith(".pth"):
                    print("   "+f)
                    os.remove(f)
            return Vocabulary()
        else:
            return torch.load(cache_file)

    def save_vocabulary(self):
        cache_file = os.path.join(self.cache_dir, "vocabulary.pth")
        if os.path.isfile(cache_file):
            os.remove(cache_file)
        torch.save(self.vocabulary, cache_file)

    def loss(self, net_output, target):
        s = list(net_output.size())
        return F.cross_entropy(net_output.view([s[0]*s[1], s[2]]), target.view([-1]), ignore_index=0,
                               reduction='sum')/s[0]

    def generate_preview_text(self, data, net_output):
        input = U.to_numpy(data["input"][0])
        reference = U.to_numpy(data["output"][0])
        net_out = U.argmax(net_output[0], -1)
        net_out = U.to_numpy(net_out)

        res = ""
        start_index = 0

        for i in range(input.shape[0]):
            if reference[i] != 0:
                if start_index < i:
                    end_index = i
                    while end_index>start_index and input[end_index]==0:
                        end_index -= 1

                    if end_index>start_index:
                        sentence = " ".join(self.vocabulary.indices_to_sentence(input[start_index:i].tolist())). \
                            replace(" .", ".").replace(" ,", ",").replace(" ?", "?").split(". ")
                        sentence = ". ".join([s.capitalize() for s in sentence])
                        res += sentence + "<br>"

                start_index = i + 1

                match = reference[i] == net_out[i]
                res += "<b><font color=\"%s\">%s [%s]</font><br></b>" % ("green" if match else "red",
                                                                         self.vocabulary.indices_to_sentence(
                                                                             [net_out[i]])[0],
                                                                         self.vocabulary.indices_to_sentence(
                                                                             [reference[i]])[0])
        return res

    def visualize_preview(self, data, net_output):
        res = self.generate_preview_text(data, net_output)

        if self._preview is None:
            self._preview = Visdom.Text("Preview")

        self._preview.set(res)

    def set_dump_dir(self, dir):
        pass