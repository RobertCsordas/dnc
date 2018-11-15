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

class Vocabulary:
    def __init__(self):
        self.words = {"-" : 0, "?": 1, "<UNK>": 2}
        self.inv_words = {0 : "-", 1: "?", 2: "<UNK>"}
        self.next_id = 3
        self.punctations = [".", "?", ","]

    def _process_word(self, w, add_words):
        if not w.isalpha() and w not in self.punctations:
            print("WARNING: word with unknown characters: %s", w)
            w = "<UNK>"

        if w not in self.words:
            if add_words:
                self.words[w] = self.next_id
                self.inv_words[self.next_id] = w
                self.next_id += 1
            else:
                w = "<UNK>"

        return self.words[w]

    def sentence_to_indices(self, sentence, add_words=True):
        for p in self.punctations:
            sentence = sentence.replace(p, " %s " % p)

        return [self._process_word(w, add_words) for w in sentence.lower().split(" ") if w]

    def indices_to_sentence(self, indices):
        return [self.inv_words[i] for i in indices]

    def __len__(self):
        return len(self.words)
