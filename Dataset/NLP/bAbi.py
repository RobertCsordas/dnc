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
import glob
import torch
from collections import namedtuple
import numpy as np
from .NLPTask import NLPTask
from Utils import Visdom

Sentence = namedtuple('Sentence', ['sentence', 'answer', 'supporting_facts'])

class bAbiDataset(NLPTask):
    URL = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'
    DIR_NAME = "tasks_1-20_v1-2"
    
    def __init__(self, dirs = ["en-10k"], sets=None, think_steps=0, dir_name=None, name=None):
        super(bAbiDataset, self).__init__()

        self._test_res_win = None
        self._test_plot_win = None
        self._think_steps = think_steps

        if dir_name is None:
            self._download()
            dir_name = os.path.join(self.cache_dir, self.DIR_NAME)

        self.data={}
        for d in dirs:
            self.data[d] = self._load_or_create(os.path.join(dir_name, d))

        self.all_tasks=None
        self.name = name
        self.use(sets=sets)

    def _make_active_list(self, tasks, sets, dirs):
        def verify(name, checker):
            if checker is None:
                return True

            if callable(checker):
                return checker(name)
            elif isinstance(checker, list):
                return name in checker
            else:
                return name==checker

        res = []
        for dirname, setlist in self.data.items():
            if not verify(dirname, dirs):
                continue

            for sname, tasklist in setlist.items():
                if not verify(sname, sets):
                    continue

                for task, data in tasklist.items():
                    name = task.split("_")[0][2:]
                    if not verify(name, tasks):
                        continue

                    res += [(d, dirname, task, sname) for d in data]

        return res

    def use(self, tasks=None, sets=None, dirs=None):
        self.all_tasks=self._make_active_list(tasks=tasks, sets=sets, dirs=dirs)

    def __len__(self):
        return len(self.all_tasks)

    def _get_seq(self, index):
        return self.all_tasks[index]

    def _seq_to_nn_input(self, seq):
        in_arr = []
        out_arr = []
        hasAnswer = False
        for sentence in seq[0]:
            in_arr += sentence.sentence
            out_arr += [0] * len(sentence.sentence)
            if sentence.answer is not None:
                in_arr += [0] * (len(sentence.answer) + self._think_steps)
                out_arr += [0] * self._think_steps + sentence.answer
                hasAnswer = True

        in_arr = np.asarray(in_arr, np.int64)
        out_arr = np.asarray(out_arr, np.int64)

        return {
            "input": in_arr,
            "output": out_arr,
            "meta": {
                "dir": seq[1],
                "task": seq[2],
                "set": seq[3]
            }
        }

    def __getitem__(self, item):
        seq = self._get_seq(item)
        return self._seq_to_nn_input(seq)

    def _load_or_create(self, directory):
        cache_name = directory.replace("/","_")
        cache_file = os.path.join(self.cache_dir, cache_name+".pth")
        if not os.path.isfile(cache_file):
            print("bAbI: Loading %s" % directory)
            res = self._load_dir(directory)
            print("Write: ", cache_file)
            self.save_vocabulary()
            torch.save(res, cache_file)
        else:
            res = torch.load(cache_file)
        return res
            
    def _download(self):
        if not os.path.isdir(os.path.join(self.cache_dir, self.DIR_NAME)):
            print(self.URL)
            print("bAbi data not found. Downloading...")
            import requests, tarfile, io
            request = requests.get(self.URL, headers={"User-agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36"})
            
            decompressed_file = tarfile.open(fileobj=io.BytesIO(request.content), mode='r|gz')
            decompressed_file.extractall(self.cache_dir)
            print("Done")
     
    def _load_dir(self, directory, parse_name = lambda x: x.split(".")[0], parse_set = lambda x: x.split(".")[0].split("_")[-1]):
        res = {}
        for f in glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True):
            basename = os.path.basename(f)
            task_name = parse_name(basename)
            set = parse_set(basename)
            print("Loading", f)

            s = res.get(set)
            if s is None:
                s = {}
                res[set] = s
            s[task_name] = self._load_task(f, task_name)
            
        return res
    
    def _load_task(self, filename, task_name):
        task = []
        currTask = []
        
        nextIndex = 1
        with open(filename, "r") as f:
            for line in f:
                line = [f.strip() for f in line.split("\t")]
                line[0] = line[0].split(" ")
                i = int(line[0][0])
                line[0] = " ".join(line[0][1:])
                
                if i!=nextIndex:
                    nextIndex = i
                    task.append(currTask)
                    currTask = []

                isQuestion = len(line)>1
                currTask.append(
                    Sentence(self.vocabulary.sentence_to_indices(line[0]), self.vocabulary.sentence_to_indices(line[1].replace(",", " "))
                            if isQuestion else None, [int(f) for f in line[2].split(" ")] if isQuestion else None)
                )
                
                nextIndex += 1
        return task

    def start_test(self):
        return {}

    def veify_result(self, test, data, net_output):
        _, net_output = net_output.max(-1)

        ref = data["output"]

        mask = 1.0 - ref.eq(0).float()

        correct = (torch.eq(net_output, ref).float() * mask).sum(-1)
        total = mask.sum(-1)

        correct = correct.data.cpu().numpy()
        total = total.data.cpu().numpy()

        for i in range(correct.shape[0]):
            task = data["meta"][i]["task"]
            if task not in test:
                test[task] = {"total": 0, "correct": 0}

            d = test[task]
            d["total"] += total[i]
            d["correct"] += correct[i]

    def _ensure_test_wins_exists(self, legend = None):
        if self._test_res_win is None:
            n = (("[" + self.name + "]") if self.name is not None else "")
            self._test_res_win = Visdom.Text("Test results" + n)
            self._test_plot_win = Visdom.Plot2D("Test results" + n, legend=legend)
        elif self._test_plot_win.legend is None:
            self._test_plot_win.set_legend(legend=legend)

    def show_test_results(self, iteration, test):
        res = {k: v["correct"]/v["total"] for k, v in test.items()}

        t = ""

        all_keys = list(res.keys())

        num_keys = [k for k in all_keys if k.startswith("qa")]
        tmp = [i[0] for i in sorted(enumerate(num_keys), key=lambda x:int(x[1][2:].split("_")[0]))]
        num_keys = [num_keys[j] for j in tmp]

        all_keys = num_keys + sorted([k for k in all_keys if not k.startswith("qa")])

        err_precent = [(1.0-res[k]) * 100.0 for k in all_keys]

        n_passed = sum([int(p<=5) for p in err_precent])
        n_total = len(err_precent)
        err_precent = err_precent + [sum(err_precent) / len(err_precent)]
        all_keys += ["mean"]

        for i, k in enumerate(all_keys):
            t += "<font color=\"%s\">%s: <b>%.2f%%</b></font><br>" % ("green" if err_precent[i] <= 5 else "red", k, err_precent[i])

        t += "<br><b>Total: %d of %d passed.</b>" % (n_passed, n_total)

        self._ensure_test_wins_exists(legend=[i.split("_")[0] if i.startswith("qa") else i for i in all_keys])

        self._test_res_win.set(t)
        self._test_plot_win.add_point(iteration, err_precent)


    def state_dict(self):
        if self._test_res_win is not None:
            return {
                "_test_res_win" : self._test_res_win.state_dict(),
                "_test_plot_win": self._test_plot_win.state_dict(),
            }
        else:
            return {}

    def load_state_dict(self, state):
        if state:
            self._ensure_test_wins_exists()
            self._test_res_win.load_state_dict(state["_test_res_win"])
            self._test_plot_win.load_state_dict(state["_test_plot_win"])
            self._test_plot_win.legend = None

    def visualize_preview(self, data, net_output):
        res = self.generate_preview_text(data, net_output)
        res = ("<b><u>%s</u></b><br>" % data["meta"][0]["task"]) + res
        if self._preview is None:
            self._preview = Visdom.Text("Preview")

        self._preview.set(res)