#!/usr/bin/env python3
#
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

import functools
import os

import torch.utils.data

import Utils.Debug as debug
from Dataset.Bitmap.AssociativeRecall import AssociativeRecall
from Dataset.Bitmap.BitmapTaskRepeater import BitmapTaskRepeater
from Dataset.Bitmap.KeyValue import KeyValue
from Dataset.Bitmap.CopyTask import CopyData
from Dataset.Bitmap.KeyValue2Way import KeyValue2Way
from Dataset.NLP.bAbi import bAbiDataset
from Models.DNC import DNC, LSTMController, FeedforwardController
from Utils import Visdom
from Utils.ArgumentParser import ArgumentParser
from Utils.Index import index_by_dim
from Utils.Saver import Saver, GlobalVarSaver, StateSaver
from Utils.Collate import MetaCollate
from Utils import gpu_allocator
from Dataset.NLP.NLPTask import NLPTask
from tqdm import tqdm
from Visualize.preview import preview
from Utils.timer import OnceEvery
from Utils import Seed
import time
import sys
import signal
import math
from Utils import Profile

Profile.ENABLED=False

def main():
    global i
    global loss_sum
    global running
    parser = ArgumentParser()
    parser.add_argument("-bit_w", type=int, default=8, help="Bit vector length for copy task")
    parser.add_argument("-block_w", type=int, default=3, help="Block width to associative recall task")
    parser.add_argument("-len", type=str, default="4", help="Sequence length for copy task", parser=lambda x: [int(a) for a in x.split("-")])
    parser.add_argument("-repeat", type=str, default="1", help="Sequence length for copy task", parser=lambda x: [int(a) for a in x.split("-")])
    parser.add_argument("-batch_size", type=int, default=16, help="Sequence length for copy task")
    parser.add_argument("-n_subbatch", type=str, default="auto", help="Average this much forward passes to a backward pass")
    parser.add_argument("-max_input_count_per_batch", type=int, default=6000, help="Max batch_size*len that can fit into memory")
    parser.add_argument("-lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("-optimizer", type=str, default="rmsprop", help="Optimizer algorithm")
    parser.add_argument("-name", type=str, help="Save training to this directory")
    parser.add_argument("-preview_interval", type=int, default=10, help="Show preview every nth iteration")
    parser.add_argument("-info_interval", type=int, default=10, help="Show info every nth iteration")
    parser.add_argument("-save_interval", type=int, default=500, help="Save network every nth iteration")
    parser.add_argument("-masked_lookup", type=bool, default=1, help="Enable masking in content lookups")
    parser.add_argument("-visport", type=int, default=-1, help="Port to run Visdom server on. -1 to disable")
    parser.add_argument("-gpu", default="auto", type=str, help="Run on this GPU.")
    parser.add_argument("-debug", type=bool, default=1, help="Enable debugging")
    parser.add_argument("-task", type=str, default="copy", help="Task to learn")
    parser.add_argument("-mem_count", type=int, default=16, help="Number of memory cells")
    parser.add_argument("-data_word_size", type=int, default=128, help="Memory word size")
    parser.add_argument("-n_read_heads", type=int, default=1, help="Number of read heads")
    parser.add_argument("-layer_sizes", type=str, default="256", help="Controller layer sizes. Separate with ,. For example 512,256,256", parser=lambda x: [int(y) for y in x.split(",") if y])
    parser.add_argument("-debug_log", type=bool, default=0, help="Enable debug log")
    parser.add_argument("-controller_type", type=str, default="lstm", help="Controller type: lstm or linear")
    parser.add_argument("-lstm_use_all_outputs", type=bool, default=1, help="Use all LSTM outputs as controller output vs use only the last layer")
    parser.add_argument("-momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("-embedding_size", type=int, default=256, help="Size of word embedding for NLP tasks")
    parser.add_argument("-test_interval", type=int, default=10000, help="Run test in this interval")
    parser.add_argument("-dealloc_content", type=bool, default=1, help="Deallocate memory content, unlike DNC, which leaves it unchanged, just decreases the usage counter, causing problems with lookup")
    parser.add_argument("-sharpness_control", type=bool, default=1, help="Distribution sharpness control for forward and backward links")
    parser.add_argument("-think_steps", type=int, default=0, help="Iddle steps before requiring the answer (for bAbi)")
    parser.add_argument("-dump_profile", type=str, save=False)
    parser.add_argument("-test_on_start", default="0", save=False)
    parser.add_argument("-dump_heatmaps", default=False, save=False)
    parser.add_argument("-test_batch_size", default=16)
    parser.add_argument("-mask_min", default=0.0)
    parser.add_argument("-load", type=str, save=False)
    parser.add_argument("-dataset_path", type=str, default="none", parser=ArgumentParser.str_or_none(), help="Specify babi path manually")
    parser.add_argument("-babi_train_tasks", type=str, default="none", parser=ArgumentParser.list_or_none(type=str), help="babi task list to use for training")
    parser.add_argument("-babi_test_tasks", type=str, default="none", parser=ArgumentParser.list_or_none(type=str), help="babi task list to use for testing")
    parser.add_argument("-babi_train_sets", type=str, default="train", parser=ArgumentParser.list_or_none(type=str), help="babi train sets to use")
    parser.add_argument("-babi_test_sets", type=str, default="test", parser=ArgumentParser.list_or_none(type=str), help="babi test sets to use")
    parser.add_argument("-noargsave", type=bool, default=False, help="Do not save modified arguments", save=False)
    parser.add_argument("-demo", type=bool, default=False, help="Do a single step with fixed seed", save=False)
    parser.add_argument("-exit_after", type=int, help="Exit after this amount of steps. Useful for debugging.", save=False)
    parser.add_argument("-grad_clip", type=float, default=10.0, help="Max gradient norm")
    parser.add_argument("-clip_controller", type=float, default=20.0, help="Max gradient norm")
    parser.add_argument("-print_test", default=False, save=False)

    parser.add_profile([
        ArgumentParser.Profile("babi", {
            "preview_interval": 10,
            "save_interval": 500,
            "task": "babi",
            "mem_count": 256,
            "data_word_size": 64,
            "n_read_heads": 4,
            "layer_sizes": "256",
            "controller_type": "lstm",
            "lstm_use_all_outputs": True,
            "momentum": 0.9,
            "embedding_size": 128,
            "test_interval": 5000,
            "think_steps": 3,
            "batch_size": 2
        }, include=["dnc-msd"]),

        ArgumentParser.Profile("repeat_copy", {
            "bit_w": 8,
            "repeat": "1-8",
            "len": "2-14",
            "task": "copy",
            "think_steps": 1,
            "preview_interval": 10,
            "info_interval": 10,
            "save_interval": 100,
            "data_word_size": 16,
            "layer_sizes": "32",
            "n_subbatch": 1,
            "controller_type": "lstm",
        }),

        ArgumentParser.Profile("repeat_copy_simple", {
            "repeat": "1-3",
        }, include="repeat_copy"),

        ArgumentParser.Profile("dnc", {
            "masked_lookup": False,
            "sharpness_control": False,
            "dealloc_content": False
        }),

        ArgumentParser.Profile("dnc-m", {
            "masked_lookup": True,
            "sharpness_control": False,
            "dealloc_content": False
        }),

        ArgumentParser.Profile("dnc-s", {
            "masked_lookup": False,
            "sharpness_control": True,
            "dealloc_content": False
        }),

        ArgumentParser.Profile("dnc-d", {
            "masked_lookup": False,
            "sharpness_control": False,
            "dealloc_content": True
        }),

        ArgumentParser.Profile("dnc-md", {
            "masked_lookup": True,
            "sharpness_control": False,
            "dealloc_content": True
        }),

        ArgumentParser.Profile("dnc-ms", {
            "masked_lookup": True,
            "sharpness_control": True,
            "dealloc_content": False
        }),

        ArgumentParser.Profile("dnc-sd", {
            "masked_lookup": False,
            "sharpness_control": True,
            "dealloc_content": True
        }),

        ArgumentParser.Profile("dnc-msd", {
            "masked_lookup": True,
            "sharpness_control": True,
            "dealloc_content": True
        }),

        ArgumentParser.Profile("keyvalue", {
            "repeat": "1",
            "len": "2-16",
            "mem_count": 16,
            "task": "keyvalue",
            "think_steps": 1,
            "preview_interval": 10,
            "info_interval": 10,
            "data_word_size": 32,
            "bit_w": 12,
            "save_interval": 1000,
            "layer_sizes": "32"
        }),

        ArgumentParser.Profile("keyvalue2way", {
            "task": "keyvalue2way",
        }, include="keyvalue"),

        ArgumentParser.Profile("associative_recall",{
            "task": "recall",
            "bit_w": 8,
            "len": "2-16",
            "mem_count": 64,
            "data_word_size": 32,
            "n_read_heads": 1,
            "layer_sizes": "128",
            "controller_type": "lstm",
            "lstm_use_all_outputs": 1,
            "think_steps": 1,
            "mask_min": 0.1,
            "info_interval": 10,
            "save_interval": 1000,
            "preview_interval": 10,
            "n_subbatch": 1,
        })
    ])

    opt = parser.parse()
    assert opt.name is not None, "Training dir (-name parameter) not given"
    opt = parser.sync(os.path.join(opt.name, "args.json"), save=not opt.noargsave)

    if opt.demo:
        Seed.fix()

    os.makedirs(os.path.join(opt.name,"save"), exist_ok=True)
    os.makedirs(os.path.join(opt.name,"preview"), exist_ok=True)

    gpu_allocator.use_gpu(opt.gpu)

    debug.enableDebug = opt.debug_log

    if opt.visport>0:
        Visdom.start(opt.visport)

    Visdom.Text("Name").set(opt.name)

    class LengthHackSampler:
        def __init__(self, batch_size, length):
            self.length = length
            self.batch_size = batch_size

        def __iter__(self):
            while True:
                len = self.length() if callable(self.length) else self.length
                yield [len] * self.batch_size

        def __len__(self):
            return 0x7FFFFFFF

    embedding = None
    test_set = None
    curriculum = None
    loader_reset = False
    if opt.task=="copy":
        dataset = CopyData(bit_w=opt.bit_w)
        in_size = opt.bit_w + 1
        out_size = in_size
    elif opt.task=="recall":
        dataset = AssociativeRecall(bit_w=opt.bit_w, block_w=opt.block_w)
        in_size = opt.bit_w + 2
        out_size = in_size
    elif opt.task=="keyvalue":
        assert opt.bit_w % 2==0, "Key-value datasets works only with even bit_w"
        dataset = KeyValue(bit_w=opt.bit_w)
        in_size = opt.bit_w + 1
        out_size = opt.bit_w//2
    elif opt.task=="keyvalue2way":
        assert opt.bit_w % 2==0, "Key-value datasets works only with even bit_w"
        dataset = KeyValue2Way(bit_w=opt.bit_w)
        in_size = opt.bit_w + 2
        out_size = opt.bit_w//2
    elif opt.task=="babi":
        dataset = bAbiDataset(think_steps=opt.think_steps, dir_name=opt.dataset_path)
        test_set = bAbiDataset(think_steps=opt.think_steps, dir_name=opt.dataset_path, name="test")
        dataset.use(opt.babi_train_tasks, opt.babi_train_sets)
        in_size = opt.embedding_size
        print("bAbi: loaded total of %d sequences." % len(dataset))
        test_set.use(opt.babi_test_tasks, opt.babi_test_sets)
        out_size = len(dataset.vocabulary)
        print("bAbi: using %d sequences for training, %d for testing" % (len(dataset), len(test_set)))
    else:
        assert False, "Invalid task: %s" % opt.task

    if opt.task in ["babi"]:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=4, pin_memory=True, shuffle=True, collate_fn=MetaCollate())
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batch_size, num_workers=opt.test_batch_size, pin_memory=True, shuffle=False, collate_fn=MetaCollate()) if test_set is not None else None
    else:
        dataset = BitmapTaskRepeater(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=LengthHackSampler(opt.batch_size, BitmapTaskRepeater.key_sampler(opt.len, opt.repeat)), num_workers=1, pin_memory=True)

    if opt.controller_type == "lstm":
        controller_constructor = functools.partial(LSTMController, out_from_all_layers=opt.lstm_use_all_outputs)
    elif opt.controller_type == "linear":
        controller_constructor = FeedforwardController
    else:
        assert False, "Invalid controller: %s" % opt.controller_type

    model = DNC(in_size, out_size, opt.data_word_size, opt.mem_count, opt.n_read_heads, controller_constructor(opt.layer_sizes),
                batch_first=True, mask=opt.masked_lookup, dealloc_content=opt.dealloc_content,
                link_sharpness_control=opt.sharpness_control,
                mask_min=opt.mask_min, clip_controller=opt.clip_controller)

    params = [
        {'params': [p for n, p in model.named_parameters() if not n.endswith(".bias")]},
        {'params': [p for n, p in model.named_parameters() if n.endswith(".bias")], 'weight_decay': 0}
    ]

    device = torch.device('cuda') if opt.gpu!="none" else torch.device("cpu")
    print("DEVICE: ", device)

    if isinstance(dataset, NLPTask):
        embedding = torch.nn.Embedding(len(dataset.vocabulary), opt.embedding_size).to(device)
        params.append({'params': embedding.parameters(), 'weight_decay': 0})

    if opt.optimizer=="sgd":
        optimizer = torch.optim.SGD(params, lr=opt.lr, weight_decay=opt.wd, momentum=opt.momentum)
    elif opt.optimizer=="adam":
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.wd)
    elif opt.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=opt.lr, weight_decay=opt.wd, momentum=opt.momentum, eps=1e-10)
    else:
        assert "Invalid optimizer: %s" % opt.optimizer

    n_params = sum([sum([t.numel() for t in d['params']]) for d in params])
    print("Number of parameters: %d" % n_params)

    model = model.to(device)
    if embedding is not None and hasattr(embedding, "to"):
        embedding = embedding.to(device)

    i=0
    loss_sum = 0

    loss_plot = Visdom.Plot2D("loss", store_interval=opt.info_interval, xlabel="iterations", ylabel="loss")

    if curriculum is not None:
        curriculum_plot = Visdom.Plot2D("curriculum lesson" +
                                    (" (last %d)" % (curriculum.n_lessons-1) if curriculum.n_lessons is not None else ""),
                                    xlabel="iterations", ylabel="lesson")
        curriculum_accuracy = Visdom.Plot2D("curriculum accuracy", xlabel="iterations", ylabel="accuracy")

    saver = Saver(os.path.join(opt.name, "save"), short_interval=opt.save_interval)
    saver.register("model", StateSaver(model))
    saver.register("optimizer", StateSaver(optimizer))
    saver.register("i", GlobalVarSaver("i"))
    saver.register("loss_sum", GlobalVarSaver("loss_sum"))
    saver.register("loss_plot", StateSaver(loss_plot))
    saver.register("dataset", StateSaver(dataset))
    if test_set:
        saver.register("test_set", StateSaver(test_set))

    if curriculum is not None:
        saver.register("curriculum", StateSaver(curriculum))
        saver.register("curriculum_plot", StateSaver(curriculum_plot))
        saver.register("curriculum_accuracy", StateSaver(curriculum_accuracy))

    if isinstance(dataset, NLPTask):
        saver.register("word_embeddings", StateSaver(embedding))
    elif embedding is not None:
        saver.register("embeddings", StateSaver(embedding))

    if not saver.load(opt.load):
        model.reset_parameters()
        if embedding is not None:
            embedding.reset_parameters()

    visualizers = {}

    debug_schemas={
        "read_head" : {
            "list_dim" : 2
        },
        "temporal_links/forward_dists" : {
            "list_dim" : 2
        },
        "temporal_links/backward_dists" : {
            "list_dim" : 2
        }
    }

    def plot_debug(debug, prefix="", schema={}):
        if debug is None:
            return

        for k, v in debug.items():
            curr_name = prefix+k
            if curr_name in debug_schemas:
                curr_schema = schema.copy()
                curr_schema.update(debug_schemas[curr_name])
            else:
                curr_schema = schema

            if isinstance(v, dict):
                plot_debug(v, curr_name+"/", curr_schema)
                continue

            data = v[0]

            if curr_schema.get("list_dim",-1) > 0:
                if data.ndim != 3:
                    print("WARNING: unknown data shape for array display: %s, tensor %s" % (data.shape, curr_name))
                    continue

                n_steps = data.shape[curr_schema["list_dim"]-1]
                if curr_name not in visualizers:
                    visualizers[curr_name] = [Visdom.Heatmap(curr_name+"_%d" % i, dumpdir=os.path.join(opt.name, "preview") if opt.dump_heatmaps else None) for i in range(n_steps)]

                for i in range(n_steps):
                    visualizers[curr_name][i].draw(index_by_dim(data, curr_schema["list_dim"]-1, i))
            else:
                if data.ndim != 2:
                    print("WARNING: unknown data shape for simple display: %s, tensor %s" % (data.shape, curr_name))
                    continue

                if curr_name not in visualizers:
                    visualizers[curr_name] = Visdom.Heatmap(curr_name, dumpdir=os.path.join(opt.name, "preview") if opt.dump_heatmaps else None)

                visualizers[curr_name].draw(data)


    def run_model(input, debug=None):
        if isinstance(dataset, NLPTask):
            input = embedding(input["input"])
        else:
            input = input["input"] * 2.0 - 1.0

        return model(input, debug=debug)

    def multiply_grads(params, mul):
        if mul==1:
            return

        for pa in params:
            for p in pa["params"]:
                p.grad.data *= mul

    def test():
        if test_set is None:
            return

        print("TESTING...")
        start_time=time.time()
        t = test_set.start_test()
        with torch.no_grad():
            for data in tqdm(test_loader):
                data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
                if hasattr(dataset, "prepare"):
                    data = dataset.prepare(data)

                net_out = run_model(data)
                test_set.veify_result(t, data, net_out)

        test_set.show_test_results(i, t)
        print("Test done in %gs" % (time.time() - start_time))

    if opt.test_on_start.lower() in ["on", "1", "true", "quit"]:
        test()
        if opt.test_on_start.lower() == "quit":
            saver.write(i)
            sys.exit(-1)

    if opt.print_test:
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in tqdm(test_loader):
                if not running:
                    return

                data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
                if hasattr(test_set, "prepare"):
                    data = test_set.prepare(data)

                net_out = run_model(data)

                c,t = test_set.curriculum_measure(net_out, data["output"])
                total += t
                correct += c

        print("Test result: %2.f%% (%d out of %d correct)" % (100.0*correct/total, correct, total))
        model.train()
        return

    iter_start_time = time.time() if i % opt.info_interval == 0 else None
    data_load_total_time = 0

    start_i = i

    if opt.dump_profile:
        profiler = torch.autograd.profiler.profile(use_cuda=True)


    if opt.dump_heatmaps:
        dataset.set_dump_dir(os.path.join(opt.name, "preview"))

    @preview()
    def do_visualize(raw_data, output, pos_map, debug):
        if pos_map is not None:
            output = embedding.backmap_output(output, pos_map, raw_data["output"].shape[1])
        dataset.visualize_preview(raw_data, output)

        if debug is not None:
            plot_debug(debug)

    preview_timer=OnceEvery(opt.preview_interval)

    pos_map = None
    start_iter = i

    if curriculum is not None:
        curriculum.init()

    while running:
        data_load_timer = time.time()
        for data in data_loader:
            if not running:
                break

            if loader_reset:
                print("Loader reset requested. Resetting...")
                loader_reset = False
                if curriculum is not None:
                    curriculum.lesson_started()
                break

            if opt.dump_profile:
                if i==start_i+1:
                    print("Starting profiler")
                    profiler.__enter__()
                elif i==start_i+5+1:
                    print("Stopping profiler")
                    profiler.__exit__(None, None, None)
                    print("Average stats")
                    print(profiler.key_averages().table("cpu_time_total"))
                    print("Writing trace to file")
                    profiler.export_chrome_trace(opt.dump_profile)
                    print("Done.")
                    sys.exit(0)
                else:
                    print("Step %d out of 5" % (i-start_i))

            debug.dbg_print("-------------------------------------")
            raw_data = data

            data = {k: v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
            if hasattr(dataset, "prepare"):
                data = dataset.prepare(data)

            data_load_total_time += time.time() - data_load_timer

            need_preview = preview_timer()
            debug_data = {} if opt.debug and need_preview else None


            optimizer.zero_grad()

            if opt.n_subbatch=="auto":
                n_subbatch = math.ceil(data["input"].numel() / opt.max_input_count_per_batch)
            else:
                n_subbatch = int(opt.n_subbatch)

            real_batch = max(math.floor(opt.batch_size/n_subbatch),1)
            n_subbatch = math.ceil(opt.batch_size/real_batch)
            remaning_batch = opt.batch_size % real_batch

            for subbatch in range(n_subbatch):
                if not running:
                    break
                input = data["input"]
                target = data["output"]

                if n_subbatch!=1:
                    input = input[subbatch * real_batch: (subbatch + 1) * real_batch]
                    target = target[subbatch * real_batch:(subbatch + 1) * real_batch]

                f2 = data.copy()
                f2["input"] = input
                output = run_model(f2, debug=debug_data if subbatch==n_subbatch-1 else None)
                l = dataset.loss(output, target)
                debug.nan_check(l, force=True)
                l.backward()

                if curriculum is not None:
                    curriculum.update(*dataset.curriculum_measure(output, target))

                if remaning_batch!=0 and subbatch == n_subbatch-2:
                    multiply_grads(params, real_batch/remaning_batch)

            if n_subbatch!=1:
                if remaning_batch==0:
                    multiply_grads(params, 1/n_subbatch)
                else:
                    multiply_grads(params, remaning_batch / opt.batch_size)

            for p in params:
                torch.nn.utils.clip_grad_norm_(p["params"], opt.grad_clip)

            optimizer.step()

            i += 1

            curr_loss = l.data.item()
            loss_plot.add_point(i, curr_loss)

            loss_sum += curr_loss


            if i % opt.info_interval == 0:
                tim = time.time()
                loss_avg = loss_sum / opt.info_interval

                if curriculum is not None:
                    curriculum_accuracy.add_point(i, curriculum.get_accuracy())
                    curriculum_plot.add_point(i, curriculum.step)

                message = "Iteration %d, loss: %.4f" % (i, loss_avg)
                if iter_start_time is not None:
                    message += " (%.2f ms/iter, load time %.2g ms/iter, visport: %s)" % (
                                (tim - iter_start_time) / opt.info_interval * 1000.0,
                                data_load_total_time / opt.info_interval * 1000.0,
                                Visdom.port)
                print(message)
                iter_start_time = tim
                loss_sum = 0
                data_load_total_time = 0

            debug.dbg_print("Iteration %d, loss %g" % (i, curr_loss))

            if need_preview:
                do_visualize(raw_data, output, pos_map, debug_data)

            if i % opt.test_interval==0:
                test()

            saver.tick(i)

            if opt.demo and opt.exit_after is None:
                running = False
                input("Press enter to quit.")

            if opt.exit_after is not None and (i-start_iter)>=opt.exit_after:
                running=False

            data_load_timer = time.time()


if __name__ == "__main__":
    global running
    running = True


    def signal_handler(signal, frame):
        global running
        print('You pressed Ctrl+C!')
        running = False


    signal.signal(signal.SIGINT, signal_handler)

    main()
