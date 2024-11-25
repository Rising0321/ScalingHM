import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
from utils.utils import init_seed, decode
from model.GPT import GPT
from utils.utils import print0, decode
from data.data import DistributedDataLoader

# using a global to toggle flash-attention
FLASH = 0


# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    traj_data = np.load(filename)
    return traj_data.shape[0] * traj_data.shape[1]  # for now just return the number of tokens


def to_grid(gps):
    x_range = [116.2075, 116.5575]
    y_range = [39.7523, 40.1023]
    grid_len = 0.002  # 200m x 200m
    if not (x_range[0] <= gps[0] and gps[0] < x_range[1] and y_range[0] <= gps[1] and gps[1] < y_range[1]):
        return -1
    x_len = int((x_range[1] - x_range[0]) / grid_len)
    y_len = int((y_range[1] - y_range[0]) / grid_len)
    x = int((gps[0] - x_range[0]) / grid_len)
    y = int((gps[1] - y_range[0]) / grid_len)
    return x * y_len + y



VOCUB_SIZE = to_grid([116.5574, 40.1022]) + 96  # VOCUB_SIZE = 30720, 30720 / 1024 = 30
SOT = to_grid([116.5574, 40.1022]) + 1

tot_time = 0


import matplotlib.pyplot as plt


def visual(now_trajs_):
    from PIL import Image

    for day in range(3):
        now_trajs = now_trajs_[day * 96:(day + 1) * 96]
        fig, axs = plt.subplots(1, 1, figsize=(36, 36))

        subway_img = Image.open("./back.png")
        x_range = [116.2075, 116.5575]
        y_range = [39.7523, 40.1023]

        axs.imshow(subway_img, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                   aspect='auto', alpha=0.3)
        axs.set_xlim(x_range)
        axs.set_ylim(y_range)

        for i in range(len(now_trajs)):
            traj = now_trajs[i]
            traj = np.array(traj)
            axs.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'./visualization/test-{day}.png')
        plt.close()


if __name__ == "__main__":
    import time
    import argparse

    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="/workdir/all_data/train/**/**_data.npy",
                        help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="/workdir/all_data/eval/**/**_data.npy",
                        help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="",
                        help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="d12",
                        help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d1|d10|d100|d300|d500|d1000")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=1345, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=8 * 4 * 1345,
                        help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=250000, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=3000, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=0.0000001,
                        help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=100, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging interesting!!!!
    parser.add_argument("--overfit_single_batch", type=int, default=0, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # write config
    parser.add_argument("--test_write_model", type=int, default=0, help="write the model to disk")
    parser.add_argument("--load_model", type=int, default=0, help="load the model from disk")
    # gen config
    parser.add_argument("--gen_batch_size", type=int, default=64, help="generate batch size")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 4096
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"d1", "d10", "d100", "d300", "d500", "d1000"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        init_seed(args.seed)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = 0  # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    # print(ddp_world_size, B, T, tokens_per_fwdbwd)
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    if args.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d1": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=6, n_head=4, n_embd=128),
            "d10": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=6, n_head=8, n_embd=512),
            "d100": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=12, n_head=12, n_embd=768),
            "d300": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=24, n_head=16, n_embd=1024),
            "d500": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=24, n_head=16, n_embd=1280),
            "d1000": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=36, n_head=24, n_embd=1536)
        }[args.model]
        model = GPT(model_config)
    else:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(args.model)
    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True  # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        modelfile = os.path.join(args.output_dir, "model.bin")
        model.module.load_state_dict(torch.load(modelfile))
    else:
        raise NotImplementedError

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if master_process:
        from tqdm import tqdm

        # large scale trajectory generation
        res_trajs = []
        for step in tqdm(range(1000 // args.gen_batch_size + 1)):
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            generate_batch_size = args.gen_batch_size
            start_ids = [SOT] * generate_batch_size
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device).view(generate_batch_size, 1))
            max_new_tokens = 96 * 3 + 1  # 24 hours
            temperature = 1
            top_k = 4096
            start_day = 0
            yg = raw_model.generate(xg, max_new_tokens, start_day=start_day, temperature=temperature, top_k=top_k)
            print0('---------------')
            for i in range(generate_batch_size):
                now = decode(yg[i].tolist())
                print0(f"Generated sequence {i + 1}:\n", now)
                res_trajs.append(now)
            print0('---------------')
        visual(res_trajs)

        # individual trajectory generation
        # begin from 7 day, each generate 8 trajectories
        # for start_day in tqdm(range(7)):
        #     model.eval()
        #     generate_batch_size = 8
        #     fig, axs = plt.subplots(generate_batch_size, 7, figsize=(32, 28))
        #     start_ids = [SOT] * generate_batch_size
        #     xg = (torch.tensor(start_ids, dtype=torch.long, device=device).view(generate_batch_size, 1))
        #     max_new_tokens = 96 * 7  # 24 hours
        #     temperature = 1
        #     top_k = 4096
        #     yg = raw_model.generate(xg, max_new_tokens, start_day=start_day, temperature=temperature, top_k=top_k)
        #     for i in range(generate_batch_size):
        #         now = decode(yg[i].tolist())
        #         for j in range(7):
        #             axs[i, j].plot(now[j * 96:(j + 1) * 96, 0], now[j * 96:(j + 1) * 96, 1], color='blue',
        #                            alpha=0.1)
        #             # set title
        #             now_day = (start_day + j) % 7 + 1
        #             axs[i, j].set_title(f"Day {now_day}")
        #     plt.tight_layout()
        #     savefile = os.path.join(args.output_dir, f"individual_trajectory-{start_day + 1}.png")
        #     plt.savefig(savefile)
        #     plt.close()
