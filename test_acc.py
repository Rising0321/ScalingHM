"""
Reference code for GPT-2 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:
1 GPU:
python train_gpt2.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16
you can also turn on flash-attention by appending --flash=1
4 GPU:
torchrun --standalone --nproc_per_node=4 train_gpt.py --write_tensors=0  --sequence_length=961 --compile=0 --tensorcores=1 --dtype=bfloat16

"""

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
import torch.distributed as dist

import time

from utils.utils import print0, decode
from data.data import DistributedDataLoader
from model.GPT import GPT

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


def xy_to_grid(x, y):
    y_range = [39.7523, 40.1023]
    grid_len = 0.002  # 200m x 200m
    y_len = int((y_range[1] - y_range[0]) / grid_len)
    return x * y_len + y


def decode_xy(grid):
    x_range = [116.2075, 116.5575]
    y_range = [39.7523, 40.1023]
    grid_len = 0.002  # 200m x 200m
    y_len = int((y_range[1] - y_range[0]) / grid_len)
    x = grid // y_len
    y = grid % y_len
    return int(x), int(y)


VOCUB_SIZE = to_grid([116.5574, 40.1022]) + 96  # VOCUB_SIZE = 30720, 30720 / 1024 = 30
SOT = to_grid([116.5574, 40.1022]) + 1

tot_time = 0


def process_input(idx, targets, start_day):
    idx_ = idx[:, start_day * 96 + 1: (start_day + 8) * 96]
    idx = torch.concat([idx[:, 0:1], idx_], dim=1)
    targets = targets[:, start_day * 96: (start_day + 8) * 96]
    return idx, targets


if __name__ == "__main__":
    import time
    import argparse
    import tiktoken

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
    parser.add_argument("--gen_batch_size", type=int, default=16, help="generate batch size")
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

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    logfile = None
    if args.output_dir:
        modelfile = os.path.join(args.output_dir, "model.bin")
        model.module.load_state_dict(torch.load(modelfile))
    else:
        raise NotImplementedError

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cnt_sample = 0
    acc_sample = 0
    from tqdm import tqdm


    def myabs(now):
        if now >= 0:
            return now
        return -now


    for step in tqdm(range(100)):
        model.eval()
        with torch.no_grad():
            for _ in range(args.val_max_steps):
                loading_time_st = time.time()
                x, y = val_loader.next_batch()
                inference_time_st = time.time()
                x, y = x.to(device), y.to(device)
                start_day = random.randint(0, 6)
                x, y = process_input(x, y, start_day)
                logits, _ = model(x, y, start_day=start_day)

                logits = logits[:, -1, :] / 0.00000001
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                for j in range(len(idx_next)):
                    cnt_sample += 1
                    next_xy = decode_xy(idx_next[j])
                    now_xy = decode_xy(y[j][-1])
                    # print(next_xy, now_xy, idx_next[j], y[j][-1])
                    # if idx_next[j] == y[j][-1]:
                    #     acc_sample += 1
                    if myabs(now_xy[0] - next_xy[0]) <= 1 and myabs(now_xy[1] - next_xy[1]) <= 1:
                        acc_sample += 1

    print(cnt_sample, acc_sample, acc_sample / cnt_sample)
