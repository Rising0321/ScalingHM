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
from model.GPT import GPT
import time
from utils.utils import print0, decode, init_seed
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


def process_input(idx, targets, start_day):
    # print(idx.shape)
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
                        help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d1|d10|d100|d300|d500|d1000|d1500")
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
    parser.add_argument("--seed", type=int, default=42, help="gen seed")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 4096
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"d1", "d10", "d100", "d300", "d500", "d1000", "d1500"}

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

    init_seed(args.seed)

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
            "d1000": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=36, n_head=24, n_embd=1536),
            "d1500": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=48, n_head=24, n_embd=1608)
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


    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)


    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        modelfile = os.path.join(args.output_dir, "model.bin")
        if args.test_write_model:
            print0("writing model to disk to test serialization")
            torch.save(raw_model.state_dict(), modelfile)
            model_config = {
                "d1": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=6, n_head=4, n_embd=128),
                "d10": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=6, n_head=8, n_embd=512),
                "d100": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=12, n_head=12, n_embd=768),
                "d300": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=24, n_head=16, n_embd=1024),
                "d500": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=24, n_head=16, n_embd=1280),
                "d1000": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=36, n_head=24, n_embd=1536),
                "d1500": GPTConfig(block_size=1024, vocab_size=VOCUB_SIZE, n_layer=48, n_head=24, n_embd=1608)
            }[args.model]
            model2 = GPT(model_config)
            model2 = model2.to(device)
            model2 = DDP(model2, device_ids=[ddp_local_rank])
            model2.module.load_state_dict(torch.load(modelfile))
            print(model.module.transformer.wte.weight)
            print(model2.module.transformer.wte.weight)
            exit(0)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0  # dummy value to print in inference-only mode
    val_time, loading_time, inference_time = 0.0, 0.0, 0.0
    best_val = 100000
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
                and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    loading_time_st = time.time()
                    x, y = val_loader.next_batch()
                    loading_time += time.time() - loading_time_st
                    inference_time_st = time.time()
                    x, y = x.to(device), y.to(device)
                    start_day = random.randint(0, 6)
                    x, y = process_input(x, y, start_day)
                    _, loss = model(x, y, start_day=start_day, return_logits=False)
                    val_loss += loss.item()
                    inference_time += time.time() - inference_time_st
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(raw_model.state_dict(), modelfile)

        # once in a while perform model inference on the master process
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
                and master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            generate_batch_size = args.gen_batch_size
            start_ids = [SOT] * generate_batch_size
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device).view(generate_batch_size, 1))
            max_new_tokens = 96  # 24 hours
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            print0('---------------')
            for i in range(generate_batch_size):
                print0(f"Generated sequence {i + 1}:\n", decode(yg[i].tolist()))
            print0('---------------')

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0  # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            loading_time_st = time.time()
            x, y = train_loader.next_batch()
            loading_time += time.time() - loading_time_st
            inference_time_st = time.time()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                start_day = random.randint(0, 6)
                x, y = process_input(x, y, start_day)
                _, loss = model(x, y, start_day=start_day, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach()  # keep track of the mean loss
            # backward pass
            if not args.inference_only:
                loss.backward()
            inference_time += time.time() - inference_time_st
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        print0(
            f"step {step + 1:4d}/{args.num_iterations} | tot_time {tot_time} | load_time {loading_time} | train_time {inference_time} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1 - t0) * 1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings) * 1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
