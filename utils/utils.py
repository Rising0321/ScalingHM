import torch
import numpy as np
import random
import os

def init_seed(seed):
    seed = seed + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def decode(grids):
    temp = []
    for grid in grids:
        x_range = [116.2075, 116.5575]
        y_range = [39.7523, 40.1023]
        grid_len = 0.002  # 200m x 200m
        y_len = int((y_range[1] - y_range[0]) / grid_len)
        x = grid // y_len
        y = grid % y_len
        temp.append([x * grid_len + x_range[0], y * grid_len + y_range[0]])
    return np.array(temp[1:])

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)