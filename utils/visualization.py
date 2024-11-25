import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.metrices import calc_metrices

from utils.utils import to_gps, renorm


def visual(now_trajs):
    plt.figure(figsize=(8, 8))

    now_trajs = decodeTrajs(now_trajs, input_type, json_file)

    for i in range(batch_size):
        traj = now_trajs[i]
        plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.1)

    plt.tight_layout()
    x_range = json_file["x_range"]
    y_range = json_file["y_range"]
    plt.savefig(f'../visualizations/test.png')
    plt.close()