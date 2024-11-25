import numpy as np
import glob
import matplotlib.pyplot as plt

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


pattern = "/workdir/all_data/train/**/**_erased.npy"

files = sorted(glob.glob(pattern, recursive=True))

file = np.load(files[0])

res_trajs = []
for traj in file:
    now = decode(traj.tolist())
    res_trajs.append(now)

def visual(res_trajs):
    from PIL import Image

    subway_img = Image.open("./back.png")
    x_range = [116.2075, 116.5575]
    y_range = [39.7523, 40.1023]

    fig, axs = plt.subplots(16, 14, figsize=(64, 56)) # 16 14
    for i in range(16):
        now = decode(res_trajs[i].tolist())
        for j in range(14):
            axs[i, j].imshow(subway_img, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                             aspect='auto', alpha=0.3)
            axs[i, j].set_xlim(x_range)
            axs[i, j].set_ylim(y_range)
            axs[i, j].plot(now[j * 96:(j + 1) * 96, 0], now[j * 96:(j + 1) * 96, 1], color='blue',
                           alpha=0.7)

    plt.tight_layout()

    savefile = "test.png"
    plt.savefig(savefile)
    plt.close()
