import numpy as np
import glob

pattern = "/workdir/all_data/train/**/**_grid.npy"

files = sorted(glob.glob(pattern, recursive=True))

from tqdm import tqdm

for file in tqdm(files):
    tokens = []
    traj_data = np.load(file).reshape([-1, 96 * 15 + 1])
    for traj in traj_data:
        last = 0
        now_grid = None
        flag = 1
        for grid in traj:
            if now_grid == None:
                now_grid = grid
                continue
            if grid == now_grid:
                last += 1
                if last >= 96:
                    flag = 0
                    break
            else:
                last = 0
                now_grid = grid
        if flag == 1:
            tokens.append(traj)
    tokens = np.array(tokens)
    print(tokens.shape)
    np.save(file.replace("grid.npy", "erased.npy"), tokens)
