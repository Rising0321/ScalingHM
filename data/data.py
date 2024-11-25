import glob

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    traj_data = np.load(filename)
    return traj_data.shape[0] * traj_data.shape[1]  # for now just return the number of tokens


# def _load_data_shard_pre(filename):
#     tokens = []
#     traj_data = np.load(filename)
#     global tot_time
#     tot_time_st = time.time()
#     for traj in traj_data:
#         now_traj = [SOT]
#         flag = 1
#         for gps in traj:
#             grid = to_grid(gps)
#             if grid == -1:
#                 flag = 0
#                 break
#             now_traj.append(grid)
#         if flag == 1:
#             tokens.extend(now_traj)
#     tot_time += time.time() - tot_time_st
#     return np.array(tokens)


def _load_data_shard(filename):
    tokens = []
    traj_data = np.load(filename.replace("data.npy", "erased.npy"))
    return np.array(traj_data)


# class DistributedDataLoaderOld:
#     def __init__(self, filename_pattern, B, T, process_rank, num_processes):
#         self.process_rank = process_rank
#         self.num_processes = num_processes
#         self.B = B
#         self.T = T
#
#         # glob files that match the pattern
#         self.files = sorted(glob.glob(filename_pattern, recursive=True))
#
#         assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
#
#         # load and validate all data shards, count number of tokens in total
#         ntok_total = 0
#         # print("calculating number tokens")
#         # for fname in tqdm(self.files):
#         #     shard_ntok = _peek_data_shard(fname)
#         #     assert shard_ntok >= num_processes * B * T + 1
#         #     ntok_total += shard_ntok
#         # self.ntok_total = ntok_total
#         # print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")
#
#         # kick things off
#         self.current_shard = None
#         self.reset()
#
#     def reset(self):
#         # we're being a bit clever here: if we already had shard 0 loaded,
#         # then don't do the work to reload it, just reset the pointer
#         if self.current_shard != 0:
#             self.current_shard = 0
#             self.tokens = _load_data_shard(self.files[self.current_shard])
#         self.current_position = self.process_rank * self.B * self.T
#         global tot_time
#         tot_time = 0
#
#     def advance(self):  # advance to next data shard
#         self.current_shard = (self.current_shard + 1) % len(self.files)
#         self.current_position = self.process_rank * self.B * self.T
#         self.tokens = _load_data_shard(self.files[self.current_shard])
#
#     def next_batch(self):
#         B = self.B
#         T = self.T
#         buf = self.tokens[self.current_position: self.current_position + B * T + 1]
#         buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
#         x = (buf[:-1]).view(B, T)  # inputs
#         y = (buf[1:]).view(B, T)  # targets
#         # advance the start pointer in current shard
#         self.current_position += B * T * self.num_processes
#         # if loading the next batch would be out of bounds advance the shard
#
#         if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
#             self.advance()
#
#         return x, y


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern, recursive=True))

        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        # print("calculating number tokens")
        # for fname in tqdm(self.files):
        #     shard_ntok = _peek_data_shard(fname)
        #     assert shard_ntok >= num_processes * B * T + 1
        #     ntok_total += shard_ntok
        # self.ntok_total = ntok_total
        # print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B
        global tot_time
        tot_time = 0

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        buf = self.tokens[self.current_position: self.current_position + B]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)  # todo: if change to gps this change
        x = buf[:, :-1]  # inputs
        y = buf[:, 1:]  # targets
        # advance the start pointer in current shard
        self.current_position += B * self.num_processes

        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * self.num_processes) > len(self.tokens):
            self.advance()

        return x, y
