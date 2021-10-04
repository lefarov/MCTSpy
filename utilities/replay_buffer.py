import typing as t
from collections import deque

# import numpy as np
from heapq import heappush, heappop


class ReplayBufferList:

    def __init__(self, max_size: int):
        # obs_shape: t.Tuple[int, ...], action_shape: t.Tuple[int, ...],
        # reward_shape: t.Tuple[int, ...]
        # self.obs_data = np.empty((max_size, *obs_shape), dtype=np.float32)
        # self.action_data = np.empty((max_size, *action_shape), dtype=np.float32)

        self.max_size = max_size
        self.data = [None] * max_size

        self.indices = []
        self.reverse_indices = [(None, None)] * max_size  # todo Explain.

    def push_back(self, data_to_insert: t.List[t.Any]):

        assert len(data_to_insert) <= self.max_size

        # Special case: empty buffer.
        if not self.indices:
            self.write_data_at_loc(data_to_insert, 0)
            return

        tail = self.indices[-1][1]
        head = self.indices[0][0]

        if tail > head:
            if self.max_size - tail >= len(data_to_insert):
                self.write_data_at_loc(data_to_insert, tail)
            else:
                tail = 0
                while head - tail < len(data_to_insert) and self.indices:
                    _, head = heappop(self.indices)

                self.write_data_at_loc(data_to_insert, tail)
        else:
            while head - tail < len(data_to_insert) and self.indices:
                _, head = heappop(self.indices)

            self.write_data_at_loc(data_to_insert, head)

    def write_data_at_loc(self, data_to_insert, index: int):
        index_start = index
        for item in data_to_insert:
            self.data[index] = item
            self.reverse_indices[index] = (index_start, index_start + len(data_to_insert))
            index += 1

        heappush(self.indices, (index_start, index))



