import typing as t
import numpy as np

from collections import deque
from heapq import heappush, heappop

from agents.blind_chess import Transition


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

        #TODO: not gonna work (it doesn't select the latest added element)
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


class HistoryRaplayBuffer:

    def __init__(
        self,
        size: int,
        obs_shape: t.Tuple,
        act_shape: t.Tuple,
        obs_dtype: t.Type=np.float32,
        act_dtype: t.Type=np.int32,
    ) -> None:

        self.size = size

        self.obs_data = np.empty((size, *obs_shape), dtype=obs_dtype)
        self.act_data = np.empty((size, *act_shape), dtype=act_dtype)
        self.rew_data = np.empty((size,), dtype=np.float32)

        self.history_indices = deque()

    def add(self, history: Transition) -> None:
        length = len(history.reward)
        assert length <= self.size

        # If buffer is empty, write the history starting at index 0.
        if not self.history_indices:
            self.write_history_at_location(history, 0)
            return

        last_entry = self.history_indices[-1][-1]
        next_entry = self.history_indices[0][0] or self.size

        # Pop history records from the buffer until we can fit current history
        # between the end of the last entry and the beginning of the next one.
        while self.history_indices and next_entry - last_entry < length:
            start, _ = self.history_indices.popleft()
            
            # If we pop history record that starts at 0, it means that we don't have
            # enough free space till the end of the buffer. Thus we need to write
            # current history from the beginning of the buffer till the next record.
            if start == 0:
                last_entry = 0

            # If indices deque is not empty, set the next entry pointer to the start
            # of the next histroy record. If next history record starts at zero, it
            # means that there're no more records till the end of the buffer.
            if self.history_indices:
                next_entry = self.history_indices[0][0] or self.size
        
        # Write history after the last entry in the buffer and rottate the indices deque.
        self.write_history_at_location(history, last_entry)

    def write_history_at_location(self, history: Transition, loc: int) -> None:
        length = len(history.reward)
        
        self.obs_data[loc:loc + length] = history.observation
        self.act_data[loc:loc + length] = history.action
        self.rew_data[loc:loc + length] = history.reward

        self.history_indices.appendleft((loc, loc + length))
        self.history_indices.rotate(-1)

    def sample_batch(self, batch_size, history_length):
        # TODO:
        # 1. Sample `batch_size` of history indices proportional to their length
        # 2. Sample one point within each history uniformly
        pass



