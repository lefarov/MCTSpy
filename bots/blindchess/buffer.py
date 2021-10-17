import random
import typing as t
import numpy as np

from collections import deque

from bots.blindchess.agent import Transition


class HistoryReplayBuffer:

    def __init__(
        self,
        size: int,
        obs_shape: t.Tuple,
        act_shape: t.Tuple,
        act_mask_shape: t.Tuple,
        obs_dtype: t.Type=np.float32,
        act_dtype: t.Type=np.int32,
    ) -> None:

        self.size = size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.act_shape = act_shape
        self.act_dtype = act_dtype
        self.act_mask_shape = act_mask_shape

        self.obs_data = np.empty((size, *obs_shape), dtype=obs_dtype)
        self.act_data = np.empty((size, *act_shape), dtype=act_dtype)
        self.rew_data = np.empty((size,), dtype=np.float32)
        self.done_data = np.empty((size,), dtype=np.float32)
        
        # Data for the opponents' moves
        self.act_opponent_data = np.empty((size, *act_shape), dtype=act_dtype)
        # Data for available moves
        self.act_next_mask_data = np.empty((size, *act_mask_shape), dtype=np.float32)

        self.history_indices = deque()

        self.is_full = False

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
                self.is_full = True
                last_entry = 0

            # If indices deque is not empty, set the next entry pointer to the start
            # of the next history record. If next history record starts at zero, it
            # means that there are no more records till the end of the buffer.
            if self.history_indices:
                next_entry = self.history_indices[0][0] or self.size
        
        # Write history after the last entry in the buffer and rotate the indices deque.
        self.write_history_at_location(history, last_entry)

    def write_history_at_location(self, history: Transition, loc: int) -> None:
        length = len(history.reward)
        
        self.obs_data[loc:loc + length] = history.observation
        self.act_data[loc:loc + length] = history.action
        self.rew_data[loc:loc + length] = history.reward
        self.done_data[loc:loc + length] = history.done

        self.act_opponent_data[loc:loc + length] = history.action_opponent
        self.act_next_mask_data[loc:loc + length] = history.action_mask

        self.history_indices.appendleft((loc, loc + length))
        self.history_indices.rotate(-1)

    def _allocate_batches(self, batch_size, slice_length, num_action_types: int=2):
        # Observation batch: [N, B, L, *O], where N is the number of different action types
        # i.e. move and sense for Recon Chess, B is the batch size, L is the sub-history length
        # and O is the dimensionality of the observation.
        obs_batch_shape = (num_action_types, batch_size, slice_length, *self.obs_shape)
        obs_batch = np.empty(obs_batch_shape, dtype=self.obs_dtype)
        obs_next_batch = np.empty(obs_batch_shape, dtype=self.obs_dtype)

        # Action batch: [N, B, *A].
        act_batch_shape = (num_action_types, batch_size, *self.act_shape)
        act_batch = np.empty(act_batch_shape, dtype=self.act_dtype)
        act_opponent_batch = np.empty(act_batch_shape, dtype=self.act_dtype)

        # Mack batch for the next action: [N, B, *M], where M is the shape of action mask, for
        # Recon Chess it's 64 * 64 float mask with 1. at valid move indices.
        act_mask_next_batch = np.empty(
            (num_action_types, batch_size, *self.act_mask_shape),
            dtype=np.float32
        )

        # Reward and Done batches: [N, B,]
        rew_batch = np.empty((num_action_types, batch_size,), dtype=np.float32)
        done_batch = np.empty((num_action_types, batch_size,), dtype=np.float32)

        return (
            obs_batch,
            act_batch,
            rew_batch,
            done_batch,
            obs_next_batch,
            act_mask_next_batch,
            act_opponent_batch,
        )

    def sample_batch(self, batch_size, slice_length, action_type_slices: list=None):
        """ Sample the batch of data for the given type of action.
        
        TODO: should we always use numpy to control the random seed?
        """
        if action_type_slices is None:
            action_type_slices = [slice(None, None)]

        # Allocate empty batches
        (
            obs_batch,
            act_batch,
            rew_batch,
            done_batch,
            obs_next_batch,
            act_next_mask_batch,
            act_opponent_batch,
        ) = self._allocate_batches(batch_size, slice_length, len(action_type_slices))

        # Iterate over action types
        for action_i, action_type_slice in enumerate(action_type_slices):

            # Sample `batch_size` of history indices proportional to their length
            history_weights = [h[1] - h[0] for h in self.history_indices]
            history_samples = random.choices(
                self.history_indices, history_weights, k=batch_size
            )

            for i, history in enumerate(history_samples):
                indices = list(range(history[0], history[1] - 1))
                indices = indices[action_type_slice]

                # Sample index uniformly
                index = random.choice(indices)

                # Build a sequence
                obs_slice = self.index_to_obs_sample(index, history[0], slice_length)
                obs_next_slice = self.index_to_obs_sample(index + 1, history[0], slice_length)

                # Write data to batches
                obs_batch[action_i, i, ...] = obs_slice
                act_batch[action_i, i, ...] = self.act_data[index]
                rew_batch[action_i, i, ...] = self.rew_data[index]
                done_batch[action_i, i, ...] = self.done_data[index]
                obs_next_batch[action_i, i, ...] = obs_next_slice
                act_next_mask_batch[action_i, i, ...] = self.act_next_mask_data[index]
                act_opponent_batch[action_i, i, ...] = self.act_opponent_data[index]

        return (
            obs_batch,
            act_batch,
            rew_batch,
            done_batch,
            obs_next_batch,
            act_next_mask_batch,
            act_opponent_batch,
        )

    def index_to_obs_sample(self, index, history_start, target_length):
        # Get the slice from the data storage
        obs_slice_end = index + 1
        obs_slice_start = max(obs_slice_end - target_length, history_start)
        obs_slice = self.obs_data[obs_slice_start:obs_slice_end]

        # Pad the time axis if needed, do not pad any other axis.
        pads = [
            (target_length - len(obs_slice), 0) if axis == 0
            else (0, 0) for axis in range(obs_slice.ndim)
        ]

        return np.pad(obs_slice, pads, "edge")
