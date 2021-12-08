import functools
import operator

import numpy as np
import torch
import typing as t

from bots.tictac import TicTacToe, Board
from bots.tictac.data_structs import DataPoint, DataTensors


class TicTacQNet(torch.nn.Module):

    ObsShape = (TicTacToe.BoardSize, TicTacToe.BoardSize, 1)

    def __init__(self, narx_memory_length, n_hidden, channels_per_layer: t.Optional[t.List[int]] = None):
        super().__init__()

        self.narx_memory_length = narx_memory_length
        self.n_hidden = n_hidden
        self.channels_per_layer = channels_per_layer or [64, 128, 256]
        in_channels = 1

        # Board convolution backbone:
        # 3D convolution layer is applied to a tensor with shape (N,C​,D​,H​,W​)
        # where N - batch size, C (channels) - one-hot-encoding of a piece,
        # D (depth) - history length, H and W are board dimensions (i.e. 8x8).

        # self.conv_stack = torch.nn.Sequential(
        #     torch.nn.Conv3d(
        #         in_channels=in_channels,
        #         out_channels=n_hidden,
        #         kernel_size=(1, 3, 3),
        #         stride=(1, 1, 1)
        #     ),
        #     torch.nn.ReLU(),
        # )
        #
        # dummy_input = torch.zeros((1, in_channels, self.narx_memory_length, Board.Size, Board.Size))
        # fc_input_size = functools.reduce(operator.mul, self.conv_stack(dummy_input).shape)

        fc_input_size = in_channels * self.narx_memory_length * Board.Size * Board.Size

        self.fc_stack = torch.nn.Sequential(
            torch.nn.Linear(fc_input_size, n_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.GELU(),
        )

        # Player heads
        self.fc_sense_q = torch.nn.Linear(self.n_hidden, TicTacToe.BoardSize ** 2)
        self.fc_move_q = torch.nn.Linear(self.n_hidden, TicTacToe.BoardSize ** 2)

    def forward(self, board_memory: torch.Tensor):

        # Compute backbone
        board_encoding = self.backbone(board_memory)

        # Compute heads
        sense_q = self.fc_sense_q(board_encoding)
        move_q = self.fc_move_q(board_encoding)

        return sense_q, move_q

    def backbone(self, board_memory: torch.Tensor):
        # # Re-align board memory to fit the shape described in init
        # # (B, T, H, W, C) -> (B, C, T, H, W)
        # assert board_memory.ndim == 5
        # board_encoding = board_memory.permute(0, 4, 1, 2, 3)
        #
        # board_encoding = self.conv_stack(board_encoding)

        # TODO DEBUG ONLY
        board_encoding = board_memory

        board_encoding = torch.flatten(board_encoding, start_dim=1)
        board_encoding = self.fc_stack(board_encoding)

        return board_encoding
    
    def convert_transitions_to_tensors(self, data_raw: t.List[DataPoint]):
        data_n = len(data_raw)

        # Use the same device and dtype as the network itself.
        dtype, device = self._detect_dtype_and_device()

        data_obs = torch.empty((data_n, self.narx_memory_length, *self.ObsShape), dtype=dtype, device=device)
        data_obs_next = torch.empty((data_n, self.narx_memory_length, *self.ObsShape), dtype=dtype, device=device)
        data_act = torch.empty((data_n, 1), dtype=torch.int64, device=device)
        data_act_next_mask = torch.zeros((data_n, Board.Size ** 2), dtype=torch.int64, device=device)
        data_rew = torch.empty((data_n, 1), dtype=dtype, device=device)
        data_done = torch.empty((data_n, 1), dtype=dtype, device=device)
        data_is_move = torch.empty((data_n, 1), dtype=dtype, device=device)

        for i_sample, point in enumerate(data_raw):
            data_obs[i_sample, ...] = self.obs_list_to_tensor([t.observation for t in point.history_now])
            data_obs_next[i_sample, ...] = self.obs_list_to_tensor([t.observation for t in point.history_next])
            data_act[i_sample] = point.transition_now.action
            data_act_next_mask[i_sample, point.transition_next.valid_actions] = 1
            data_rew[i_sample] = point.transition_now.reward
            data_done[i_sample] = point.transition_now.done
            data_is_move[i_sample] = int(point.transition_now.is_move)
        
        return DataTensors(data_obs, data_obs_next, data_act, data_act_next_mask, data_rew, data_done, data_is_move)

    def obs_list_to_tensor(self, obs_list: t.List[np.ndarray]):
        # Pad the obs length to the required memory length.
        if len(obs_list) < self.narx_memory_length:
            obs_list = obs_list + [np.zeros_like(obs_list[0])] * (self.narx_memory_length - len(obs_list))

        dtype, device = self._detect_dtype_and_device()

        # Convert to a tensor and add a trivial channel dimension.
        return torch.tensor(np.stack(obs_list), dtype=dtype, device=device).unsqueeze(-1)

    def _detect_dtype_and_device(self):
        some_net_params = next(self.fc_stack.parameters())

        return some_net_params.dtype, some_net_params.device

    # @staticmethod
    # def action_to_one_hot(action):
    #     return torch.nn.functional.one_hot(torch.tensor(action), TicTacToe.BoardSize ** 2)
