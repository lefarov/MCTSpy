import functools
import operator
import torch
import typing as t

from bots.tictac import TicTacToe


class TicTacQNet(torch.nn.Module):

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

        self.conv_stack = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=n_hidden,
                kernel_size=(1, 5, 5),
                stride=(1, 3, 3)
            ),
            torch.nn.ReLU(),
        )

        dummy_input = torch.zeros((1, in_channels, self.narx_memory_length, 8, 8))
        fc_input_size = functools.reduce(operator.mul, self.conv_stack(dummy_input).shape)

        self.fc_stack = torch.nn.Sequential(
            torch.nn.Linear(fc_input_size, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
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
        # Re-align board memory to fit the shape described in init
        # (B, T, H, W, C) -> (B, C, T, H, W)
        assert board_memory.ndim == 5
        board_encoding = board_memory.permute(0, 4, 1, 2, 3)

        board_encoding = self.conv_stack(board_encoding)
        board_encoding = torch.flatten(board_encoding, start_dim=1)
        board_encoding = self.fc_stack(board_encoding)

        return board_encoding
