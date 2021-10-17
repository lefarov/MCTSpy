import functools
import operator
import torch
import typing as t

from bots.blindchess.utilities import PIECE_INDEX


class TestQNet(torch.nn.Module):

    def __init__(self, narx_memory_length, n_hidden, channels_per_layer: t.Optional[t.List[int]] = None):
        super().__init__()

        self.narx_memory_length = narx_memory_length
        self.n_hidden = n_hidden
        self.channels_per_layer = channels_per_layer or [64, 128, 256]

        # Board convolution backbone:
        # 3D convolution layer is applied to a thensor with shape (N,C​,D​,H​,W​)
        # where N - batch size, C (channels) - one-hot-encoding of a piece,
        # D (depth) - history length, H and W are board dimentions (i.e. 8x8).

        self.conv_stack = torch.nn.Sequential(
            # torch.nn.Conv3d(
            #     in_channels=len(PIECE_INDEX),
            #     out_channels=self.channels_per_layer[0],
            #     kernel_size=(3, 3, 3),
            #     # stride=(2, 2, 2)
            # ),
            # torch.nn.ReLU(),
            # torch.nn.Conv3d(
            #     in_channels=self.channels_per_layer[0],
            #     out_channels=self.channels_per_layer[1],
            #     kernel_size=(3, 3, 3),
            #     # stride=(2, 2, 2)
            # ),
            # torch.nn.ReLU(),
            # torch.nn.Conv3d(
            #     in_channels=self.channels_per_layer[1],
            #     out_channels=self.channels_per_layer[2],
            #     kernel_size=(3, 3, 3),
            #     # stride=(2, 2, 2)
            # ),
            # torch.nn.ReLU()
            torch.nn.Conv3d(
                in_channels=len(PIECE_INDEX),
                out_channels=64,
                kernel_size=(5, 5, 5),
                stride=(3, 3, 3)
            ),
            torch.nn.ReLU(),
            # torch.nn.Conv3d(
            #     in_channels=64,
            #     out_channels=128,
            #     kernel_size=(3, 3, 3),
            #     stride=(3, 3, 3)
            # ),
            # torch.nn.ReLU()
        )

        dummy_input = torch.zeros((1, len(PIECE_INDEX), self.narx_memory_length, 8, 8))
        fc_input_size = functools.reduce(operator.mul, self.conv_stack(dummy_input).shape)

        self.fc_stack = torch.nn.Sequential(
            torch.nn.Linear(fc_input_size, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
        )

        # Player heads
        self.fc_state_val = torch.nn.Linear(self.n_hidden, 1)
        self.fc_sense_adv = torch.nn.Linear(self.n_hidden, 64)
        self.fc_move_adv = torch.nn.Linear(self.n_hidden, 64 * 64)
        # Opponent heads
        self.fc_opponent_move = torch.nn.Linear(self.n_hidden, 64 * 64)

    def forward(self, board_memory: torch.Tensor):
        # Compute backbone
        board_encoding = self.backbone(board_memory)

        # Compute heads
        state_value = self.value(board_encoding)
        sense_q = self.sense(board_encoding, state_value)
        move_q = self.move(board_encoding, state_value)
        opponent_move = self.opponent_move(board_encoding)

        return state_value, sense_q, move_q, opponent_move

    def backbone(self, boar_memory: torch.Tensor):
        # Re-align board memory to fit the shape described in init
        # (B, T, H, W, C) -> (B, C, T, H, W)
        board_encoding = boar_memory.permute(0, 4, 1, 2, 3)
        board_encoding = self.conv_stack(board_encoding)
        board_encoding = torch.flatten(board_encoding, start_dim=1)
        board_encoding = self.fc_stack(board_encoding)

        return board_encoding

    def value(self, board_encoding: torch.Tensor):
        state_value = torch.nn.functional.relu(self.fc_state_val(board_encoding))
        return state_value

    def sense(self, board_encoding: torch.Tensor, state_value: torch.Tensor=None):
        if state_value is None:
            state_value = self.value(board_encoding)

        sense_adv = torch.nn.functional.relu(self.fc_sense_adv(board_encoding))
        sense_q = state_value + sense_adv - sense_adv.mean(-1, keepdim=True)
        return sense_q

    def move(self, board_encoding: torch.Tensor, state_value: torch.Tensor=None):
        if state_value is None:
            state_value = self.value(board_encoding)

        move_adv = torch.nn.functional.relu(self.fc_move_adv(board_encoding))
        move_q = state_value + move_adv - move_adv.mean(-1, keepdim=True)
        return move_q

    def opponent_move(self, board_encoding: torch.Tensor):
        opponent_move = torch.nn.functional.relu(self.fc_opponent_move(board_encoding))
        return opponent_move
