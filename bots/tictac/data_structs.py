from dataclasses import dataclass
import typing as t

import torch
import numpy as np

from bots.tictac import Square, Board


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    valid_actions: t.List[int]
    reward: float
    is_move: bool
    done: float = 0.0

    @staticmethod
    def get_empty_transition():
        return Transition(np.full(Board.Shape, fill_value=Square.Empty), -1, [-1],  0.0, False)


class Episode(t.NamedTuple):
    transitions: t.List[Transition]

    def __len__(self):
        return len(self.transitions)


class DataPoint(t.NamedTuple):
    transition_history: t.List[Transition]

    @property
    def transition_now(self):
        # The train transition is stored as the next to last in the history.
        return self.transition_history[-2]

    @property
    def transition_next(self):
        # The next (t + 1_ transition is stored as the last in the history.
        return self.transition_history[-1]

    @property
    def history_now(self):
        # All but the last, which is the next transition (t + 1).
        return self.transition_history[:-1]

    @property
    def history_next(self):
        # All but the first, which is too old for the history of the next transition.
        return self.transition_history[1:]


class DataTensors(t.NamedTuple):
    obs: torch.Tensor
    obs_next: torch.Tensor
    act: torch.Tensor
    act_next_mask: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor
    is_move: torch.Tensor
