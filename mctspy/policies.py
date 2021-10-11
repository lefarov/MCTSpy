from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mctspy.tree import DecisionNode

import math
import random
import typing as t


def uct(
    value: float, visits: int, total_visits: int, exploration_constant: float
) -> float:
    return value / visits + exploration_constant * math.sqrt(math.log(total_visits) / visits)


def uct_action(
    decision_node: DecisionNode, exploration_constant: float = 1 / math.sqrt(2)
) -> t.Hashable:
    """ Select the action in decision node according to UCB formula.
    """
    best_actions, best_score = [], None
    
    for action, chance_node in decision_node.children.items():
        
        score = uct(chance_node.value, chance_node.visits, decision_node.visits, exploration_constant)

        if best_score is None or score > best_score:
            best_score = score
            best_actions = [action]

        if score == best_score:
            best_actions.append(action)
    
    return random.choice(best_actions)


def random_action(decision_node: DecisionNode) -> t.Hashable:
    """ Choose actoins randomly at every decision node.
    """
    return random.choice(tuple(decision_node.children.keys()))
