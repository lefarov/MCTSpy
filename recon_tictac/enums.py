from enum import IntEnum


class ActionType(IntEnum):
    Unknown = -1
    Sense = 0
    Move = 1


class Player(IntEnum):
    Nought = 0
    Cross = 1


class WinReason(IntEnum):
    Draw = 0
    MatchThree = 1


class Square(IntEnum):
    Empty = -1
    Cross = Player.Cross
    Nought = Player.Nought