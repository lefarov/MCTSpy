import typing as t

from recon_tictac.board import Board
from recon_tictac.render import render_board


def render_board(
    board: Board,
    lastmove: t.Optional[int] = None,
    squares: t.Optional[t.Iterable] = None,
    colors: t.Optional[t.Dict] = None
):
    """Render Tic-Tac-Toe board to SVG. The function follows the reconchess.svg interface."""
    canvas = render_board(board, lastmove, squares, colors)
    return canvas.asSvg()

