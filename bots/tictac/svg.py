# %%
import drawSvg as draw
import typing as t

from bots.tictac.game import Board, Square, TicTacToe


GRID_STROKE_WIDTH = 5.
MARK_STROKE_WIDTH = 10.
COLORS = {
    "grid": "#5d6d7e",
    "cross": "#f1948a",
    "nought": "#2471a3",
    "square": "#7d3c98",
    "last_move": "#148f77",
}


def get_grid_svg(**kwargs):
    # TODO: inherit from draw.DrawingBasicElement
    return (
        draw.Line(100, 0, 100, 300, **kwargs),
        draw.Line(200, 0, 200, 300, **kwargs),
        draw.Line(0, 100, 300, 100, **kwargs),
        draw.Line(0, 200, 300, 200, **kwargs),
    )


def get_cross_svg(cx, cy, l, **kwargs):
    # TODO: inherit from draw.DrawingBasicElement
    return (
        draw.Line(cx - l/2, cy - l/2, cx + l/2, cy + l/2, **kwargs),
        draw.Line(cx - l/2, cy + l/2, cx + l/2, cy - l/2, **kwargs),
    )


def board(
    board: Board,
    lastmove: t.Optional[int] = None,
    squares: t.Optional[t.Iterable] = None,
    colors: t.Optional[t.Dict] = None):
    """Render Tic-Tac-Toe board to SVG. The function follows the reconchess.svg interface."""

    if colors is None:
        colors = COLORS

    canvas = draw.Drawing(300, 300, displayInline=False)
    canvas.extend(get_grid_svg(stroke_width=GRID_STROKE_WIDTH, stroke=colors["grid"]))

    for square in range(TicTacToe.BoardSize ** 2):
        # define position for the mark
        cx = 50 + (square % 3) * 100
        cy = 250 - (square // 3) * 100

        if board[square] == Square.Cross:
            canvas.extend(get_cross_svg(cx, cy, 50, stroke_width=MARK_STROKE_WIDTH, stroke=colors["cross"]))
        elif board[square] == Square.Nought:
            canvas.append(draw.Circle(cx, cy, 30, fill_opacity=0.0, stroke_width=MARK_STROKE_WIDTH, stroke=colors["nought"]))
        else:
            # Empty square
            pass

    # Mark last move
    if lastmove is not None:
        # define position for the square
        x = (lastmove % 3) * 100
        y = 200 - (lastmove // 3) * 100

        canvas.append(draw.Rectangle(x, y, 100, 100, fill_opacity=0.3, fill=colors["last_move"], stroke_width=0.,))

    # Mark provided squares (i.e. Sense actions)
    if squares is not None:
        for square in squares:
            # define position for the square
            x = (square % 3) * 100
            y = 200 - (square // 3) * 100

            canvas.append(draw.Rectangle(x, y, 100, 100, fill_opacity=0.3, fill=colors["square"], stroke_width=0.,))

    # TODO: Implement the win-line highlighting

    return canvas


# %%
if __name__ == "__main__":

    # %%
    b = Board()
    b[0] = Square.Cross
    b[5] = Square.Nought
    b[6] = Square.Nought
    b[7] = Square.Cross

    board(b, lastmove=0, squares=(3,))
