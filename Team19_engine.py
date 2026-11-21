"""Rule and state engine for Group 19, owned by member 1.

This module is the single source of truth for Tablut legality, captures,
terminal states and position hashing. Every rule the official ``Tablut.py``
referee applies is reproduced exactly, so that search scores and tournament
games can never disagree with the official adjudicator, while the API stays
stateless and avoids ``copy.deepcopy``.

Public interface agreed in ``docs/INTERFACES.md``::

    compute_hash(board, is_attacker) -> int
    get_legal_moves(board, is_attacker) -> list[list[list[int]]]
    apply_move(board, move, is_attacker) -> list[list[str]]
    check_winner(board, is_attacker_next, has_legal_moves=True) -> str | None

A board is a 9x9 ``list[list[str]]`` whose cells are ``A`` (attacker),
``D`` (defender), ``K`` (king), ``.`` (empty square) and ``X`` (the castle
when the king is not on it).  Coordinates are ``[row, col]`` with zero-based
indices.  No function modifies its arguments: ``apply_move`` always returns
an independent board and ``get_legal_moves`` returns fresh lists.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

# ---------------------------------------------------------------------------
# Board alphabet and constants
# ---------------------------------------------------------------------------

ATTACKER = "A"
DEFENDER = "D"
KING = "K"
EMPTY = "."
CASTLE = "X"

SIZE = 9
CASTLE_ROW = 4
CASTLE_COL = 4

#: Terminal markers returned by :func:`check_winner`.
WIN_ATTACKER = "A"
WIN_DEFENDER = "D"
DRAW = "DRAW"

#: The seed required by the brief so that every team hashes identically.
ZOBRIST_SEED = 42

# Piece codes used by the official Zobrist table: attacker 0, defender 1,
# king 2.  Empty squares and the castle contribute nothing to the hash.
_CODE_BY_CELL = {ATTACKER: 0, DEFENDER: 1, KING: 2}

# Same order as the official ``dir_row``/``dir_col`` arrays (up, left, down,
# right) so that capture resolution visits neighbours in an identical order.
_DIRECTIONS = ((-1, 0), (0, -1), (1, 0), (0, 1))


# ---------------------------------------------------------------------------
# Zobrist hashing
# ---------------------------------------------------------------------------

def _build_zobrist_table(size: int = SIZE) -> dict[tuple[int, int, int], int]:
    """Build the official hash table with the mandated random draw order.

    ``Tablut.__init__`` draws ``getrandbits(64)`` for every square, then every
    piece code, then both sides to move, from a Mersenne Twister seeded with
    :data:`ZOBRIST_SEED`.  A private ``random.Random`` instance reproduces that
    exact sequence without touching the module-wide generator.
    """
    rng = random.Random(ZOBRIST_SEED)
    table: dict[tuple[int, int, int], int] = {}
    for square in range(size * size):
        for piece in range(3):
            for turn in range(2):
                table[(square, piece, turn)] = rng.getrandbits(64)
    return table


_ZOBRIST = _build_zobrist_table()
_HASH_KEYS = tuple(
    tuple(
        {piece: (_ZOBRIST[(row * SIZE + col, code, 0)],
                 _ZOBRIST[(row * SIZE + col, code, 1)])
         for piece, code in _CODE_BY_CELL.items()}
        for col in range(SIZE)
    )
    for row in range(SIZE)
)


def compute_hash(board: list[list[str]], is_attacker: bool) -> int:
    """Return the Zobrist hash of ``board`` for the side to move.

    The hash covers the board configuration *and* the player whose turn it
    is, exactly like ``Tablut.compute_zobrist_hash``.  Two positions that
    differ only in the side to move therefore hash differently, which is what
    the repetition rule needs.
    """
    turn = 0 if is_attacker else 1
    table = _ZOBRIST
    codes = _CODE_BY_CELL
    total = 0
    width = len(board)
    for row_index in range(width):
        row = board[row_index]
        offset = row_index * width
        for col_index in range(width):
            code = codes.get(row[col_index])
            if code is not None:
                total ^= table[(offset + col_index, code, turn)]
    return total


# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------

def get_legal_moves(
    board: list[list[str]], is_attacker: bool
) -> list[list[list[int]]]:
    """Return every legal move for the side to move.

    Pieces slide orthogonally any distance without jumping.  The empty castle
    may be crossed but never occupied, so a slide continues through ``X`` and
    only stops on ``.`` squares.  Moves are listed in a deterministic
    row-major, direction-major order; the search relies on that order for its
    first-move fallback and for tie breaking.
    """
    width = len(board)
    moves: list[list[list[int]]] = []
    append = moves.append
    if is_attacker:
        own = (ATTACKER,)
    else:
        own = (DEFENDER, KING)

    for row_index in range(width):
        row = board[row_index]
        for col_index in range(width):
            if row[col_index] not in own:
                continue
            for dr, dc in _DIRECTIONS:
                rr = row_index + dr
                cc = col_index + dc
                while 0 <= rr < width and 0 <= cc < width:
                    target = board[rr][cc]
                    if target == EMPTY:
                        append([[row_index, col_index], [rr, cc]])
                    elif target != CASTLE:
                        break
                    # ``CASTLE`` is traversable: keep sliding past it.
                    rr += dr
                    cc += dc
    return moves


def _has_legal_move(board: list[list[str]], is_attacker: bool) -> bool:
    """Check the first landing on each ray without building a move list."""
    width = len(board)
    own = (ATTACKER,) if is_attacker else (DEFENDER, KING)
    for row, cells in enumerate(board):
        for col, piece in enumerate(cells):
            if piece not in own:
                continue
            for dr, dc in _DIRECTIONS:
                rr, cc = row + dr, col + dc
                while 0 <= rr < width and 0 <= cc < width:
                    cell = board[rr][cc]
                    if cell == EMPTY:
                        return True
                    if cell != CASTLE:
                        break
                    rr += dr
                    cc += dc
    return False


def _is_blocked(board: list[list[str]], r1: int, c1: int, r2: int, c2: int) -> bool:
    """Return whether a straight slide from (r1, c1) to (r2, c2) is blocked."""
    if r1 == r2:
        step = 1 if c2 > c1 else -1
        for col_index in range(c1 + step, c2, step):
            cell = board[r1][col_index]
            if cell != EMPTY and cell != CASTLE:
                return True
        return False
    step = 1 if r2 > r1 else -1
    for row_index in range(r1 + step, r2, step):
        cell = board[row_index][c1]
        if cell != EMPTY and cell != CASTLE:
            return True
    return False


# ---------------------------------------------------------------------------
# Capture resolution
# ---------------------------------------------------------------------------

def _capture_axes(width: int, row: int, col: int) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (row + dr, col + dc, row + 2 * dr, col + 2 * dc)
        for dr, dc in _DIRECTIONS
        if 0 <= row + 2 * dr < width and 0 <= col + 2 * dc < width
    )


_CAPTURE_AXES = tuple(
    tuple(_capture_axes(SIZE, row, col) for col in range(SIZE))
    for row in range(SIZE)
)


def _resolve_captures(
    board: list[list[str]], row: int, col: int,
) -> list[tuple[int, int, str]]:
    """Apply every capture caused by the piece that just landed on row/col.

    ``board`` is the freshly copied board and is updated in place, mirroring
    ``Tablut.test_captures``.  Only the four neighbours of the moved piece are
    examined, so the mover itself can never be captured on its own turn, and
    all captures of one move are resolved with the same rules.
    """
    piece = board[row][col]
    width = len(board)
    axes = _CAPTURE_AXES[row][col] if width == SIZE else _capture_axes(width, row, col)
    captured: list[tuple[int, int, str]] = []

    if piece == ATTACKER:
        for nr, nc, br, bc in axes:
            neighbour = board[nr][nc]
            if neighbour == DEFENDER:
                beyond = board[br][bc]
                if beyond == ATTACKER or beyond == CASTLE:
                    board[nr][nc] = EMPTY
                    captured.append((nr, nc, neighbour))
                elif beyond == KING and (br, bc) == (CASTLE_ROW, CASTLE_COL):
                    # Exception: a defender beside the king in the castle can
                    # be taken once the castle is attacked from three sides.
                    if sum(board[br + dr][bc + dc] == ATTACKER for dr, dc in _DIRECTIONS) >= 3:
                        board[nr][nc] = EMPTY
                        captured.append((nr, nc, neighbour))
            elif neighbour == KING:
                # The king in or directly beside the castle must be surrounded
                # on all sides, where the empty castle counts as hostile.
                if abs(nr - CASTLE_ROW) + abs(nc - CASTLE_COL) <= 1:
                    trapped = all(
                        board[nr + dr][nc + dc] in (ATTACKER, CASTLE)
                        for dr, dc in _DIRECTIONS
                    )
                else:
                    trapped = board[br][bc] == ATTACKER
                if trapped:
                    board[nr][nc] = EMPTY
                    captured.append((nr, nc, neighbour))
        return captured

    # Defenders and the king capture attackers between two friendly squares,
    # where the empty castle also counts as friendly.
    if piece == DEFENDER or piece == KING:
        for nr, nc, br, bc in axes:
            if board[nr][nc] == ATTACKER and board[br][bc] in (DEFENDER, KING, CASTLE):
                board[nr][nc] = EMPTY
                captured.append((nr, nc, ATTACKER))
    return captured


def _slide(board: list[list[str]], r1: int, c1: int, r2: int, c2: int) -> list[list[str]]:
    child = [row[:] for row in board]
    child[r1][c1] = CASTLE if (r1, c1) == (CASTLE_ROW, CASTLE_COL) else EMPTY
    child[r2][c2] = board[r1][c1]
    return child


# ---------------------------------------------------------------------------
# Move application
# ---------------------------------------------------------------------------

def apply_move(
    board: list[list[str]], move: list[list[int]], is_attacker: bool
) -> list[list[str]]:
    """Return a new board with ``move`` played and its captures resolved.

    ``move`` is ``[[r1, c1], [r2, c2]]`` and must be legal for the side to
    move; an illegal or malformed move raises ``ValueError`` so that
    integration mistakes surface immediately instead of producing a position
    the official referee would reject.  The input board is never modified and
    the returned board shares no rows with it.
    """
    width = len(board)
    try:
        start, end = move
        r1, c1 = int(start[0]), int(start[1])
        r2, c2 = int(end[0]), int(end[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError("move must be [[r1, c1], [r2, c2]]") from None

    if not 0 <= r1 < width or not 0 <= c1 < width:
        raise ValueError("start square is outside the board")
    if not 0 <= r2 < width or not 0 <= c2 < width:
        raise ValueError("destination square is outside the board")

    piece = board[r1][c1]
    if is_attacker:
        if piece != ATTACKER:
            raise ValueError("attackers may only move attacker pieces")
    elif piece != DEFENDER and piece != KING:
        raise ValueError("defenders may only move defender pieces and the king")

    if r1 == r2 and c1 == c2:
        raise ValueError("a move must change the row or the column")
    if r1 != r2 and c1 != c2:
        raise ValueError("pieces move orthogonally, not diagonally")
    if board[r2][c2] != EMPTY:
        raise ValueError("destination square is not empty")
    if _is_blocked(board, r1, c1, r2, c2):
        raise ValueError("the path between the two squares is blocked")

    new_board = _slide(board, r1, c1, r2, c2)
    _resolve_captures(new_board, r2, c2)
    return new_board


def _on_edge(row: int, col: int) -> bool:
    return row == 0 or col == 0 or row == SIZE - 1 or col == SIZE - 1


@dataclass(slots=True)
class _SearchPosition:
    """A search-only board with the features that can be updated from a move."""

    board: list[list[str]]
    attacker_hash: int
    defender_hash: int
    attackers: int
    defenders: int
    king: tuple[int, int] | None

    def winner(self) -> str | None:
        if self.king is None:
            return WIN_ATTACKER
        if self.attackers < 2 or _on_edge(*self.king):
            return WIN_DEFENDER
        return None


def _search_position(board: list[list[str]]) -> _SearchPosition:
    """Scan the root once for both official hashes and material."""
    attacker_hash = defender_hash = attackers = defenders = 0
    king = None
    for row, cells in enumerate(board):
        for col, piece in enumerate(cells):
            keys = _HASH_KEYS[row][col].get(piece)
            if keys is None:
                continue
            attacker_hash ^= keys[0]
            defender_hash ^= keys[1]
            if piece == ATTACKER:
                attackers += 1
            elif piece == DEFENDER:
                defenders += 1
            else:
                king = (row, col)
    return _SearchPosition(board, attacker_hash, defender_hash, attackers, defenders, king)


def _apply_legal_move(
    position: _SearchPosition, move: list[list[int]],
) -> _SearchPosition:
    """Apply a generated move, reusing the public capture rules and updating both hashes."""
    (r1, c1), (r2, c2) = move
    piece = position.board[r1][c1]
    child = _slide(position.board, r1, c1, r2, c2)
    captures = _resolve_captures(child, r2, c2)
    start_a, start_d = _HASH_KEYS[r1][c1][piece]
    end_a, end_d = _HASH_KEYS[r2][c2][piece]
    attacker_hash = position.attacker_hash ^ start_a ^ end_a
    defender_hash = position.defender_hash ^ start_d ^ end_d
    attackers, defenders = position.attackers, position.defenders
    king = (r2, c2) if piece == KING else position.king
    for row, col, captured in captures:
        captured_a, captured_d = _HASH_KEYS[row][col][captured]
        attacker_hash ^= captured_a
        defender_hash ^= captured_d
        if captured == ATTACKER:
            attackers -= 1
        elif captured == DEFENDER:
            defenders -= 1
        else:
            king = None
    return _SearchPosition(child, attacker_hash, defender_hash, attackers, defenders, king)


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

def check_winner(
    board: list[list[str]],
    is_attacker_next: bool,
    has_legal_moves: bool = True,
) -> str | None:
    """Return ``"A"``, ``"D"``, ``"DRAW"`` or ``None`` for the given board.

    The checks follow the order of the official ``check_if_game_over``:

    1. the king stands on an edge square -> defenders win;
    2. no king on the board -> attackers win (the king was captured);
    3. fewer than two attackers -> defenders win;
    4. the side to move has no legal move -> draw.

    ``has_legal_moves`` is the caller's answer to step 4, normally
    ``bool(get_legal_moves(board, is_attacker_next))``; the search already has
    the move list and passes it in.  Its default ``True`` only means "assume
    moves exist" and must not be relied on for draw detection.

    Repetition is not checked here: it needs the real move history or the
    current search path, which the search module and the player own.
    """
    width = len(board)
    attackers = 0
    king_found = False
    last = width - 1
    for row_index in range(width):
        row = board[row_index]
        for col_index in range(width):
            cell = row[col_index]
            if cell == KING:
                if row_index == 0 or row_index == last:
                    return WIN_DEFENDER
                if col_index == 0 or col_index == last:
                    return WIN_DEFENDER
                king_found = True
            elif cell == ATTACKER:
                attackers += 1

    if not king_found:
        return WIN_ATTACKER
    if attackers < 2:
        return WIN_DEFENDER
    if not has_legal_moves:
        return DRAW
    return None


# Only the four agreed entry points are public functions; every helper above
# is private so that nobody integrates against a name that is not in
# docs/INTERFACES.md.

__all__ = [
    "ATTACKER",
    "CASTLE",
    "DEFENDER",
    "DRAW",
    "EMPTY",
    "KING",
    "WIN_ATTACKER",
    "WIN_DEFENDER",
    "apply_move",
    "check_winner",
    "compute_hash",
    "get_legal_moves",
]
