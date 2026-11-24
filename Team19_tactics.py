"""Native escape-lane tactics used by the timed search."""

import numpy as np
from numba import njit

from Team19_engine import _A, _AREA, _CASTLE, _D, _K, _KING_CASTLE, _NEAR, _X, _king_win


_RAY = np.full((_AREA, _AREA), -1, dtype=np.int8)
for _start in range(_AREA):
    for _direction in range(4):
        _square = _NEAR[_start, _direction]
        while _square >= 0:
            _RAY[_start, _square] = _direction
            _square = _NEAR[_square, _direction]

# A lane is identified by its edge square. Parallel exits on different
# files or ranks get different bits, even if they point in the same direction.
_LANE_KEYS = np.zeros((_AREA, 16), dtype=np.int64)
for _start in range(_AREA):
    _row, _col = divmod(_start, 9)
    _edges = (_col, 9 + _row, 18 + _col, 27 + _row)
    for _mask in range(1, 16):
        for _direction, _edge in enumerate(_edges):
            if _mask & (1 << _direction):
                _LANE_KEYS[_start, _mask] |= 1 << _edge

@njit(inline="always")
def _clear(piece):
    return piece == 0 or piece == _X


@njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def escape_lines(board, king, stops):
    """Return masks for open rays and rays held by a single attacker or guard."""
    open_lanes = lone_attackers = lone_guards = 0
    for direction in range(4):
        square = _NEAR[king, direction]
        while square >= 0 and _clear(board[square]):
            square = _NEAR[square, direction]
        stops[direction] = square
        if square < 0:
            open_lanes |= 1 << direction
            continue
        piece = board[square]
        square = _NEAR[square, direction]
        while square >= 0 and _clear(board[square]):
            square = _NEAR[square, direction]
        if square < 0:
            if piece == _A:
                lone_attackers |= 1 << direction
            elif piece == _D:
                lone_guards |= 1 << direction
    return open_lanes, lone_attackers, lone_guards


@njit(inline="always")
def guard_releases(source, target, king, lone_guards):
    ray = _RAY[king, source]
    return ray >= 0 and lone_guards & (1 << ray) != 0 and _RAY[king, target] != ray


@njit(inline="always")
def leaves_open_lane(king, target, open_lanes):
    ray = _RAY[king, target]
    return ray < 0 or open_lanes != 1 << ray


@njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def king_turn_lanes(board, source, target):
    """New escape rays from a position where the king had no direct exit."""
    # Moving on a blocked rank or file cannot unblock that same line.
    # Only the two rays perpendicular to the slide can open.
    lanes = 0
    flank = (_RAY[source, target] + 1) & 3
    for direction in (flank, (flank + 2) & 3):
        square = _NEAR[target, direction]
        while square >= 0 and _clear(board[square]):
            square = _NEAR[square, direction]
        if square < 0:
            lanes |= 1 << direction
    return lanes


@njit(inline="always")
def _escape_blocks(board, king, lane, limit):
    # On a clear escape ray, a blocking attacker can only arrive laterally.
    # Scanning outward from each landing finds the only piece that can reach
    # it on each side. The throne can be crossed, but cannot be a landing.
    replies = 0
    landing = _NEAR[king, lane]
    flank = (lane + 1) & 3
    while landing >= 0:
        if landing != _CASTLE:
            for direction in (flank, (flank + 2) & 3):
                square = _NEAR[landing, direction]
                while square >= 0 and _clear(board[square]):
                    square = _NEAR[square, direction]
                if square >= 0 and board[square] == _A:
                    replies += 1
                    if replies == limit:
                        return replies
        landing = _NEAR[landing, lane]
    return replies


@njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def king_turn_replies(board, source, target, lanes, limit):
    """Count direct blocks or reject a king turn that permits capture."""
    board[source] = _X if source == _CASTLE else 0
    board[target] = _K
    replies = 0
    # A king capture wins immediately. Without one, a single slide cannot
    # close two rays. Actual search still adjudicates repetition draws.
    if _king_win(board, True, target) >= 0:
        replies = limit
    elif lanes & (lanes - 1) == 0:
        direction = 0
        while lanes != 1 << direction:
            direction += 1
        replies = _escape_blocks(board, target, direction, limit)
    board[source], board[target] = _K, 0
    return replies


@njit(inline="always")
def _capture_support(board, king, lane):
    if not _KING_CASTLE[king]:
        return board[_NEAR[king, (lane + 2) & 3]] == _A
    for direction in range(4):
        if direction != lane:
            piece = board[_NEAR[king, direction]]
            if piece != _A and piece != _X:
                return False
    return True


@njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def guard_reply_profile(board, source, king, lane, limit):
    """Count replies as if the guard were lifted, with their possible obstructions."""
    count = captures = 0
    blocks_low = blocks_high = captures_low = captures_high = np.uint64(0)
    landing = _NEAR[king, lane]
    exposed = source == landing and _capture_support(board, king, lane)
    flank = (lane + 1) & 3
    while landing >= 0:
        if landing != _CASTLE:
            for direction in (flank, (flank + 2) & 3):
                square = _NEAR[landing, direction]
                lower = upper = np.uint64(0)
                while square >= 0 and _clear(board[square]):
                    bit = np.uint64(1) << np.uint64(square & 63)
                    if square < 64:
                        lower |= bit
                    else:
                        upper |= bit
                    square = _NEAR[square, direction]
                if square >= 0 and board[square] == _A:
                    count += 1
                    blocks_low |= lower
                    blocks_high |= upper
                    if exposed and landing == source:
                        captures += 1
                        captures_low |= lower
                        captures_high |= upper
                    # The guard can obstruct at most one of these replies.
                    if count > limit:
                        return count, captures, blocks_low, blocks_high, captures_low, captures_high
        landing = _NEAR[landing, lane]
    return count, captures, blocks_low, blocks_high, captures_low, captures_high


@njit(inline="always")
def guard_release_replies(profile, target, limit):
    count, captures, blocks_low, blocks_high, captures_low, captures_high = profile
    bit = np.uint64(1) << np.uint64(target & 63)
    if captures > int(bool((captures_low if target < 64 else captures_high) & bit)):
        return limit
    return count - int(bool((blocks_low if target < 64 else blocks_high) & bit))


@njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def defender_capture_targets(board):
    """Squares worth checking for captures by a defender on this board."""
    lower = upper = np.uint64(0)
    for victim in range(_AREA):
        if board[victim] != _A:
            continue
        for direction in range(4):
            target = _NEAR[victim, direction]
            anchor = _NEAR[victim, (direction + 2) & 3]
            if target < 0 or anchor < 0 or board[target] != 0:
                continue
            piece = board[anchor]
            if piece == _D or piece == _K or piece == _X:
                bit = np.uint64(1) << np.uint64(target & 63)
                if target < 64:
                    lower |= bit
                else:
                    upper |= bit
    return lower, upper


@njit(inline="always")
def _guard_reaches(board, target, anchor):
    for ray in range(4):
        square = _NEAR[target, ray]
        while square >= 0:
            piece = board[square]
            if not _clear(piece):
                if (square != anchor or square == _CASTLE) and (piece == _D or piece == _K):
                    return True
                break
            square = _NEAR[square, ray]
    return False


@njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def capture_landings(board, blocker):
    """Empty squares from which the defenders can capture this attacker next turn."""
    lower = upper = np.uint64(0)
    for direction in range(4):
        target = _NEAR[blocker, direction]
        anchor = _NEAR[blocker, (direction + 2) & 3]
        if target < 0 or anchor < 0 or board[target] != 0:
            continue
        support = board[anchor]
        if (support == _D or support == _K or support == _X) and _guard_reaches(board, target, anchor):
            bit = np.uint64(1) << np.uint64(target & 63)
            if target < 64:
                lower |= bit
            else:
                upper |= bit
    return lower, upper


@njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def exposed_stops(board, stops, lone_attackers):
    """Identify one-piece screens that a legal reply can take."""
    vulnerable = 0
    lower = upper = np.uint64(0)
    for direction in range(4):
        if lone_attackers & (1 << direction):
            low, high = capture_landings(board, stops[direction])
            if low or high:
                vulnerable |= 1 << direction
                lower |= low
                upper |= high
    return vulnerable, lower, upper


@njit(inline="always")
def screen_priority(source, target, king, open_lanes, lone_attackers, vulnerable, lower, upper):
    ray = _RAY[king, target]
    source_ray = _RAY[king, source]
    priority = 0
    if ray >= 0 and ray != source_ray:
        bit = 1 << ray
        if open_lanes & bit:
            priority = 24000 if open_lanes == bit else 2300
        elif lone_attackers & bit:
            priority = 3200 if vulnerable & bit else 900
    if (lower if target < 64 else upper) & (np.uint64(1) << np.uint64(target & 63)):
        priority += 2800
    return priority


@njit(inline="always")
def captures_screen(board, source, target, stops, lone_attackers):
    """A guard may replace the captured blocker on the ray, then step aside."""
    for direction in range(4):
        if not lone_attackers & (1 << direction):
            continue
        victim = stops[direction]
        for axis in range(4):
            if _NEAR[victim, axis] != target:
                continue
            support = _NEAR[victim, (axis + 2) & 3]
            if support >= 0 and (support != source or support == _CASTLE):
                piece = board[support]
                if piece == _D or piece == _K or piece == _X:
                    return True
    return False
