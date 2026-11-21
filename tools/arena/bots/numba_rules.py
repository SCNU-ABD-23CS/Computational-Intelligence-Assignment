"""Flat, incrementally hashed JC4004 engine and its reference-test interface."""

from __future__ import annotations

import random

import numpy as np
from numba import njit as _njit

import Tablut


ATTACKER, DEFENDER, KING, EMPTY, CASTLE = "A", "D", "K", ".", "X"
SIZE, CASTLE_ROW, CASTLE_COL = 9, 4, 4
WIN_ATTACKER, WIN_DEFENDER, DRAW = "A", "D", "DRAW"
_A, _D, _K, _X = 1, 2, 3, 4
_CELLS = ".ADKX"
_CODES = {cell: index for index, cell in enumerate(_CELLS)}
_CASTLE = 40
_AREA = 81
_MOVE_CAPACITY = 324
_NEAR = np.full((_AREA, 4), -1, dtype=np.int16)
_BEYOND = np.full_like(_NEAR, -1)
_EDGE = np.zeros(_AREA, dtype=np.bool_)
_KING_CASTLE = np.zeros(_AREA, dtype=np.bool_)
for _square in range(_AREA):
    _row, _col = divmod(_square, SIZE)
    _EDGE[_square] = _row in (0, 8) or _col in (0, 8)
    _KING_CASTLE[_square] = abs(_row - 4) + abs(_col - 4) <= 1
    for _index, (_dr, _dc) in enumerate(((-1, 0), (0, -1), (1, 0), (0, 1))):
        if 0 <= _row + _dr < SIZE and 0 <= _col + _dc < SIZE:
            _NEAR[_square, _index] = _square + _dr * SIZE + _dc
        if 0 <= _row + 2 * _dr < SIZE and 0 <= _col + 2 * _dc < SIZE:
            _BEYOND[_square, _index] = _square + 2 * (_dr * SIZE + _dc)

_ZOBRIST = np.zeros((_AREA, 5, 2), dtype=np.uint64)
_rng = random.Random(Tablut.ZOBRIST_SEED)
for _square in range(_AREA):
    for _piece in (_A, _D, _K):
        for _turn in range(2):
            _ZOBRIST[_square, _piece, _turn] = _rng.getrandbits(64)


def _flat(board: list[list[str]]) -> np.ndarray:
    return np.fromiter((_CODES[cell] for row in board for cell in row), count=_AREA, dtype=np.int8)


def _nested(board: np.ndarray) -> list[list[str]]:
    return [[_CELLS[cell] for cell in row] for row in board.reshape((9, 9))]


def _move_list(code: int) -> list[list[int]]:
    start, end = divmod(code, _AREA)
    return [list(divmod(start, SIZE)), list(divmod(end, SIZE))]


@_njit(cache=True)
def _describe(board):
    hash_a = np.uint64(0)
    hash_d = np.uint64(0)
    attackers = defenders = 0
    king = -1
    for square in range(_AREA):
        piece = board[square]
        if 1 <= piece <= _K:
            hash_a ^= _ZOBRIST[square, piece, 0]
            hash_d ^= _ZOBRIST[square, piece, 1]
            if piece == _A:
                attackers += 1
            elif piece == _D:
                defenders += 1
            else:
                king = square
    return hash_a, hash_d, attackers, defenders, king


@_njit(cache=True, inline="always")
def _friendly(piece, attacker):
    return piece == _A if attacker else piece == _D or piece == _K


@_njit(cache=True)
def _legal(board, attacker, moves):
    length = 0
    for source in range(_AREA):
        if not _friendly(board[source], attacker):
            continue
        for direction in range(4):
            destination = int(_NEAR[source, direction])
            while destination >= 0:
                occupant = board[destination]
                if occupant == 0:
                    moves[length] = source * _AREA + destination
                    length += 1
                elif occupant != _X:
                    break
                destination = int(_NEAR[destination, direction])
    return length


@_njit(cache=True)
def _has_legal(board, attacker):
    for source in range(_AREA):
        if not _friendly(board[source], attacker):
            continue
        for direction in range(4):
            target = int(_NEAR[source, direction])
            while target >= 0 and board[target] == _X:
                target = int(_NEAR[target, direction])
            if target >= 0 and board[target] == 0:
                return True
    return False


@_njit(cache=True, inline="always")
def _winner(attackers, king):
    if king < 0:
        return _A
    if attackers < 2 or _EDGE[king]:
        return _D
    return 0


@_njit(cache=True)
def _push(board, move, hash_a, hash_d, attackers, defenders, king, squares, pieces):
    source, target = divmod(move, _AREA)
    moved = board[source]
    hash_a ^= _ZOBRIST[source, moved, 0] ^ _ZOBRIST[target, moved, 0]
    hash_d ^= _ZOBRIST[source, moved, 1] ^ _ZOBRIST[target, moved, 1]
    board[source] = _X if source == _CASTLE else 0
    board[target] = moved
    if moved == _K:
        king = target

    count = 0
    for direction in range(4):
        far_square = _BEYOND[target, direction]
        if far_square < 0:
            continue
        adjacent = _NEAR[target, direction]
        victim = board[adjacent]
        far = board[far_square]
        capture = False
        if moved != _A:
            capture = victim == _A and (far == _D or far == _K or far == _X)
        elif victim == _D:
            capture = far == _A or far == _X
            if far == _K and far_square == _CASTLE:
                surrounding = 0
                for axis in range(4):
                    surrounding += board[_NEAR[_CASTLE, axis]] == _A
                capture = surrounding >= 3
        elif victim == _K:
            if _KING_CASTLE[adjacent]:
                capture = True
                for axis in range(4):
                    around = board[_NEAR[adjacent, axis]]
                    if around != _A and around != _X:
                        capture = False
                        break
            else:
                capture = far == _A
        if capture:
            board[adjacent] = 0
            squares[count] = adjacent
            pieces[count] = victim
            count += 1
            hash_a ^= _ZOBRIST[adjacent, victim, 0]
            hash_d ^= _ZOBRIST[adjacent, victim, 1]
            if victim == _A:
                attackers -= 1
            elif victim == _D:
                defenders -= 1
            else:
                king = -1
    return hash_a, hash_d, attackers, defenders, king, count


@_njit(cache=True, inline="always")
def _pop(board, move, squares, pieces, captures):
    source, target = divmod(move, _AREA)
    board[source] = board[target]
    board[target] = 0
    for capture in range(captures):
        board[squares[capture]] = pieces[capture]


def compute_hash(board: list[list[str]], is_attacker: bool) -> int:
    hash_a, hash_d, _, _, _ = _describe(_flat(board))
    return int(hash_a if is_attacker else hash_d)


def get_legal_moves(board: list[list[str]], is_attacker: bool) -> list[list[list[int]]]:
    results = np.empty(_MOVE_CAPACITY, dtype=np.int16)
    length = _legal(_flat(board), is_attacker, results)
    return [_move_list(int(move)) for move in results[:length]]


def apply_move(board: list[list[str]], move: list[list[int]], is_attacker: bool) -> list[list[str]]:
    try:
        (r, c), (rr, cc) = move
        r, c, rr, cc = map(int, (r, c, rr, cc))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Expected two coordinates") from error
    if not all(0 <= value < SIZE for value in (r, c, rr, cc)):
        raise ValueError("Coordinate outside the board")
    encoded = (SIZE * r + c) * _AREA + SIZE * rr + cc
    flat = _flat(board)
    legal = np.empty(_MOVE_CAPACITY, dtype=np.int16)
    if encoded not in legal[:_legal(flat, is_attacker, legal)]:
        raise ValueError("Illegal slide")
    hash_a, hash_d, attackers, defenders, king = _describe(flat)
    _push(flat, encoded, np.uint64(hash_a), np.uint64(hash_d), attackers, defenders, king,
          np.empty(4, dtype=np.int16), np.empty(4, dtype=np.int8))
    return _nested(flat)


def check_winner(board: list[list[str]], is_attacker_next: bool,
                 has_legal_moves: bool = True) -> str | None:
    _, _, attackers, _, king = _describe(_flat(board))
    outcome = _winner(attackers, king)
    return _CELLS[outcome] if outcome else None if has_legal_moves else DRAW
