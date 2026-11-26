"""Offline evidence gathering for the Team 19 standard-opening book.

The tool deliberately does not write a book.  It turns a small set of exact
standard-opening decision points into reproducible evidence records.  A book
entry is only ready for ``Team19_book_data.BOOK_LINES`` after its candidate
has survived the requested continuation trials.

Examples:

    D:/anaconda/python.exe -X utf8 -B -m tools.arena.book_trials
    D:/anaconda/python.exe -X utf8 -B -m tools.arena.book_trials --run --budget 60
    D:/anaconda/python.exe -X utf8 -B -m tools.arena.book_trials --root-finalists --run --budget 10

The first command reports the current v4 recommendation and historical
alternatives.  ``--run`` plays each selected line to the end through the
official referee, against every requested opponent.  It never imports the
online book module and it stores the report only when ``--output`` is given.
"""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import Tablut
import Team19_engine as engine
import Team19_search as search

from . import RESULTS
from .openings import Opening, start
from .runner import OfficialClock, ObservedPlayer, ObservedReferee, TEAMS, _player, _reason


MoveKey = tuple[int, int, int, int]
_ARCHIVE_RESULTS = Path(__file__).resolve().parents[3] / "results"


def _move_key(move: list[list[int]] | tuple[tuple[int, int], tuple[int, int]]) -> MoveKey:
    return move[0][0], move[0][1], move[1][0], move[1][1]


def _move_list(move: MoveKey) -> list[list[int]]:
    return [[move[0], move[1]], [move[2], move[3]]]


def _line_key(moves: tuple[MoveKey, ...]) -> str:
    return " ".join("%d%d%d%d" % move for move in moves)


@dataclass(frozen=True)
class Case:
    """A known decision point described only by moves from the initial board."""

    name: str
    opponents: tuple[str, ...]
    prefix: tuple[MoveKey, ...]
    side: str


# These paths come from the standard-opening logs named in the handoff.  The
# final move of a reported candidate is the only move that can become a book
# recommendation.  The preceding moves are evidence and lookup context.
KNOWN_CASES = (
    Case(
        "qs-attacker-ply5", ("QSearch-PVS",),
        ((0, 5, 1, 5), (2, 4, 2, 3), (0, 3, 1, 3), (2, 3, 2, 4)), "A",
    ),
    Case(
        "v3-defender-ply2", ("Team19-v3",), ((0, 5, 1, 5),), "D",
    ),
    Case(
        "v3-attacker-ply11", ("Team19-v3",),
        ((0, 5, 1, 5), (4, 6, 5, 6), (5, 0, 5, 1), (4, 5, 7, 5),
         (3, 8, 3, 6), (4, 4, 4, 5), (4, 7, 6, 7), (5, 4, 5, 3),
         (4, 8, 4, 6), (5, 6, 6, 6)), "A",
    ),
)

# The three best D4-family representatives in the 10-second, five-policy
# screen (``book_screen_results.json``), ordered by their worst v4 continuation
# score: 0313 (-2476), 1412 (-3439), and 0323 (-3578).  They are constants so a
# later continuation comparison cannot silently change its candidate set.
ROOT_FINALISTS: tuple[MoveKey, ...] = (
    (0, 3, 1, 3),
    (1, 4, 1, 2),
    (0, 3, 2, 3),
)
_ROOT_FINALIST_OPPONENTS = ("Team19-v3", "QSearch-PVS")


def root_finalist_cases() -> tuple[Case, ...]:
    """Return the fixed root candidates as attacker-to-move continuation cases."""
    return tuple(
        Case(f"root-finalist-{_line_key((move,))}", _ROOT_FINALIST_OPPONENTS, (), "A")
        for move in ROOT_FINALISTS
    )


def _official_position(prefix: tuple[MoveKey, ...]) -> tuple[list[list[str]], bool, set[int]]:
    """Replay a prefix using the supplied referee and retain its exact hashes."""
    game = Tablut.Tablut()
    game.past_positions = [game.compute_zobrist_hash(True)]
    with redirect_stdout(StringIO()):
        for ply, move in enumerate(prefix):
            attacker = ply % 2 == 0
            if not game.make_move(attacker, _move_list(move)):
                raise ValueError(f"Illegal prefix at ply {ply + 1}: {move}")
            if game.check_if_game_over(not attacker):
                raise ValueError(f"Terminal prefix at ply {ply + 1}: {prefix}")
    side = len(prefix) % 2 == 0
    return [row[:] for row in game.board], side, set(game.past_positions)


def _v4_choice(prefix: tuple[MoveKey, ...], seconds: float) -> MoveKey:
    board, attacker, history = _official_position(prefix)
    # The compiled v4 evaluator owns its independent 22-weight vector in
    # Team19_weights.  Passing the legacy seven-feature Python vector causes
    # the native selector to reject it and would measure a different engine.
    native = search._NativeSearch(board)
    move = native.choose(board, attacker, perf_counter() + seconds, history)
    return _move_key(move)


def _policy_choice(team: str, prefix: tuple[MoveKey, ...], budget: float) -> MoveKey:
    """Ask one frozen arena policy for its move at an exact recorded state."""
    opening = Opening(f"probe-{_line_key(prefix)}", [_move_list(move) for move in prefix], "")
    started = start(opening)
    _seed_missing_predecessor(started, opening)
    attacker = len(prefix) % 2 == 0
    with ExitStack() as stack:
        stack.enter_context(patch.object(Tablut, "TIME_BUDGET", budget))
        stack.enter_context(patch.dict(sys.modules, {"Team19_book": None}))
        player = _player(team, 20261126, started, attacker)
        move = (player.play_attacker(deepcopy(started.referee.board)) if attacker
                else player.play_defender(deepcopy(started.referee.board)))
    return _move_key(move)


def _forced_child(case: Case, candidate: MoveKey, seconds: float) -> dict:
    """Score a candidate only after the opponent's actual frozen-v4 reply.

    Each search starts with the fully replayed referee history, then receives
    the candidate child and its side-to-move hash.  Scores are reported from
    both the Attackers' perspective and the book side's perspective.
    """
    board, attacker, history = _official_position(case.prefix)
    child = engine.apply_move(board, _move_list(candidate), attacker)
    reply_side = not attacker
    history.add(engine.compute_hash(child, reply_side))
    reply_search = search._NativeSearch(child)
    reply = reply_search.choose(child, reply_side, perf_counter() + seconds, history)
    reply_score = reply_search.last_search.score
    after_reply = engine.apply_move(child, reply, reply_side)
    history.add(engine.compute_hash(after_reply, attacker))
    recovery_search = search._NativeSearch(after_reply)
    recovery = recovery_search.choose(after_reply, attacker, perf_counter() + seconds, history)
    recovery_score = recovery_search.last_search.score
    sign = 1.0 if case.side == "A" else -1.0
    return {
        "reply": _move_key(reply),
        "replyDepth": reply_search.last_search.depth,
        "scoreAfterReplyAttackPerspective": reply_score,
        "scoreAfterReplyBookPerspective": sign * reply_score,
        "recovery": _move_key(recovery),
        "recoveryDepth": recovery_search.last_search.depth,
        "scoreAfterRecoveryAttackPerspective": recovery_score,
        "scoreAfterRecoveryBookPerspective": sign * recovery_score,
    }


def _attempt_moves(row: dict) -> list[MoveKey]:
    return [_move_key(event["move"]) for event in row.get("prefix", ())] + [
        _move_key(attempt["event"]["move"])
        for attempt in row.get("attempts", ()) if "event" in attempt
    ]


def _history_candidates(prefix: tuple[MoveKey, ...], side: str) -> dict[MoveKey, list[dict]]:
    """Find all observed continuations of the exact prefix in saved arena logs."""
    found: dict[MoveKey, list[dict]] = {}
    paths = set(RESULTS.glob("*.jsonl"))
    if _ARCHIVE_RESULTS.is_dir():
        paths.update(_ARCHIVE_RESULTS.glob("*.jsonl"))
    for path in sorted(paths):
        try:
            records = path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeDecodeError:
            continue
        for raw in records:
            try:
                row = json.loads(raw)
                moves = _attempt_moves(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if tuple(moves[:len(prefix)]) != prefix or len(moves) <= len(prefix):
                continue
            candidate = moves[len(prefix)]
            if (len(prefix) % 2 == 0) != (side == "A"):
                raise AssertionError("Case side does not agree with its prefix")
            found.setdefault(candidate, []).append({
                "source": path.name,
                "id": row.get("id"),
                "teams": row.get("teams"),
                "winner": row.get("winner"),
                "reason": row.get("reason"),
            })
    return found


def _opening(case: Case, candidate: MoveKey) -> Opening:
    moves = tuple(_move_list(move) for move in (*case.prefix, candidate))
    return Opening(f"book-{case.name}-{_line_key((*case.prefix, candidate))}", list(moves), "")


def _seed_missing_predecessor(started, opening: Opening) -> None:
    """Make a one-ply attacker replay constructible for the next Defender.

    ``runner._player`` restores a previous board for both search players.
    After exactly one attacker move the replay has no defender predecessor;
    the initial board is the real predecessor of that defender turn.  The
    production Player identifies this as an opening reply, resets, and seeds
    exactly the initial/current hashes before it searches.
    """
    if len(opening.moves) == 1 and False not in started.last_board:
        started.last_board[False] = deepcopy(started.positions[0]["board"])


def _trial(case: Case, candidate: MoveKey, opponent: str, budget: float) -> dict:
    """Run a referee-checked continuation from either side's book move.

    ``Tablut.play`` always starts with an Attacker move.  Two of the target
    decisions leave Defenders to reply, so using it would silently swap the
    side to move.  This short loop is intentionally a literal replay of the
    referee's turn, move, clock, and terminal checks, beginning from the side
    that follows the supplied opening.
    """
    opening = _opening(case, candidate)
    attacker_name, defender_name = (
        ("Team19-v4", opponent) if case.side == "A" else (opponent, "Team19-v4")
    )
    game = ObservedReferee()
    started = start(opening, game)
    _seed_missing_predecessor(started, opening)
    attempts: list[dict] = []
    game.attempts = attempts
    game.prefix_plies = len(opening.moves)
    clock = OfficialClock()
    remaining = {"A": budget, "D": budget}
    next_attacker = len(opening.moves) % 2 == 0
    output = StringIO()
    seed = 20261126

    # Existing book code is excluded from a counterfactual continuation.  The
    # state after the candidate must be evaluated by frozen v4 search alone.
    with ExitStack() as stack:
        stack.enter_context(patch.object(Tablut, "TIME_BUDGET", budget))
        stack.enter_context(patch.dict(sys.modules, {"Team19_book": None}))
        player_a = _player(attacker_name, seed, started, True)
        player_d = _player(defender_name, seed, started, False)
        observers = {
            True: ObservedPlayer(attacker_name, player_a, attempts),
            False: ObservedPlayer(defender_name, player_d, attempts),
        }
        for observer in observers.values():
            observer.observe_search(stack)
        stack.enter_context(patch.object(Tablut, "time", clock))
        with redirect_stdout(output):
            while True:
                side = "A" if next_attacker else "D"
                before = deepcopy(game.board)
                started_at = clock.time()
                move = (observers[next_attacker].play_attacker(before)
                        if next_attacker else observers[next_attacker].play_defender(before))
                elapsed = clock.time() - started_at
                remaining[side] -= elapsed
                if remaining[side] < 0:
                    game.winner = "D" if next_attacker else "A"
                    output.write(f"{side} exceeded time limit!")
                    break
                if not game.make_move(next_attacker, move):
                    game.winner = "D" if next_attacker else "A"
                    output.write(f"Illegal move by {side}!")
                    break
                if game.check_if_game_over(not next_attacker):
                    break
                next_attacker = not next_attacker

    reason = _reason(game, output.getvalue())
    score = 1 if game.winner is None else 2 * (game.winner == case.side)
    return {
        "opponent": opponent, "scoreForV4": score, "winner": game.winner,
        "reason": reason, "remaining": remaining,
        "v4Fallbacks": sum(
            bool(attempt.get("fallback")) for attempt in attempts
            if attempt["team"] == "Team19-v4"
        ),
        "pliesAfterCandidate": len(attempts),
    }


def investigate(case: Case, *, seconds: float, deep_seconds: float, run: bool, budget: float,
                max_candidates: int, models: tuple[str, ...], trial_opponents: tuple[str, ...]) -> dict:
    board, attacker, history = _official_position(case.prefix)
    if attacker != (case.side == "A"):
        raise AssertionError(f"{case.name}: wrong side")
    legal = {_move_key(move) for move in engine.get_legal_moves(board, attacker)}
    history = _history_candidates(case.prefix, case.side)
    proposed = _v4_choice(case.prefix, seconds)
    model_choices = {}
    for team in models:
        try:
            model_choices[team] = _policy_choice(team, case.prefix, seconds)
        except Exception as error:
            model_choices[team] = f"{type(error).__name__}: {error}"
    candidate_sources = {proposed: "v4-native"}
    for move in sorted(history):
        candidate_sources.setdefault(move, "archive")
    for team, choice in model_choices.items():
        if isinstance(choice, tuple):
            candidate_sources.setdefault(choice, f"policy:{team}")
    candidates = [move for move in candidate_sources if move in legal][:max_candidates]
    reports = []
    for candidate in candidates:
        report = {
            "candidate": candidate,
            "line": (*case.prefix, candidate),
            "source": candidate_sources[candidate],
            "historic": history.get(candidate, []),
        }
        if deep_seconds:
            report["forcedChild"] = _forced_child(case, candidate, deep_seconds)
        if run:
            report["trials"] = [_trial(case, candidate, opponent, budget)
                                for opponent in trial_opponents]
        reports.append(report)
    return {
        "case": case.name, "side": case.side, "prefix": case.prefix,
        "v4NativeChoice": proposed, "legalCandidates": len(legal),
        "policyChoices": model_choices,
        "reports": reports,
    }


def investigate_root_finalist(case: Case, candidate: MoveKey, *, budget: float) -> dict:
    """Finish each fixed attacker root against both requested independent policies."""
    board, attacker, _ = _official_position(case.prefix)
    if case.prefix or not attacker or case.side != "A":
        raise AssertionError("root finalist must be an initial attacker decision")
    if _move_list(candidate) not in engine.get_legal_moves(board, True):
        raise ValueError(f"Illegal root finalist: {candidate}")
    historic = _history_candidates((), "A").get(candidate, [])
    return {
        "case": case.name,
        "side": "A",
        "prefix": (),
        "candidate": candidate,
        "line": (candidate,),
        "source": "book_screen_top3_worst_v4_score",
        "opponents": case.opponents,
        "historic": historic,
        "trials": [_trial(case, candidate, opponent, budget) for opponent in case.opponents],
    }


def main(argv=None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.name for case in KNOWN_CASES])
    parser.add_argument("--root-finalists", action="store_true",
                        help="finish fixed top-three root families versus v3 and QSearch-PVS")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="offline v4 analysis budget per decision")
    parser.add_argument("--run", action="store_true", help="play continuation trials")
    parser.add_argument("--budget", type=float, default=10.0,
                        help="per-player referee budget for continuation trials")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--deep-seconds", type=float, default=0.0,
                        help="per reply and recovery frozen-v4 refutation budget")
    parser.add_argument("--models", default="Team19-v3,Grandmaster-Plus,Grandmaster-Numba,QSearch-PVS",
                        help="comma-separated arena policies used to propose alternatives")
    parser.add_argument("--trial-opponents",
                        help="comma-separated continuation opponents, defaulting to the case opponent")
    parser.add_argument("--output", type=Path, help="optional JSON evidence path")
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.budget <= 0 or args.deep_seconds < 0 or args.max_candidates < 1:
        parser.error("seconds, budget and max-candidates must be positive")
    if args.root_finalists and args.case:
        parser.error("--root-finalists cannot be combined with --case")
    if args.root_finalists and not args.run:
        parser.error("--root-finalists requires --run")
    if args.root_finalists and args.trial_opponents:
        parser.error("--root-finalists always compares Team19-v3 and QSearch-PVS")
    cases = [case for case in KNOWN_CASES if args.case in (None, case.name)]
    models = tuple(token.strip() for token in args.models.split(",") if token.strip())
    for team in models:
        if team not in TEAMS:
            parser.error(f"Unknown arena team: {team}")
    selected_opponents = (tuple(token.strip() for token in args.trial_opponents.split(",")
                                if token.strip()) if args.trial_opponents else None)
    if selected_opponents is not None and any(team not in TEAMS for team in selected_opponents):
        parser.error("trial-opponents contains an unknown arena team")
    if args.root_finalists:
        evidence = {
            "format": 3,
            "mode": "fixed-root-finalists",
            "selection": {
                "screen": "book_screen_results.json",
                "criterion": "top three D4 families by worstV4Score",
                "opponents": _ROOT_FINALIST_OPPONENTS,
            },
            "trialBudget": args.budget,
            "ranTrials": True,
            "rootFinalists": [investigate_root_finalist(case, candidate, budget=args.budget)
                              for case, candidate in zip(root_finalist_cases(), ROOT_FINALISTS)],
        }
    else:
        evidence = {
            "format": 2, "searchSeconds": args.seconds, "deepSeconds": args.deep_seconds,
            "trialBudget": args.budget,
            "ranTrials": args.run,
            "cases": [investigate(case, seconds=args.seconds, deep_seconds=args.deep_seconds,
                                  run=args.run, budget=args.budget, max_candidates=args.max_candidates,
                                  models=models,
                                  trial_opponents=selected_opponents or case.opponents)
                      for case in cases],
        }
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
