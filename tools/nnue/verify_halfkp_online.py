"""Run the HalfKP CPU release gates and write their measured results."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import ctypes
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
import platform
import shutil
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numba
import numpy as np

import Tablut
import Team19
import Team19_engine as engine
import Team19_halfkp as halfkp
import Team19_search as search
from tests.test_nnue_online import SCRIPTED_OPENING, fixed_iteration, opening_positions
from .common import write_json


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "tools/nnue/runs/halfkp_v1"
VALIDATION = ROOT / "tools/nnue/results/volume_halfkp/team19_nnue_val.npz"
ALIGNMENT_LIMIT = 5e-4
NPS_FLOOR = 0.65
NPS_TARGET = 0.75


def zero_head_audit() -> dict:
    baseline = search._NativeSearch(Team19._INITIAL_BOARD)
    work = halfkp.make_halfkp_work(search._NATIVE_PLY_SLOTS, zero_head=True)
    cleared = search._NativeSearch(Team19._INITIAL_BOARD, halfkp_work=work)
    positions = opening_positions()
    cases = nodes = 0
    for board, side in positions:
        for depth in (1, 2, 3):
            before = fixed_iteration(baseline, board, side, depth)
            after = fixed_iteration(cleared, board, side, depth)
            if before[:2] != after[:2] or not np.array_equal(before[2], after[2]):
                raise AssertionError(f"Zero-head search diverged at position {cases // 3}, D{depth}")
            nodes += int(before[2][search._NODE_COUNT])
            cases += 1
    return {"positions": len(positions), "depths": [1, 2, 3], "matching_searches": cases,
            "nodes_per_version": nodes, "identical_score_move_and_counters": True}


def incremental_audit(steps: int = 320) -> dict:
    """Replay legal referee games and check both the child and the restored parent."""
    work = halfkp.make_halfkp_work(search._NATIVE_PLY_SLOTS)
    rng = np.random.default_rng(20261127)
    moves = np.empty(engine._MOVE_CAPACITY, dtype=np.int16)
    squares = np.empty(4, dtype=np.int16)
    pieces = np.empty(4, dtype=np.int8)
    reference = np.empty(halfkp.HIDDEN, dtype=np.float32)
    counts = dict(ordinary=0, single_captures=0, multiple_captures=0,
                  king_moves=0, throne_departures=0, king_captures=0)
    maximum = 0.0
    games = 0
    referee = None
    for index in range(steps):
        if referee is None:
            referee = Tablut.Tablut()
            referee.past_positions = [referee.compute_zobrist_hash(True)]
            attacker, ply = True, 0
            games += 1
        board = engine._flat(referee.board)
        ha, hd, a, d, king = engine._describe(board)
        if not ply or ply == len(work.ksq_stack) - 1:
            halfkp._halfkp_root(board, king, work)
            ply = 0
        before = board.copy()
        parent = work.acc_stack[ply].copy()
        count = engine._legal(board, attacker, moves)
        hashes = np.uint64(ha), np.uint64(hd)
        if index < len(SCRIPTED_OPENING):
            start, target = SCRIPTED_OPENING[index]
            move = (start[0] * 9 + start[1]) * 81 + target[0] * 9 + target[1]
        else:
            move = int(moves[int(rng.integers(count))])
            if not counts["multiple_captures"]:
                # Prioritize a legally available double capture so the audit
                # exercises all victim rows, even with a short replay.
                for candidate in moves[:count]:
                    trial = int(candidate)
                    *_, victims = engine._push(board, trial, *hashes, a, d, king, squares, pieces)
                    engine._pop(board, trial, squares, pieces, victims)
                    if victims > 1:
                        move = trial
                        break
        source, target = divmod(move, 81)
        _, _, _, _, new_king, captured = engine._push(
            board, move, *hashes, a, d, king, squares, pieces,
        )
        halfkp._halfkp_push(board, source, target, new_king, ply, squares, pieces, captured, work)
        halfkp._halfkp_full(board, new_king, work.embedding, work.parameters, reference)
        maximum = max(maximum, float(np.max(np.abs(reference - work.acc_stack[ply + 1]))))
        if work.ksq_stack[ply + 1] != new_king:
            raise AssertionError("HalfKP saved the wrong king square")
        if not referee.make_move(attacker, engine._move_list(move)):
            raise AssertionError("Referee rejected an audit move")
        np.testing.assert_array_equal(board, engine._flat(referee.board))
        engine._pop(board, move, squares, pieces, captured)
        np.testing.assert_array_equal(board, before)
        np.testing.assert_array_equal(work.acc_stack[ply], parent)
        if work.ksq_stack[ply] != king:
            raise AssertionError("Undo corrupted the parent king square")
        moved_king = before[source] == engine._K
        counts["ordinary"] += int(not moved_king and not captured)
        counts["single_captures"] += int(captured == 1)
        counts["multiple_captures"] += int(captured > 1)
        counts["king_moves"] += int(moved_king)
        counts["throne_departures"] += int(moved_king and source == engine._CASTLE)
        counts["king_captures"] += int(new_king < 0)
        ply += 1
        attacker = not attacker
        with redirect_stdout(StringIO()):
            ended = referee.check_if_game_over(attacker)
        if ended:
            referee = None
    return {"steps": steps, "games": games, **counts, "max_abs_accumulator_error": maximum}


def alignment_audit(weights_path: Path, validation_path: Path) -> dict:
    from Team19_halfkp_weights import SOURCE_SHA256
    from .halfkp_model import load_weights, numpy_residual

    digest = sha256(weights_path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Validation must use the exact exported Stage 3 NPZ")
    weights = load_weights(weights_path)
    with np.load(validation_path, allow_pickle=False) as data:
        boards = data["boards"]
    if boards.shape != (3166, 81) or not np.all((boards == engine._K).sum(axis=1) == 1):
        raise ValueError("Expected 3,166 locked validation positions with one king each")
    work = halfkp.make_halfkp_work(search._NATIVE_PLY_SLOTS)
    actual = np.empty((2, len(boards)), dtype=np.float32)
    for index, (board, king) in enumerate(zip(boards, (boards == engine._K).argmax(axis=1))):
        halfkp._halfkp_root(board, int(king), work)
        for side in (0, 1):
            actual[side, index] = halfkp._halfkp_residual(bool(side), 0, work)
    by_head = {}
    for side, label in ((0, "defender"), (1, "attacker")):
        expected = numpy_residual(weights, boards, np.full(len(boards), side, dtype=np.int8))
        difference = np.abs(actual[side] - expected)
        by_head[label] = {"index": side, "max_abs_error": float(difference.max()),
                          "p99_abs_error": float(np.quantile(difference, 0.99))}
    return {"positions_per_head": len(boards), "total_predictions": 2 * len(boards),
            "validation_sha256": sha256(validation_path.read_bytes()).hexdigest(),
            "heads": by_head, "max_abs_error": max(head["max_abs_error"] for head in by_head.values())}


def _measure_nps(bot, boards, side, seconds: float) -> dict:
    nodes = evaluations = iterations = 0
    started = perf_counter()
    while True:
        for board in boards:
            # The next iteration must search its static leaves. Reusing the
            # previous search's root TT would measure almost only TT cutoffs.
            bot.reset_game()
            _, _, counts = fixed_iteration(bot, board, side)
            nodes += int(counts[search._NODE_COUNT])
            evaluations += int(counts[search._EVALUATIONS])
            iterations += 1
        elapsed = perf_counter() - started
        if elapsed >= seconds:
            return {"nodes": nodes, "evaluations": evaluations, "seconds": elapsed,
                    "nps": nodes / elapsed, "iterations": iterations}


@contextmanager
def _single_logical_cpu():
    if sys.platform != "win32":
        yield None
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    mask_type = ctypes.c_size_t
    kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(mask_type),
                                             ctypes.POINTER(mask_type))
    kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, mask_type)
    process = kernel.GetCurrentProcess()
    available, system = mask_type(), mask_type()
    if not kernel.GetProcessAffinityMask(process, ctypes.byref(available), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    selected = available.value & -available.value
    if not kernel.SetProcessAffinityMask(process, selected):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield selected.bit_length() - 1
    finally:
        if not kernel.SetProcessAffinityMask(process, available.value):
            raise ctypes.WinError(ctypes.get_last_error())


def nps_audit(seconds: float) -> dict:
    """Compare paired, warmed D3 iterations in alternating order on one thread."""
    numba.set_num_threads(1)
    with _single_logical_cpu() as cpu:
        return {"logical_cpu": cpu, **_paired_nps(seconds)}


def _paired_nps(seconds: float) -> dict:
    started = perf_counter()
    baseline = search._NativeSearch(Team19._INITIAL_BOARD)
    base_init = perf_counter() - started
    started = perf_counter()
    candidate = search._NativeSearch(Team19._INITIAL_BOARD, use_halfkp=True)
    candidate_init = perf_counter() - started
    positions = opening_positions()
    results = {}
    for side, label in ((True, "attacker"), (False, "defender")):
        boards = [board for board, attacker in positions if attacker == side]
        measurements = []
        for trial in range(3):
            pair = ((baseline, "v4"), (candidate, "halfkp"))
            if trial % 2:
                pair = pair[::-1]
            record = {name: _measure_nps(bot, boards, side, seconds) for bot, name in pair}
            record["ratio"] = record["halfkp"]["nps"] / record["v4"]["nps"]
            measurements.append(record)
        results[label] = {
            "v4_nps": median(record["v4"]["nps"] for record in measurements),
            "halfkp_nps": median(record["halfkp"]["nps"] for record in measurements),
            "ratio": median(record["ratio"] for record in measurements), "rounds": measurements,
        }
    geo = math.sqrt(results["attacker"]["ratio"] * results["defender"]["ratio"])
    return {
        **results, "geometric_mean_ratio": geo, "rounds_per_side": 3, "workers": 1,
        "fixed_depth": 3, "tt_reset_each_position": True,
        "numba_threads": numba.get_num_threads(), "minimum_sample_seconds": seconds,
        "v4_setup_seconds": base_init, "halfkp_setup_seconds": candidate_init,
        "v4_warmup_seconds": baseline.warmup_seconds,
        "halfkp_warmup_seconds": candidate.warmup_seconds,
    }


def read_only_audit() -> dict:
    """Launch only root submission files under a write-denying Python audit hook."""
    script = r"""
import json
import os
import sys
sys.path.insert(0, os.getcwd())
def forbid_writes(event, args):
    if event == "open" and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        raise PermissionError(f"Write attempted: {args[0]}")
    if event in ("os.mkdir", "os.rmdir", "os.remove", "os.rename", "subprocess.Popen"):
        raise PermissionError(f"Unexpected side effect: {event}")
sys.addaudithook(forbid_writes)
import Tablut
import Team19
import Team19_engine
from Team19_search import _native_visit
Tablut.TIME_BUDGET = 1.0
player = Team19.Player(use_halfkp=True)
assert player.native_init_error is None, player.native_init_error
signatures = tuple(_native_visit.signatures)
board = Tablut.Tablut().board
move = player.play_attacker(board)
assert move in Team19_engine.get_legal_moves(board, True)
assert player.last_error is None and not player.last_search.fallback
assert signatures and tuple(_native_visit.signatures) == signatures
assert not any(name.split(".")[0] in ("tools", "torch") for name in sys.modules)
print(json.dumps({"legal_move": move, "depth": player.last_search.depth,
                  "warmup_seconds": player.warmup_seconds, "late_compilation": False,
                  "tools_or_torch_imported": False, "write_denied_during_smoke": True}))
"""
    with TemporaryDirectory(prefix="team19-halfkp-") as temporary:
        location = Path(temporary)
        for path in [ROOT / "Tablut.py", *ROOT.glob("Team19*.py")]:
            copied = shutil.copyfile(path, location / path.name)
            Path(copied).chmod(0o444)
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=location,
                                   capture_output=True, text=True, check=True, timeout=180)
    return json.loads(completed.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--weights", type=Path, default=RUN / "stage3.npz")
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--output", type=Path, default=RUN / "online_verification.json")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    artifact = ROOT / "Team19_halfkp_weights.py"
    report = {
        "schema": 1, "platform": platform.platform(), "processor": platform.processor(),
        "source": {"file": args.weights.name, "sha256": sha256(args.weights.read_bytes()).hexdigest()},
        "export": {"file": artifact.name, "bytes": artifact.stat().st_size,
                   "sha256": sha256(artifact.read_bytes()).hexdigest()},
    }
    for name, audit in (
        ("nps", lambda: nps_audit(args.seconds)), ("zero_head", zero_head_audit),
        ("incremental", incremental_audit),
        ("alignment", lambda: alignment_audit(args.weights, args.validation)),
        ("read_only", read_only_audit),
    ):
        report[name] = audit()
        print(name, json.dumps(report[name]), flush=True)
    report["gates"] = {
        "zero_head_exact": report["zero_head"]["identical_score_move_and_counters"],
        "incremental_coverage": report["incremental"]["steps"] >= 200 and all(
            report["incremental"][key] > 0 for key in (
                "ordinary", "single_captures", "multiple_captures", "king_moves", "throne_departures",
            )
        ),
        "incremental_precision": report["incremental"]["max_abs_accumulator_error"] < 1e-4,
        "alignment_precision": report["alignment"]["max_abs_error"] < ALIGNMENT_LIMIT,
        "nps_hard_floor": all(report["nps"][side]["ratio"] >= NPS_FLOOR
                              for side in ("attacker", "defender")),
        "read_only_cpu": report["read_only"]["write_denied_during_smoke"],
    }
    report["nps"]["meets_target_both_sides"] = all(
        report["nps"][side]["ratio"] >= NPS_TARGET for side in ("attacker", "defender")
    )
    write_json(args.output, report)
    print(f"verification={args.output.resolve()}", flush=True)
    if not all(report["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
