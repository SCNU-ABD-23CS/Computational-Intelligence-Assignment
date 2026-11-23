"""Play official games with the submission in an isolated read-only directory."""

from argparse import ArgumentParser
import csv
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory

from . import HERE


_ISOLATED = r"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from time import perf_counter

root = Path.cwd()
sys.path.insert(0, str(root))
assert not (root / "tools").exists()
def write_denied(path):
    try:
        with path.open("ab"):
            pass
    except PermissionError:
        return
    raise AssertionError(f"The isolated workspace allowed writing {path.name}")

write_denied(root / "write-probe")
modules = sorted(root.glob("Team19*.py"))
assert modules and (root / "Tablut.py").is_file()
for file in [*modules, root / "Tablut.py"]:
    write_denied(file)

import Tablut
import Team19

assert Path(Tablut.__file__).resolve().parent == root

class Observer:
    def __init__(self):
        self.player = Team19.Player()
        assert self.player._native is not None, self.player.native_init_error
        self.depths = []

    def play(self, callback, board):
        move = callback(board)
        result = self.player.last_search
        assert self.player.last_error is None and result and not result.fallback
        self.depths.append(result.depth)
        return move

    def play_attacker(self, board):
        return self.play(self.player.play_attacker, board)

    def play_defender(self, board):
        return self.play(self.player.play_defender, board)

results = []
for budget in (10.0, 60.0):
    Tablut.TIME_BUDGET = budget
    started = perf_counter()
    attacker, defender = Observer(), Observer()
    referee = Tablut.Tablut()
    output = StringIO()
    with redirect_stdout(output):
        referee.play(attacker, defender)
    lines = output.getvalue().splitlines()
    if any(word in output.getvalue() for word in ("exceeded time limit", "Illegal move by", "Exception caught")):
        raise AssertionError("\n".join(lines[-8:]))
    depths = attacker.depths + defender.depths
    assert attacker.depths and defender.depths and min(depths) >= 1
    results.append({"budget": budget, "winner": referee.winner, "plies": len(depths),
                    "lastMessages": lines[-2:], "elapsed": perf_counter() - started,
                    "minDepth": min(depths), "meanDepth": sum(depths) / len(depths),
                    "fallbacks": 0, "exceptions": 0, "timeouts": 0, "illegalMoves": 0})

names = sorted(name for name in sys.modules if name.startswith("Team19"))
assert all(Path(sys.modules[name].__file__).resolve().parent == root for name in names)
assert not any(name == "tools" or name.startswith("tools.") for name in sys.modules)
cache_files = list(root.rglob("*.pyc")) + list(root.rglob("*.nbc")) + list(root.rglob("*.nbi"))
assert not cache_files, cache_files
print(json.dumps({"directory": str(root), "writeAccessDenied": True, "toolsPresent": False,
                  "localModules": names, "cacheFiles": len(cache_files), "games": results}))
"""


def _windows_acl(directory, deny):
    identity = subprocess.check_output(
        ("whoami", "/user", "/fo", "csv", "/nh"), text=True,
    )
    sid = next(csv.reader(io.StringIO(identity)))[1]
    if deny:
        command = ("icacls", str(directory), "/deny", f"*{sid}:(OI)(CI)(WD,AD,WEA,WA)")
    else:
        command = ("icacls", str(directory), "/remove:d", f"*{sid}", "/T")
    subprocess.run(command, check=True, capture_output=True)


def _set_read_only(directory, deny):
    if os.name == "nt":
        _windows_acl(directory, deny)
    else:
        if not deny:
            directory.chmod(stat.S_IRWXU)
        for path in directory.iterdir():
            path.chmod(0o444 if deny else 0o600)
        if deny:
            directory.chmod(0o555)


def verify(repo, scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    scratch = scratch.resolve(strict=True)
    with TemporaryDirectory(prefix="tablut-readonly-", dir=scratch) as temp:
        directory = Path(temp).resolve()
        if directory.parent != scratch:
            raise RuntimeError("The isolated copy is outside the selected scratch directory")
        for path in sorted(repo.glob("Team19*.py")):
            shutil.copyfile(path, directory / path.name)
        # The grading framework supplies the referee separately from the Team19 modules.
        shutil.copyfile(repo / "Tablut.py", directory / "Tablut.py")
        try:
            _set_read_only(directory, True)
            output = subprocess.check_output(
                (sys.executable, "-I", "-B", "-c", _ISOLATED), cwd=directory, text=True,
            )
        finally:
            _set_read_only(directory, False)
    return json.loads(output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    parser.add_argument("--scratch", type=Path, default=HERE / "results")
    args = parser.parse_args()
    result = verify(args.repo, args.scratch)
    path = HERE / "results" / "team19-v2-readonly.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
