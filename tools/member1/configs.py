"""Freeze isolated configuration packages for the fairness-controlled blind test.

Every blind-evaluation group must differ from the others in exactly one thing:
the frozen v4 baseline, the same baseline plus the opening book, the same
baseline plus the book and the residual NNUE. Copying the root ``Team19*.py``
files into a private package and rewriting only the intra-package imports gives
three configurations that can run side by side in one process without touching
the working tree, and records the SHA-256 of every source file so the report can
prove which code produced which score.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

from tools.arena.export_snapshots import package_imports


REPO = Path(__file__).resolve().parents[2]
CONFIGS = Path(__file__).with_name("frozen")
CORE = ("Team19.py", "Team19_engine.py", "Team19_eval.py", "Team19_search.py",
        "Team19_tt.py", "Team19_tactics.py", "Team19_weights.py")
OPTIONAL = ("Team19_book.py", "Team19_nnue.py")


@dataclass(frozen=True)
class Config:
    label: str
    directory: Path
    files: tuple[str, ...]
    sha256: dict[str, str]

    @property
    def module(self) -> str:
        return f"..member1.frozen.{self.label}.Team19"

    @property
    def search(self) -> str:
        return f"..member1.frozen.{self.label}.Team19_search"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(label: str, *, source: Path = REPO, extras: tuple[str, ...] = (),
          configs: Path = CONFIGS) -> Config:
    missing = [name for name in CORE if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing submission files in {source}: {missing}")
    for name in extras:
        if name not in OPTIONAL:
            raise ValueError(f"{name} is not an optional member module")
        if not (source / name).is_file():
            raise FileNotFoundError(f"missing {name} in {source}")
    target = configs / label
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    (target / "__init__.py").write_text(
        '"""Freeze of one blind-evaluation configuration."""\n', encoding="utf-8")
    files = CORE + extras
    hashes = {}
    for name in files:
        original = source / name
        source_text = original.read_text(encoding="utf-8").replace("\r\n", "\n")
        adjusted = package_imports(source_text)
        compile(adjusted, name, "exec")
        (target / name).write_text(adjusted, encoding="utf-8", newline="\n")
        hashes[name] = _digest(original)
    record = {"label": label, "source": str(source), "files": list(files),
              "sha256": hashes, "extras": list(extras)}
    (target / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return Config(label, target, files, hashes)


def load(label: str, configs: Path = CONFIGS) -> Config:
    directory = configs / label
    record = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    files = tuple(record["files"])
    for name in files:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"{directory / name} is missing")
    hashes = {name: _digest(REPO / name) if (REPO / name).is_file() else record["sha256"][name]
              for name in files}
    changed = {name: (record["sha256"][name], hashes[name]) for name in files
               if record["sha256"][name] != hashes[name]}
    if changed:
        raise ValueError(f"config {label} no longer matches its sources: {changed}. Rebuild it.")
    return Config(label, directory, files, record["sha256"])


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--extra", action="append", default=[],
                        help=f"optional member module, one of {OPTIONAL}")
    parser.add_argument("--source", type=Path, default=REPO)
    args = parser.parse_args()
    built = build(args.label, source=args.source, extras=tuple(args.extra))
    print(f"{built.label}: {len(built.files)} files -> {built.directory}")
    for name, digest in built.sha256.items():
        print(f"  {name:<22} {digest[:16]}")
