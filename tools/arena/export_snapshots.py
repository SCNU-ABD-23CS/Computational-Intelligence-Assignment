"""Export historical Team19 code with only intra-package import changes."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import subprocess

from . import HERE


COMMITS = {
    "v_99671ae": "99671ae4dbe564992fff874ec8e1d89dbfffd8a0",
    "v_e3d64c4": "e3d64c4308d63be81453d691db8cb215368de645",
}
FILES = ("Team19.py", "Team19_engine.py", "Team19_eval.py", "Team19_search.py")


def package_imports(source: str) -> str:
    source = re.sub(r"(?m)^(\s*)from (Team19_\w+) import ",
                    r"\1from .\2 import ", source)
    source = re.sub(r"(?m)^(\s*)import (Team19_\w+)( as \w+)?$",
                    r"\1from . import \2\3", source)
    source = source.replace(
        "__import__(module_name, fromlist=[attribute])",
        "__import__(module_name, globals(), fromlist=[attribute], level=1)",
    )
    return re.sub(
        r'__import__\(("Team19_\w+"), fromlist=(\[[^\]]*\])\)',
        r"__import__(\1, globals(), fromlist=\2, level=1)",
        source,
    )


def export(repo: Path) -> None:
    for package, commit in COMMITS.items():
        path = HERE / "snapshots" / package
        path.mkdir(exist_ok=True)
        (path / "__init__.py").write_text('"""Isolated Team19 snapshot."""\n', encoding="utf-8")
        hashes = {}
        for name in FILES:
            raw = subprocess.check_output(("git", "show", f"{commit}:{name}"), cwd=repo)
            source = package_imports(raw.decode("utf-8").replace("\r\n", "\n"))
            compile(source, name, "exec")
            (path / name).write_text(source, encoding="utf-8", newline="\n")
            hashes[name] = hashlib.sha256(raw).hexdigest()
        (path / "source.json").write_text(
            json.dumps({"commit": commit, "original_sha256": hashes}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    export(parser.parse_args().repo)
