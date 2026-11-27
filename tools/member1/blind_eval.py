"""Run the blind evaluation on the frozen unseen branches (member 1, task C).

Three groups are defined by the *same* baseline code and differ by exactly one
component, so a score difference can only come from that component:

* ``base``  frozen v4, no book, no NNUE            (baseline)
* ``book``  frozen v4 + pure-search opening book   (fallback submission)
* ``nnue``  frozen v4 + book + residual NNUE       (ceiling candidate)

All groups play the same unseen branches, the same opponent pool, the same
budgets, one worker, and both colours on every branch.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from tools.arena import runner
from tools.arena.runner import Match, Team, paired, run_schedule
from tools.member1 import branches as branchlib
from tools.member1.configs import CONFIGS, OPTIONAL, load


# One group per increment of the optional modules, so the three arms cannot drift
# apart from the freeze list they are built from.
GROUPS = {"base": (), "book": OPTIONAL[:1], "nnue": OPTIONAL}


def register(label: str) -> str:
    """Expose one frozen configuration as an arena entrant."""
    config = load(label, CONFIGS)
    name = f"M1-{label}"
    runner.TEAMS[name] = Team(config.module, label, f"成员1 盲评配置 {label}",
                              config.search)
    runner.TEAM_NAMES[label.casefold()] = name
    return name


def plan(label: str, opponents: tuple[str, ...], *, branch_file: Path,
         budget: float, per_branch: int) -> list[Match]:
    suite = branchlib.load(branch_file)
    openings = [branch.opening() for branch in suite[:per_branch]]
    if len(openings) != per_branch:
        raise ValueError(f"suite has only {len(suite)} branches, asked for {per_branch}")
    me = register(label)
    pairs = [(me, runner.TEAM_NAMES[opponent.casefold()] if opponent.casefold()
              in runner.TEAM_NAMES else opponent) for opponent in opponents]
    return paired(pairs, openings, budget)


def main(argv=None) -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=tuple(GROUPS))
    parser.add_argument("--opponents", default="v3,gmp,gm,qs",
                        help="comma separated arena aliases, or another M1 label")
    parser.add_argument("--branches", type=Path,
                        default=Path(__file__).with_name("blind_branches.json"))
    parser.add_argument("--branch-count", type=int, default=24,
                        help="how many frozen branches to use (24 by default)")
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

    opponents = tuple(token.strip() for token in args.opponents.split(",") if token.strip())
    opponents = tuple(register(token) if token in GROUPS else token for token in opponents)
    matches = plan(args.config, opponents, branch_file=args.branches,
                   budget=args.budget, per_branch=args.branch_count)
    run_id = args.run_id or f"m1-blind-{args.config}-{int(args.budget)}s"
    if args.dry_run:
        print(f"{run_id}: {len(matches)} games · workers=1")
        for name in sorted({match.attacker for match in matches}
                           | {match.defender for match in matches}):
            print("  " + runner.TEAMS[name].alias + " " + name)
        print(f"  results -> {runner.RESULTS / (run_id + '.jsonl')}")
        return
    path = runner.RESULTS / f"{run_id}.jsonl"
    if path.exists() and not (args.resume or args.fresh):
        raise SystemExit(f"{path.name} exists, use --resume or --fresh")
    if args.fresh and path.exists():
        path.unlink()
    path = run_schedule(matches, run_id, "m1-blind", resume=True, limit=args.limit)
    print(f"写入 {path}")


if __name__ == "__main__":
    main()
