"""Run the v2 seats of the archived standard and fast schedules."""

from argparse import ArgumentParser
from collections import Counter

from . import openings, report, runner
from .overview import audit_game


VERSION = "Team19-v2"


def schedule(segment):
    pairs = [(VERSION, opponent) for opponent in runner.EXTERNAL]
    if segment == "official":
        return runner.paired(pairs, [openings.standard()], 60.0)
    return runner.paired(pairs, openings.load_suite()[:6], 10.0)


def run(segment, resume):
    stage = f"evolution-{segment}"
    matches = schedule(segment)
    path = runner.run_schedule(matches, stage, stage, resume=resume)
    games = report.load(path)
    expected = {match.id: match for match in matches}
    if {row["id"] for row in games} != expected.keys():
        raise ValueError(f"The {stage} matches are incomplete")
    plies = sum(audit_game(row, expected[row["id"]]) for row in games)
    scores = Counter()
    for row in games:
        seat = next(seat for seat, name in row["teams"].items() if name == VERSION)
        scores[seat] += report.points(row["winner"], seat)
    report.write_report(path)
    print(f"{stage} | {len(games)} games and {plies} verified plies | "
          f"{scores['A']} attack + {scores['D']} defence points", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("official", "fast"))
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        for segment in (args.stage,) if args.stage else ("official", "fast"):
            run(segment, args.resume)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
