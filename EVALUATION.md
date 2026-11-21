# Group 19 board evaluation

`Team19_eval.evaluate_board(board, weights=None)` returns one finite floating
point score for a 9 by 9 Tablut board. Positive values favour Attackers and
negative values favour Defenders. It does not mutate either argument and does
not import the rules engine or the search module.

The rules engine reports a captured or escaped King, insufficient Attackers,
and no legal moves. Search checks repeated states in both its current path and
the real-game history, then maps every reported outcome to `+100000.0`,
`-100000.0`, or `0.0` before it asks for a heuristic.

## Feature and weight contract

`FEATURE_NAMES` fixes the seven positions in every weight vector. Each feature
is in `[-1, 1]`; therefore the allowed `[-10000, 10000]` coefficient range
keeps an ordinary score in `(-100000, 100000)`.

| Index | Name | Formula | Positive direction |
| --- | --- | --- | --- |
| 0 | `material_balance` | `min(A,16)/16 - min(D,8)/8` | Attacker material |
| 1 | `low_attacker_penalty` | `-max(0, 5-min(A,16))/5` | Fewer low-material penalties |
| 2 | `king_escape_routes` | `-R/4` | Fewer direct routes to an empty edge |
| 3 | `king_edge_proximity` | `-(1-min(distance_to_edge,4)/4)` | King farther from an edge |
| 4 | `king_capture_pressure` | See capture geometry below | Stronger static capture setup |
| 5 | `king_guard_cover` | `-(0.55 * L/16 + C * S/4)` | Fewer King landings and active shields |
| 6 | `perimeter_blockade` | `sum((9 - d) * factor)/32` | More control of the King's escape lanes |

`R` is the number of orthogonal rays from an interior King that reach an empty
edge square without a piece. An empty castle `X` is traversable, just as it is
in the official movement rule. An already-edge King receives zero route value
because terminal adjudication belongs to search.

`L` counts empty squares the King can reach along the four rays. The empty
castle can be crossed but is not a landing. `S` counts adjacent Defenders that
shield an axis by standing immediately across the King from an adjacent
Attacker. `C` is 0.45 on or beside the castle and 0.18 elsewhere. An ordinary
shield prevents capture-pressure credit along that axis. With the default
weights, closing from distance two to one still gains 350 points from
blockade and reduced King mobility, against 315 points of ordinary shield
credit. A guard may reduce mobility by blocking a ray.

Blockade considers the first Attacker at distance `d` along each ray. Its
factor is 1 for a direct blocker and 0.25 when it stands behind a Defender.
An Attacker behind a guard contributes through blockade. On that ray,
moving the nearest Attacker closer increases the default score by
`3500 * 0.25 / 32` per square when the other features are fixed.
Attackers off the King's rays do not earn points just for standing on the
edge. Weight vectors tuned against the previous formulas should be
evaluated again.

Capture pressure follows the official King locations. At the centre it is the
four adjacent Attackers divided by four. Adjacent to the castle it counts the
three non-castle neighbours divided by three. Elsewhere it takes the stronger
of the vertical and horizontal adjacent pairs, divided by two. A normal pair is
considered only when both cells are Attackers or empty, so a Defender or castle
already occupying a proposed sandwich slot removes that axis from this static
estimate. Edge Kings receive zero pressure because they are terminal positions.

The hand-set default vector is:

```python
[8500.0, 7500.0, 10000.0, 9000.0, 10000.0, 7000.0, 3500.0]
```

`weights=None` uses the public `DEFAULT_WEIGHTS` list. An explicit vector must
have exactly seven finite numeric entries in the documented range. Numeric
strings are rejected so configuration mistakes do not silently become weights.
Set one entry to zero for a feature ablation and pass a copy when an experiment
needs a different vector.

If a diagnostic board lacks a King, King-dependent features are zero and the
function still returns a bounded finite material score. Valid games
will be resolved as terminal by the rules engine before that fallback matters.

## Verification and handoff

Run the evaluator and search suites with:

```powershell
python -m unittest discover -s tests -v
```

For a representative local timing measurement, use a valid midgame board and
repeat `evaluate_board(board)` with `timeit`; do not treat a wall-clock threshold
as a correctness test. The implementation performs one board scan and a small
constant number of ray or neighbour checks.

Member 5 should use paired role-swapped games with a fixed opponent pool and
fixed search depth when optimising this vector. Compare a candidate against the
default weights on held-out paired matches. For ablation, set exactly one weight
to zero, leave all other coefficients unchanged, and report the paired result.
