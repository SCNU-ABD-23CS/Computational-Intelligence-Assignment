# Local Tablut book corpus audit

Corpus: `results`

- JSONL files: 42; records: 3406; canonical records: 3045
- Archived records: 336; duplicate replays: 361
- Replay JSON files: 135; parse errors: 0
- Symmetry classes observed among opening labels: 25

## Strong-engine outcomes

- **Team19-v4**: draw=11, loss=85, win=185
- **Team19-v3**: draw=23, loss=221, win=339
- **Grandmaster-Plus**: draw=30, loss=230, win=233
- **Grandmaster-Numba**: draw=72, loss=423, win=542
- **QSearch-PVS**: draw=34, loss=229, win=210

## Front-four exact sequences

- `10-o01-Team19-v2-A-Grandmaster-Numba` {'A': 'Team19-v2', 'D': 'Grandmaster-Numba'} winner=A reason=king captured moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Grandmaster-Numba-A-Team19-v2` {'A': 'Grandmaster-Numba', 'D': 'Team19-v2'} winner=None reason=repeated position moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Team19-v2-A-QSearch-PVS` {'A': 'Team19-v2', 'D': 'QSearch-PVS'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-QSearch-PVS-A-Team19-v2` {'A': 'QSearch-PVS', 'D': 'Team19-v2'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Grandmaster-Numba-A-QSearch-PVS` {'A': 'Grandmaster-Numba', 'D': 'QSearch-PVS'} winner=A reason=king captured moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-QSearch-PVS-A-Grandmaster-Numba` {'A': 'QSearch-PVS', 'D': 'Grandmaster-Numba'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Grandmaster-Numba-A-Team19-91ffcc8` {'A': 'Grandmaster-Numba', 'D': 'Team19-91ffcc8'} winner=A reason=king captured moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Team19-91ffcc8-A-Grandmaster-Numba` {'A': 'Team19-91ffcc8', 'D': 'Grandmaster-Numba'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Grandmaster-Numba-A-BFS-Strategist` {'A': 'Grandmaster-Numba', 'D': 'BFS-Strategist'} winner=A reason=king captured moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-BFS-Strategist-A-Grandmaster-Numba` {'A': 'BFS-Strategist', 'D': 'Grandmaster-Numba'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Grandmaster-Numba-A-Guided-MCTS` {'A': 'Grandmaster-Numba', 'D': 'Guided-MCTS'} winner=None reason=repeated position moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Guided-MCTS-A-Grandmaster-Numba` {'A': 'Guided-MCTS', 'D': 'Grandmaster-Numba'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-QSearch-PVS-A-Team19-91ffcc8` {'A': 'QSearch-PVS', 'D': 'Team19-91ffcc8'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Team19-91ffcc8-A-QSearch-PVS` {'A': 'Team19-91ffcc8', 'D': 'QSearch-PVS'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-QSearch-PVS-A-BFS-Strategist` {'A': 'QSearch-PVS', 'D': 'BFS-Strategist'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-BFS-Strategist-A-QSearch-PVS` {'A': 'BFS-Strategist', 'D': 'QSearch-PVS'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-QSearch-PVS-A-Guided-MCTS` {'A': 'QSearch-PVS', 'D': 'Guided-MCTS'} winner=A reason=king captured moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `10-o01-Guided-MCTS-A-QSearch-PVS` {'A': 'Guided-MCTS', 'D': 'QSearch-PVS'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `60-o01-Team19-v2-A-Grandmaster-Numba` {'A': 'Team19-v2', 'D': 'Grandmaster-Numba'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `60-o01-Grandmaster-Numba-A-Team19-v2` {'A': 'Grandmaster-Numba', 'D': 'Team19-v2'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `60-o01-Team19-v2-A-QSearch-PVS` {'A': 'Team19-v2', 'D': 'QSearch-PVS'} winner=A reason=king captured moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `60-o01-QSearch-PVS-A-Team19-v2` {'A': 'QSearch-PVS', 'D': 'Team19-v2'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `60-o01-Grandmaster-Numba-A-QSearch-PVS` {'A': 'Grandmaster-Numba', 'D': 'QSearch-PVS'} winner=A reason=king captured moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]
- `60-o01-QSearch-PVS-A-Grandmaster-Numba` {'A': 'QSearch-PVS', 'D': 'Grandmaster-Numba'} winner=D reason=king escaped to the edge moves=[[[8, 5], [8, 6]], [[2, 4], [2, 6]], [[4, 7], [1, 7]], [[2, 6], [2, 3]]]

## Interpretation

Counts describe the available schedules. Equal team labels, budget, initial state, prefix, played moves and result count once across filenames and fixture IDs. Labels may span revisions. Opponent pools and budgets differ between teams.
