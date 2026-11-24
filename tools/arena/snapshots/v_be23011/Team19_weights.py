"""Named weights for the compiled Team19 search.

The seven-feature Python evaluation keeps its own defaults. This independent
vector was selected in fixed-depth training and adopted after the 60-second
and 10-second held-out comparisons.
"""

import numpy as np


FEATURE_NAMES = (
    "attacker_material_opening", "attacker_material_endgame", "defender_material",
    "low_attacker_material", "escape_trapped", "escape_one", "escape_two",
    "escape_three", "escape_four", "escape_five_plus", "king_mobility",
    "king_edge_proximity", "direct_escape_defender_turn", "direct_escape_attacker_turn",
    "attacker_pst_interior", "attacker_pst_edge", "attacker_near_king",
    "attacker_adjacent_king", "guard_near_king", "guard_mobility",
    "blocked_king_directions", "king_ray_pressure",
)

DEFAULT_WEIGHTS = (
    645.3554830625503, 569.1597576528295, -1402.4989923847843,
    -506.3488399770489, 2754.4309586019413, -9262.94514422622,
    -3476.1412520412737, -2274.0384791347997, -1245.952331313671,
    23.2330505783072, -220.5403475366551, -433.65441038198753,
    -573.0917069731438, -208.74307093332706, 139.98856953610317,
    30.057565990280217, 169.90663778157892, 253.82185431482623,
    8.68740019720347, -59.89926875163037, 243.95530980005992,
    -15.56174865048471,
)


def native_weights(values=None) -> np.ndarray:
    """Validate and copy the fixed-order weight vector before any timed search."""
    source = np.asarray(DEFAULT_WEIGHTS if values is None else values)
    if source.shape != (len(FEATURE_NAMES),) or source.dtype.kind not in "iuf":
        raise ValueError(f"Expected {len(FEATURE_NAMES)} numeric native weights")
    result = np.array(source, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all() or (np.abs(result) > 10_000).any():
        raise ValueError("Native weights must be finite and between -10,000 and 10,000")
    return result
