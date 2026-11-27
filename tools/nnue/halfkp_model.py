"""PyTorch training model and portable float32 HalfKP weight format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .halfkp_features import (
    FEATURES, KING_STRIDE, PADDING_INDEX, PADDING_PIECE, SQUARES,
    feature_indices, pack_boards,
)
from .model import EVAL_SCALE, RESIDUAL_CLIP

LAYOUT = "tablut9_halfkp_ksq_piece_sq_v1"


def torch_indices(kings: torch.Tensor, pieces: torch.Tensor, views: torch.Tensor,
                  permutations: torch.Tensor) -> torch.Tensor:
    # A fixed-width bag and one reserved index avoid a GPU-to-CPU sync for offsets.
    codes = pieces.long()
    squares = permutations[views[:, None], codes.remainder(SQUARES)]
    ids = (permutations[views, kings.long()][:, None] * KING_STRIDE
           + codes.div(SQUARES, rounding_mode="floor") * SQUARES + squares)
    return ids.masked_fill(codes == PADDING_PIECE, PADDING_INDEX)


class HalfKP(nn.Module):
    def __init__(self, hidden: int = 64, hidden2: int = 32,
                 cap: float = RESIDUAL_CLIP) -> None:
        super().__init__()
        self.cap = float(cap)
        self.embedding = nn.EmbeddingBag(FEATURES, hidden, mode="sum", padding_idx=PADDING_INDEX)
        self.acc_bias = nn.Parameter(torch.full((hidden,), 0.2))
        self.hidden2 = nn.Linear(hidden, hidden2) if hidden2 else None
        self.heads = nn.Linear(hidden2 or hidden, 2)
        self.side_calib = nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, std=0.035)
            self.embedding.weight[PADDING_INDEX].zero_()
        if self.hidden2 is not None:
            nn.init.constant_(self.hidden2.bias, 0.15)
        nn.init.normal_(self.heads.weight, std=1e-3)
        nn.init.zeros_(self.heads.bias)

    def forward(self, indices: torch.Tensor, sides: torch.Tensor) -> torch.Tensor:
        hidden = (self.embedding(indices) + self.acc_bias).clamp(0.0, 1.0)
        if self.hidden2 is not None:
            hidden = self.hidden2(hidden).clamp(0.0, 1.0)
        raw = self.heads(hidden).gather(1, sides[:, None]).squeeze(1).float()
        return self.cap * torch.tanh(raw)

    def logits(self, scores: torch.Tensor, sides: torch.Tensor) -> torch.Tensor:
        return scores / EVAL_SCALE + self.side_calib[sides]


def export_weights(path: Path, model: HalfKP) -> None:
    """Export rows [index,H], optional [H2,H], then heads [D/A,H2 or H]."""
    state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    fields = {
        "feature_layout": np.array(LAYOUT),
        "eval_scale": np.float32(EVAL_SCALE),
        "residual_clip": np.float32(model.cap),
        "hidden2": np.int32(model.hidden2.out_features if model.hidden2 is not None else 0),
        "embedding": state["embedding.weight"],
        "acc_bias": state["acc_bias"],
        "head_weight": state["heads.weight"],
        "head_bias": state["heads.bias"],
    }
    if model.hidden2 is not None:
        fields.update(hidden2_weight=state["hidden2.weight"], hidden2_bias=state["hidden2.bias"])
    np.savez_compressed(path, **fields)


def load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        weights = dict(dataset)
    if str(weights["feature_layout"]) != LAYOUT or float(weights["eval_scale"]) != EVAL_SCALE:
        raise ValueError(f"{path.name}: unexpected HalfKP feature layout or eval scale")
    cap, hidden2 = float(weights["residual_clip"]), int(weights["hidden2"])
    hidden = len(weights["acc_bias"])
    shapes = {
        "embedding": (FEATURES, hidden), "acc_bias": (hidden,),
        "head_weight": (2, hidden2 or hidden), "head_bias": (2,),
    }
    if hidden2:
        shapes.update(hidden2_weight=(hidden2, hidden), hidden2_bias=(hidden2,))
    if cap <= 0 or not np.isfinite(cap) or hidden <= 0 or hidden2 < 0:
        raise ValueError(f"{path.name}: invalid model dimensions or cap")
    for name, shape in shapes.items():
        array = weights[name]
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"{path.name}: invalid shape, dtype or value for {name}")
    if np.any(weights["embedding"][PADDING_INDEX] != 0):
        raise ValueError(f"{path.name}: padding row must be zero")
    return weights


def numpy_residual(weights: dict[str, np.ndarray], boards: np.ndarray,
                   sides: np.ndarray) -> np.ndarray:
    """Inference using the exported arrays, in the original orientation."""
    kings, pieces = pack_boards(boards)
    ids = feature_indices(kings, pieces)
    accumulator = weights["embedding"][ids].sum(axis=1) + weights["acc_bias"]
    hidden = np.clip(accumulator, 0.0, 1.0)
    if int(weights["hidden2"]):
        hidden = np.clip(hidden @ weights["hidden2_weight"].T + weights["hidden2_bias"],
                         0.0, 1.0)
    heads = hidden @ weights["head_weight"].T + weights["head_bias"]
    raw = heads[np.arange(len(sides)), sides]
    return weights["residual_clip"] * np.tanh(raw)
