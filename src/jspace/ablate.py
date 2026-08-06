"""J-space ablation: remove the residual stream's component in the span of the
top-k active J-lens vectors, at every token position across a band of layers.

Deviations from Gurnee et al. (2026), forced by compute budget and documented in the
report:

1. The paper solves for a sparse non-negative combination of k J-lens vectors by gradient
   pursuit. We instead select the k vectors by highest non-negative correlation in one
   shot and then remove the exact least-squares projection onto their span. This is
   "thresholding + refit", a standard cheap surrogate for pursuit. It agrees with pursuit
   when the selected vectors are near-orthogonal and over-selects correlated duplicates
   when they are not.
2. The dictionary is restricted to the V' most frequent tokens rather than the full
   vocabulary (151936 for Qwen3-4B). Full-vocabulary pursuit at every token, layer and
   decode step is ~150x more expensive than our entire compute budget.

Both reduce the *strength* of the ablation, so they bias toward under-ablating (a null
result), not toward confirming the hypothesis.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from .lens import _decoder_blocks


@dataclasses.dataclass
class AblationConfig:
    layers: list[int]                # workspace band to ablate
    k: int = 10                      # top-k active lens vectors per position
    mode: str = "jspace"             # see below
    exclude_clean_topk: int = 10     # skip tokens in the clean forward pass's top-10
    seed: int = 0
    rank_offset: int = 1000          # ctrl_rank uses ranks [offset, offset+k)

# Modes:
#   none         no intervention (the clean baseline)
#   jspace       remove the span of the top-k active lens vectors
#   ctrl_random  remove k random directions, rescaled so that the residual-stream
#                displacement has the SAME NORM as the jspace removal would have had at
#                that position. Isolates "does removing this much signal hurt?"
#   ctrl_rank    remove lens vectors at ranks [rank_offset, rank_offset+k): same vector
#                family, same geometry, wrong contents. Isolates "are lens-shaped
#                directions load-bearing, or specifically the active ones?"
#   random       legacy unmatched random control


def _remove_span(h: Tensor, dirs: Tensor) -> Tensor:
    """Remove the least-squares projection of h onto span(dirs).

    Args:
        h: [N, d] residual vectors (N = B*T flattened).
        dirs: [N, k, d] unit-norm directions per position.

    Returns:
        [N, d] with the component in each span removed.
    """
    # G a = b, where G = dirs dirs^T [N,k,k], b = dirs h [N,k]
    G = torch.einsum("nkd,njd->nkj", dirs, dirs)
    b = torch.einsum("nkd,nd->nk", dirs, h)
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype).expand_as(G)
    a = torch.linalg.solve(G + 1e-3 * eye, b.unsqueeze(-1)).squeeze(-1)   # [N,k]
    return h - torch.einsum("nk,nkd->nd", a, dirs)


class JSpaceAblator:
    """Forward hooks that ablate the J-space on a band of decoder layers.

    `set_excluded(ids)` supplies, per position, the token ids to protect (the clean
    forward pass's top-10). Call it before each ablated forward pass.
    """

    def __init__(self, model, dictionaries: dict[int, Tensor], cfg: AblationConfig):
        """
        Args:
            dictionaries: {layer: [V', d]} unit-normalized J-lens vectors.
            cfg: ablation configuration.
        """
        self.cfg = cfg
        self.dicts = dictionaries
        self._handles = []
        self._excluded: Tensor | None = None
        self._gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self.n_calls = 0
        self.enabled = True    # toggled off for the paired clean forward pass

        if cfg.mode in ("none", "clean"):
            return
        blocks = _decoder_blocks(model)
        for li in cfg.layers:
            self._handles.append(blocks[li].register_forward_hook(self._make_hook(li)))

    def set_excluded(self, excluded: Tensor | None):
        """excluded: [B, T, m] LongTensor of protected token ids (indices into the
        dictionary's token subset), or None."""
        self._excluded = excluded

    def _make_hook(self, li: int):
        def hook(_module, _inp, out):
            if not self.enabled:
                return out
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            new_h = self._ablate(h, li)
            if is_tuple:
                return (new_h,) + out[1:]
            return new_h
        return hook

    @torch.no_grad()
    def _ranked_idx(self, flat: Tensor, D: Tensor, lo: int, k: int) -> Tensor:
        """Indices of the lens vectors at correlation ranks [lo, lo+k)."""
        Vp = D.shape[0]
        c = flat @ D.T                                                # [N, V']
        if self._excluded is not None:
            exc = self._excluded.reshape(flat.shape[0], -1).to(flat.device)
            c = torch.cat([c, torch.zeros(flat.shape[0], 1, device=c.device)], dim=1)
            c.scatter_(1, exc, float("-inf"))
            c = c[:, :Vp]
        c = c.clamp_min(0.0)                                          # non-negative activation
        return c.topk(min(lo + k, Vp), dim=-1).indices[:, lo:lo + k]

    @torch.no_grad()
    def _ablate(self, h: Tensor, li: int) -> Tensor:
        D = self.dicts[li].to(device=h.device, dtype=torch.float32)   # [V', d]
        B, T, d = h.shape
        flat = h.reshape(B * T, d).float()
        Vp, k, mode = D.shape[0], self.cfg.k, self.cfg.mode
        self.n_calls += 1

        if mode == "jspace":
            out = _remove_span(flat, D[self._ranked_idx(flat, D, 0, k)])

        elif mode == "ctrl_rank":
            idx = self._ranked_idx(flat, D, self.cfg.rank_offset, k)
            out = _remove_span(flat, D[idx])

        elif mode == "ctrl_random":
            # Norm-matched: same displacement magnitude as the jspace removal would have
            # had at this position, but in directions unrelated to the active contents.
            ref = _remove_span(flat, D[self._ranked_idx(flat, D, 0, k)])
            target = (flat - ref).norm(dim=-1, keepdim=True)           # [N,1]
            ridx = torch.randint(0, Vp, (B * T, k), generator=self._gen).to(h.device)
            rnd = _remove_span(flat, D[ridx])
            delta = flat - rnd
            delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * target
            out = flat - delta

        elif mode == "random":                                         # legacy, unmatched
            ridx = torch.randint(0, Vp, (B * T, k), generator=self._gen).to(h.device)
            out = _remove_span(flat, D[ridx])

        else:
            return h

        return out.reshape(B, T, d).to(h.dtype)

    def remove(self):
        for hd in self._handles:
            hd.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
