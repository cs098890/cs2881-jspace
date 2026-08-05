"""Jacobian lens: fit the averaged input-output Jacobian J_l = E[d h_final / d h_l].

Following Gurnee et al. (2026), "Verbalizable Representations Form a Global Workspace
in Language Models". The J-lens vectors are the rows of W_U @ J_l.

Estimator
---------
For a probe vector u ~ N(0, I_d), backpropagating the scalar

    s = sum_{t'} u . h_final,t'

to the layer-l residual stream at position t yields, by causal masking,

    g_{l,t} = d s / d h_{l,t} = sum_{t' >= t} (d h_final,t' / d h_{l,t})^T u = J_{l,t}^T u

so that E[u g_{l,t}^T] = E[u u^T J_{l,t}] = J_{l,t}. Averaging over positions, probes and
prompts gives an unbiased estimate of J_l with one forward+backward per probe, and one
backward supplies every layer at once.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import torch
from torch import Tensor


@dataclasses.dataclass
class LensConfig:
    layers: list[int]           # decoder layer indices to fit (0-indexed, output of block)
    n_probes: int = 4096        # total random probes; variance ~ d/N, d=2560
    batch_size: int = 16
    seq_len: int = 128
    seed: int = 0


class _ResidualTap:
    """Captures the output hidden state of selected decoder layers and retains grads."""

    def __init__(self, model, layers: list[int]):
        self.layers = layers
        self.acts: dict[int, Tensor] = {}
        self._handles = []
        blocks = _decoder_blocks(model)
        for li in layers:
            self._handles.append(blocks[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, li: int):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad()
            self.acts[li] = h
            return out
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


def _decoder_blocks(model):
    """Return the list of transformer blocks for a HF causal LM."""
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"could not locate decoder blocks on {type(model)}")


def fit_jacobian(
    model,
    token_batches: Iterable[Tensor],
    cfg: LensConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> dict[int, Tensor]:
    """Estimate J_l for each layer in cfg.layers.

    Args:
        model: HF causal LM in eval mode.
        token_batches: iterable of LongTensor [B, T] input_ids.
        cfg: fitting configuration.

    Returns:
        {layer_index: J_l} with J_l of shape [d_model, d_model], float32, on CPU.
    """
    d = model.config.hidden_size
    accum = {li: torch.zeros(d, d, dtype=dtype, device=device) for li in cfg.layers}
    n_seen = 0
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    tap = _ResidualTap(model, cfg.layers)
    try:
        for input_ids in token_batches:
            if n_seen >= cfg.n_probes:
                break
            input_ids = input_ids.to(device)
            B = input_ids.shape[0]

            # One independent probe per sequence in the batch.
            u = torch.randn(B, d, generator=gen).to(device=device, dtype=torch.float32)
            u = u / u.norm(dim=-1, keepdim=True) * (d ** 0.5)

            tap.acts.clear()
            model.zero_grad(set_to_none=True)
            out = model(input_ids, output_hidden_states=True, use_cache=False)
            h_final = out.hidden_states[-1].float()          # [B, T, d]

            # s = sum_b sum_t' u_b . h_final[b, t']; causality restricts grads to t' >= t.
            s = torch.einsum("btd,bd->", h_final, u)
            s.backward()

            for li in cfg.layers:
                g = tap.acts[li].grad                        # [B, T, d]
                if g is None:
                    raise RuntimeError(f"no grad captured at layer {li}")
                g_mean = g.float().mean(dim=1)               # average over positions -> [B, d]
                # accum += sum_b u_b (g_mean_b)^T
                accum[li] += torch.einsum("bi,bj->ij", u, g_mean).to(dtype)

            n_seen += B
    finally:
        tap.remove()
        model.zero_grad(set_to_none=True)

    return {li: (accum[li] / max(n_seen, 1)).cpu() for li in cfg.layers}


def lens_vectors(
    J: Tensor,
    W_U: Tensor,
    token_subset: Tensor | None = None,
    normalize: bool = True,
) -> Tensor:
    """J-lens vectors = rows of W_U @ J, optionally restricted to a token subset.

    Args:
        J: [d, d] averaged Jacobian for one layer.
        W_U: [vocab, d] unembedding matrix.
        token_subset: optional LongTensor of token ids to keep (for tractability).
        normalize: unit-normalize rows so they can be used as projection directions.

    Returns:
        [V', d] float32 matrix of J-lens vectors.
    """
    W = W_U if token_subset is None else W_U[token_subset]
    V = (W.float() @ J.float())                              # [V', d]
    if normalize:
        V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return V


def apply_lens(J: Tensor, W_U: Tensor, h: Tensor, norm_fn=None) -> Tensor:
    """lens(h) = W_U norm(J h) -> vocabulary logits. Used for sanity checks."""
    x = h.float() @ J.float().T
    if norm_fn is not None:
        x = norm_fn(x)
    return x @ W_U.float().T
