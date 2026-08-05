"""Paired clean/ablated batched decoding.

The paper protects "tokens that appear in the top-10 tokens of a clean forward pass".
Under ablation the generated context diverges from the clean run, so the protected set
must be recomputed against the *ablated run's own context*. We therefore keep two KV
caches over the same token sequence: one advanced with ablation off (to read the clean
top-10) and one with ablation on (which actually produces the tokens).

Cost is ~2x a plain decode. Set `exclude_clean_topk=0` to skip the clean pass entirely.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def generate_paired(
    model,
    tokenizer,
    prompts: list[str],
    ablator,
    full_to_sub: Tensor,
    max_new_tokens: int = 320,
    device: str = "cuda",
) -> list[str]:
    """Greedy-decode `prompts` with `ablator` active.

    Args:
        ablator: JSpaceAblator (mode "none" gives the clean baseline).
        full_to_sub: [vocab] LongTensor mapping vocab id -> dictionary index, with V'
            (the sentinel) for tokens outside the dictionary subset.

    Returns:
        Decoded completions (prompt stripped).
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    B = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    m = ablator.cfg.exclude_clean_topk
    need_clean = m > 0 and ablator.cfg.mode != "none"
    full_to_sub = full_to_sub.to(device)

    clean_cache = None
    abl_cache = None
    cur = input_ids
    cur_attn = attn
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    eos = tokenizer.eos_token_id
    out_tokens = [[] for _ in range(B)]

    for step in range(max_new_tokens):
        # --- clean pass: read the protected token set for this context ---
        if need_clean:
            ablator.enabled = False
            co = model(cur, attention_mask=cur_attn, past_key_values=clean_cache, use_cache=True)
            clean_cache = co.past_key_values
            top = co.logits.topk(m, dim=-1).indices             # [B, T, m]
            excluded = full_to_sub[top]                          # -> dictionary indices
            ablator.set_excluded(excluded)
        else:
            ablator.set_excluded(None)

        # --- ablated pass: actually generate ---
        ablator.enabled = True
        ao = model(cur, attention_mask=cur_attn, past_key_values=abl_cache, use_cache=True)
        abl_cache = ao.past_key_values
        next_tok = ao.logits[:, -1, :].argmax(dim=-1)            # [B] greedy

        next_tok = torch.where(finished, torch.full_like(next_tok, eos), next_tok)
        for b in range(B):
            if not finished[b]:
                out_tokens[b].append(next_tok[b].item())
        finished |= next_tok == eos
        if finished.all():
            break

        cur = next_tok.unsqueeze(-1)                             # [B, 1]
        cur_attn = torch.cat([cur_attn, (~finished).long().unsqueeze(-1)], dim=1)

    ablator.enabled = False
    return [tokenizer.decode(t, skip_special_tokens=True) for t in out_tokens]


def build_full_to_sub(vocab_size: int, token_subset: Tensor) -> Tensor:
    """Inverse map from vocab id to dictionary index; sentinel = len(token_subset)."""
    sentinel = token_subset.numel()
    m = torch.full((vocab_size,), sentinel, dtype=torch.long)
    m[token_subset] = torch.arange(sentinel, dtype=torch.long)
    return m
