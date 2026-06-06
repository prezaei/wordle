"""Tracer bullet: one thin end-to-end GRPO slice for the generation policy (spec §6; Plan: V ★).

The minimal loop the full GRPO trainer (Plan: Q) extends. One update: random-init generator →
sample a group of rollouts per secret (via `rollout.play_game`) → trajectory reward
(`reward.compute_reward`) → group-relative advantage (mean-centered, **no ÷std**, zero-variance
groups filtered) → clipped surrogate + k3 KL vs a frozen reference, **summed over the guess-letter
tokens only** → one optimizer step → scalars to TensorBoard.

The policy is per-token generation, so the differentiable trajectory log-prob is recomputed by a
**teacher-forced causal forward** over the realized game sequence (`encode_completed_game`): the
logit at position `q-1` predicts the letter at position `q`; we `log_softmax` over the **26-letter
action space** (same mask as generation — spec §5.3 / migration C2) and index the realized letter.
At this single update `θ_old = θ`, so the ratio is 1 by construction (its gradient is `∇logπ_θ`);
the trainer Q snapshots `θ_old` weights for K>1. This module asserts the *mechanics* (incl. the loss
mask), not that reward rises (the model is random).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
from torch import nn

from wordle_slm.config import GRPOConfig, RewardConfig
from wordle_slm.model.serialization import encode_completed_game, guess_letter_target_positions
from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.model.transformer import WordleGenerator
from wordle_slm.rl.reward import compute_reward
from wordle_slm.rl.rollout import letter_id_tensor, play_game
from wordle_slm.telemetry.run_log import RunLog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TracerStats:
    """One update's diagnostics (all finite; the GRPO echo-trap signals — spec §8)."""

    reward_mean: float
    advantage_var: float
    loss: float
    kl: float
    entropy: float
    grad_norm: float
    kept_groups: int


def make_reference(model: WordleGenerator) -> WordleGenerator:
    """A frozen deep copy of `model` to serve as `π_ref` for the KL penalty (spec §6.3)."""
    ref = copy.deepcopy(model)
    ref.eval()
    ref.requires_grad_(False)
    return ref


def compute_group_advantages(
    rewards: torch.Tensor, *, filter_zero_variance: bool
) -> torch.Tensor | None:
    """Mean-centered advantages for one same-secret group (Dr. GRPO: no ÷std).

    Returns ``None`` when the group carries no learning signal (all rewards equal → zero variance;
    StarPO-S filter, spec §6.3). Tolerance, not exact == 0, for float dust. Advantages carry no
    gradient (rewards are constants).
    """
    advantages = rewards - rewards.mean()
    if filter_zero_variance and bool((advantages.abs() < 1e-9).all()):
        return None
    return advantages


def per_guess_logps(
    model: WordleGenerator,
    tokenizer: Tokenizer,
    game,
    letter_ids: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-guess-letter log-probs ``[n_positions]`` + summed entropy. Caller sets the grad context.

    The single teacher-forced recompute shared by the tracer and the GRPO trainer (so the importance
    ratio is computed identically in both): encode the realized game, take the guess-letter target
    positions (logit ``q-1`` predicts letter ``q``), ``log_softmax`` over the 26-letter action space
    (same mask as generation — spec §5.3 / migration C2), and index the realized letter.
    """
    seq_list = encode_completed_game(game.turns, tokenizer)
    seq = torch.tensor(seq_list, device=device).unsqueeze(0)  # [1, L]
    targets = guess_letter_target_positions(seq_list, tokenizer)  # the generated letter positions
    letter_lo = tokenizer.letter_lo  # letters are contiguous (vocab: specials then a-z)
    logits = model.forward(seq)[0]  # [L, vocab]
    logps: list[torch.Tensor] = []
    entropy = torch.zeros((), device=device)
    for q in targets:
        logp_all = torch.log_softmax(logits[q - 1][letter_ids], dim=0)  # over the action space
        logps.append(logp_all[seq_list[q] - letter_lo])  # the realized letter's log-prob
        entropy = entropy + -(logp_all.exp() * logp_all).sum()
    return torch.stack(logps), entropy


def trajectory_terms(
    model: WordleGenerator,
    ref_model: WordleGenerator,
    tokenizer: Tokenizer,
    game,
    advantage: torch.Tensor,
    letter_ids: torch.Tensor,
    *,
    clip_eps: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Teacher-forced replay of one game → (Σ clipped-surrogate, Σ k3-KL, Σ entropy, n_positions).

    Recomputes the generation log-probs WITH gradients over the realized sequence. ``advantage`` is
    the (constant) group-relative advantage broadcast to every guess-letter token of the trajectory.
    """
    logps, entropy = per_guess_logps(model, tokenizer, game, letter_ids, device)
    with torch.no_grad():
        ref_logps, _ = per_guess_logps(ref_model, tokenizer, game, letter_ids, device)

    surrogate = torch.zeros((), device=device)
    kl = torch.zeros((), device=device)
    for logp, ref_logp in zip(logps, ref_logps, strict=True):
        # Importance ratio at θ_old = θ: value 1, gradient ∇logπ_θ.
        ratio = torch.exp(logp - logp.detach())
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = surrogate + torch.min(ratio * advantage, clipped * advantage)
        log_ratio = ref_logp - logp  # k3 KL(π_θ ‖ π_ref): exp(Δ) − Δ − 1 ≥ 0
        kl = kl + (torch.exp(log_ratio) - log_ratio - 1.0)
    return surrogate, kl, entropy, len(logps)


def grpo_tracer_step(
    model: WordleGenerator,
    ref_model: WordleGenerator,
    tokenizer: Tokenizer,
    secrets: tuple[str, ...],
    *,
    grpo: GRPOConfig,
    reward: RewardConfig,
    optimizer: torch.optim.Optimizer,
    group_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
    run_log: RunLog | None = None,
    step: int = 0,
) -> TracerStats:
    """Run one end-to-end GRPO update over `secrets` and return finite diagnostics.

    Per secret: sample `group_size` rollouts, score them, mean-center the rewards into advantages,
    drop zero-variance groups, accumulate the clipped surrogate + KL over the guess-letter tokens,
    then one clipped optimizer step. Raises if all groups are filtered or any tensor is non-finite.
    """
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    letter_ids = letter_id_tensor(tokenizer, device)
    was_training = model.training
    model.eval()  # dropout off so the replayed log-probs match the sampled rollouts
    total_surrogate = torch.zeros((), device=device)
    total_kl = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    total_positions = 0
    kept_groups = 0
    reward_values: list[float] = []
    advantage_values: list[float] = []

    try:
        for secret in secrets:
            games = [
                play_game(model, tokenizer, secret, sample=True, generator=generator, device=device)
                for _ in range(group_size)
            ]
            rewards = torch.tensor([compute_reward(g, reward).total for g in games], device=device)
            reward_values.extend(rewards.tolist())
            advantages = compute_group_advantages(
                rewards, filter_zero_variance=grpo.filter_zero_variance
            )
            if advantages is None:
                continue
            kept_groups += 1
            advantage_values.extend(advantages.tolist())
            for game, advantage in zip(games, advantages, strict=True):
                surrogate, kl, entropy, n = trajectory_terms(
                    model,
                    ref_model,
                    tokenizer,
                    game,
                    advantage,
                    letter_ids,
                    clip_eps=grpo.clip_eps,
                    device=device,
                )
                total_surrogate = total_surrogate + surrogate
                total_kl = total_kl + kl
                total_entropy = total_entropy + entropy
                total_positions += n
    finally:
        model.train(was_training)

    if kept_groups == 0:
        raise RuntimeError("every group was zero-variance: no learning signal (use ≥2 outcomes)")

    loss = -total_surrogate / total_positions + grpo.kl_beta * total_kl / total_positions
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite GRPO loss: {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grpo.max_grad_norm)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"non-finite grad norm: {grad_norm.item()}")
    optimizer.step()

    stats = TracerStats(
        reward_mean=float(sum(reward_values) / len(reward_values)),
        advantage_var=float(torch.tensor(advantage_values).var(unbiased=False)),
        loss=float(loss.detach()),
        kl=float(total_kl.detach() / total_positions),
        entropy=float(total_entropy.detach() / total_positions),
        grad_norm=float(grad_norm),
        kept_groups=kept_groups,
    )
    if run_log is not None:
        for tag, value in (
            ("reward_mean", stats.reward_mean),
            ("advantage_var", stats.advantage_var),
            ("loss", stats.loss),
            ("kl", stats.kl),
            ("entropy", stats.entropy),
            ("grad_norm", stats.grad_norm),
            ("kept_groups", float(stats.kept_groups)),
        ):
            run_log.log_scalar(f"tracer/{tag}", value, step)
    logger.info(
        "tracer step %d: reward_mean=%.4f adv_var=%.4f loss=%.4f kl=%.4f entropy=%.4f "
        "grad_norm=%.4f kept_groups=%d",
        step,
        stats.reward_mean,
        stats.advantage_var,
        stats.loss,
        stats.kl,
        stats.entropy,
        stats.grad_norm,
        stats.kept_groups,
    )
    return stats
