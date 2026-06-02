"""Tracer bullet: one thin end-to-end GRPO slice for the v3 policy (spec §6; Plan: V ★).

The minimal loop the full GRPO trainer (Plan: Q) later extends to full fidelity. One update:

  random-init `CandidateScorer` → sample a group of rollouts per secret (via `rollout.play_game`)
  → trajectory reward (`reward.compute_reward`) → group-relative advantage (mean-centered, **no
  ÷std**, zero-variance groups filtered) → clipped surrogate + k3 KL against a frozen reference
  → one optimizer step → scalars to TensorBoard.

The v3 policy is the softmax over the still-consistent candidates' scores, so to get a
*differentiable* trajectory log-prob we **replay** each sampled game and recompute the per-step
scores with gradients
(`play_game` samples under `no_grad`). At this single update the old policy equals the current one
(`θ_old = θ`), so the importance ratio is 1 by construction (its gradient is `∇logπ_θ`); the trainer
Q will snapshot `θ_old` separately to support K>1 inner epochs. This module **does not** assert the
reward rises — the model is random; it asserts the *mechanics* are wired and finite.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
from torch import nn

from wordle_slm.config import GRPOConfig, RewardConfig
from wordle_slm.engine import Game, Turn, filter_consistent
from wordle_slm.model.scorer import CandidateScorer
from wordle_slm.model.serialization import encode_board, encode_word
from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.rl.reward import compute_reward
from wordle_slm.rl.rollout import play_game
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


def make_reference(model: CandidateScorer) -> CandidateScorer:
    """A frozen deep copy of `model` to serve as `π_ref` for the KL penalty (spec §6.3)."""
    ref = copy.deepcopy(model)
    ref.eval()
    ref.requires_grad_(False)
    return ref


def compute_group_advantages(
    rewards: torch.Tensor, *, filter_zero_variance: bool
) -> torch.Tensor | None:
    """Mean-centered advantages for one same-secret group (Dr. GRPO: no ÷std).

    Returns ``None`` when the group carries no learning signal (all rewards equal → zero variance),
    which StarPO-S filters out (spec §6.3). Advantages carry no gradient (rewards are constants).
    """
    advantages = rewards - rewards.mean()
    # Tolerance, not exact == 0: on MPS the mean of bit-identical rewards can carry ULP dust, and
    # real reward gaps here are ≥ step_cost (~0.02), far above this floor — no true signal is lost.
    if filter_zero_variance and bool((advantages.abs() < 1e-9).all()):
        return None
    return advantages


def trajectory_surrogate(
    model: CandidateScorer,
    ref_model: CandidateScorer,
    tokenizer: Tokenizer,
    game: Game,
    pool: tuple[str, ...],
    advantage: torch.Tensor,
    *,
    clip_eps: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Replay one game; return (Σ clipped-surrogate, Σ k3-KL, Σ entropy, n_steps) over its turns.

    Recomputes the per-turn candidate scores WITH gradients (unlike `play_game`). Because the old
    policy equals the current one here, the per-step ratio ``exp(logπ_θ − logπ_θ.detach())`` is 1 in
    value but carries ``∇logπ_θ`` — the policy-gradient signal. ``advantage`` is the (constant)
    group-relative advantage for this whole trajectory.
    """
    pad_id = tokenizer.pad_id
    candidates: tuple[str, ...] = tuple(pool)
    surrogate = torch.zeros((), device=device)
    kl = torch.zeros((), device=device)
    entropy = torch.zeros((), device=device)
    history: list[Turn] = []
    for turn in game.turns:
        board = torch.tensor(encode_board(history, tokenizer), device=device).unsqueeze(0)
        cand_ids = torch.tensor([encode_word(w, tokenizer) for w in candidates], device=device)
        logits = model.score(board, cand_ids, pad_id)  # [N], with gradient
        logprobs = torch.log_softmax(logits, dim=0)
        index = candidates.index(turn.guess)  # the candidate the rollout actually played
        logp = logprobs[index]

        # Importance ratio at θ_old = θ: value 1, gradient ∇logπ_θ. Clipped PPO/GRPO surrogate.
        ratio = torch.exp(logp - logp.detach())
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = surrogate + torch.min(ratio * advantage, clipped * advantage)

        # k3 KL(π_θ ‖ π_ref) at the chosen action: exp(Δ) − Δ − 1 ≥ 0, Δ = logπ_ref − logπ_θ.
        with torch.no_grad():
            ref_logp = torch.log_softmax(ref_model.score(board, cand_ids, pad_id), dim=0)[index]
        log_ratio = ref_logp - logp
        kl = kl + (torch.exp(log_ratio) - log_ratio - 1.0)

        entropy = entropy + -(logprobs.exp() * logprobs).sum()
        history.append(turn)
        candidates = filter_consistent(candidates, turn)
    return surrogate, kl, entropy, len(game.turns)


def grpo_tracer_step(
    model: CandidateScorer,
    ref_model: CandidateScorer,
    tokenizer: Tokenizer,
    secrets: tuple[str, ...],
    pool: tuple[str, ...],
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

    For each secret: sample `group_size` rollouts, score them, mean-center the rewards into
    advantages, drop zero-variance groups, accumulate the clipped surrogate + KL over the kept
    trajectories, then take a single clipped optimizer step. Scalars are logged to `run_log` (if
    given) and at INFO. Raises if all groups are filtered (no signal) or any tensor is non-finite.
    """
    if group_size < 2:
        # A group needs ≥2 rollouts or its mean-centered advantages are all zero (no signal).
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    was_training = model.training
    model.eval()  # disable dropout so the replayed log-probs match the sampled rollouts
    total_surrogate = torch.zeros((), device=device)
    total_kl = torch.zeros((), device=device)
    total_entropy = torch.zeros((), device=device)
    total_steps = 0
    kept_groups = 0
    reward_values: list[float] = []
    advantage_values: list[float] = []

    try:
        for secret in secrets:
            games = [
                play_game(
                    model, tokenizer, secret, pool, sample=True, generator=generator, device=device
                )
                for _ in range(group_size)
            ]
            rewards = torch.tensor(
                [compute_reward(g, reward, pool).total for g in games], device=device
            )
            reward_values.extend(rewards.tolist())
            advantages = compute_group_advantages(
                rewards, filter_zero_variance=grpo.filter_zero_variance
            )
            if advantages is None:
                logger.debug("secret %r: zero-variance group filtered", secret)
                continue
            kept_groups += 1
            advantage_values.extend(advantages.tolist())
            for game, advantage in zip(games, advantages, strict=True):
                surrogate, kl, entropy, n_steps = trajectory_surrogate(
                    model,
                    ref_model,
                    tokenizer,
                    game,
                    pool,
                    advantage,
                    clip_eps=grpo.clip_eps,
                    device=device,
                )
                total_surrogate = total_surrogate + surrogate
                total_kl = total_kl + kl
                total_entropy = total_entropy + entropy
                total_steps += n_steps
    finally:
        model.train(was_training)  # restore the caller's train/eval state

    if kept_groups == 0:
        raise RuntimeError("every group was zero-variance: no learning signal (use ≥2 outcomes)")

    # Token-mean GRPO objective: maximize surrogate, penalize KL drift from π_ref.
    loss = -total_surrogate / total_steps + grpo.kl_beta * total_kl / total_steps
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
        kl=float(total_kl.detach() / total_steps),
        entropy=float(total_entropy.detach() / total_steps),
        grad_norm=float(grad_norm),
        kept_groups=kept_groups,
    )
    if run_log is not None:
        run_log.log_scalar("tracer/reward_mean", stats.reward_mean, step)
        run_log.log_scalar("tracer/advantage_var", stats.advantage_var, step)
        run_log.log_scalar("tracer/loss", stats.loss, step)
        run_log.log_scalar("tracer/kl", stats.kl, step)
        run_log.log_scalar("tracer/entropy", stats.entropy, step)
        run_log.log_scalar("tracer/grad_norm", stats.grad_norm, step)
        run_log.log_scalar("tracer/kept_groups", float(stats.kept_groups), step)
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
