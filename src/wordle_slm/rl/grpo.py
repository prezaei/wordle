"""GRPO trainer — full-fidelity RL over the generation policy (spec §6.3; Plan: Q).

Extends the tracer bullet (`rl.tracer`) to the real training loop:
- a **`θ_old` snapshot** per rollout batch so K>1 inner epochs have a meaningful clipped ratio
  (the tracer's K=1 has ratio≡1); old log-probs are frozen once and reused across inner epochs,
- a **budget-sized rollout loop** (`secrets_per_update` secrets × `group_size` rollouts),
- **curriculum + hard-word replay** for secret selection (`rl.curriculum`),
- **LR warmup**, grad clipping, and the **echo-trap telemetry** (entropy, KL, grad-norm, advantage
  variance, kept-group fraction — spec §8/T) to TensorBoard,
- a frozen **`π_ref`** (the SFT checkpoint) for the k3 KL, and periodic held-out eval + curriculum
  promotion.

Same GRPO math as the tracer: group-relative mean-centered advantage (no ÷std), zero-variance
groups filtered, clipped surrogate + k3 KL, all summed over the guess-letter tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from random import Random

import torch
from torch import nn

from wordle_slm.config import GRPOConfig, RewardConfig
from wordle_slm.model.serialization import encode_completed_game, guess_letter_target_positions
from wordle_slm.model.tokenizer import Tokenizer
from wordle_slm.model.transformer import WordleGenerator
from wordle_slm.rl.curriculum import Curriculum
from wordle_slm.rl.reward import compute_reward
from wordle_slm.rl.rollout import letter_id_tensor, play_game
from wordle_slm.rl.tracer import compute_group_advantages
from wordle_slm.telemetry.run_log import RunLog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateStats:
    reward_mean: float
    advantage_var: float
    kl: float
    entropy: float
    grad_norm: float
    loss: float
    kept_secrets: int
    n_secrets: int
    stepped: bool
    hard_secrets: tuple[str, ...]  # secrets whose group didn't fully win → replay queue


def _trajectory_logps(
    model: WordleGenerator,
    tokenizer: Tokenizer,
    game,
    letter_ids: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-guess-letter log-probs ``[n_positions]`` + summed entropy. Caller sets the grad context.

    Teacher-forced over the realized game (logit q-1 predicts letter q), log_softmax over the
    26-letter action space (same mask as generation).
    """
    seq_list = encode_completed_game(game.turns, tokenizer)
    seq = torch.tensor(seq_list, device=device).unsqueeze(0)
    targets = guess_letter_target_positions(seq_list, tokenizer)
    letter_lo = int(letter_ids.min().item())
    logits = model.forward(seq)[0]
    logps: list[torch.Tensor] = []
    entropy = torch.zeros((), device=device)
    for q in targets:
        logp_all = torch.log_softmax(logits[q - 1][letter_ids], dim=0)
        logps.append(logp_all[seq_list[q] - letter_lo])
        entropy = entropy + -(logp_all.exp() * logp_all).sum()
    return torch.stack(logps), entropy


def grpo_update(
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
) -> UpdateStats:
    """One full GRPO update over `secrets` (K inner epochs); skips the step if all groups filter."""
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    letter_ids = letter_id_tensor(tokenizer, device)
    was_training = model.training
    model.eval()  # dropout off so sampled and recomputed log-probs are consistent
    reward_values: list[float] = []
    advantage_values: list[float] = []
    hard_secrets: list[str] = []
    kept: list[tuple[object, float]] = []  # (game, advantage)
    kept_secrets = 0  # groups that survived the zero-variance filter

    try:
        with torch.no_grad():  # sampling + advantages
            for secret in secrets:
                games = [
                    play_game(
                        model, tokenizer, secret, sample=True, generator=generator, device=device
                    )
                    for _ in range(group_size)
                ]
                if sum(g.won for g in games) < group_size:
                    hard_secrets.append(secret)  # at least one rollout lost → replay it
                rewards = torch.tensor(
                    [compute_reward(g, reward).total for g in games], device=device
                )
                reward_values.extend(rewards.tolist())
                advantages = compute_group_advantages(
                    rewards, filter_zero_variance=grpo.filter_zero_variance
                )
                if advantages is None:
                    continue
                kept_secrets += 1
                advantage_values.extend(advantages.tolist())
                kept.extend((game, float(a)) for game, a in zip(games, advantages, strict=True))

        reward_mean = sum(reward_values) / len(reward_values)
        if not kept:  # every group was zero-variance — nothing to learn this update
            return UpdateStats(
                reward_mean, 0.0, 0.0, 0.0, 0.0, 0.0, 0, len(secrets), False, tuple(hard_secrets)
            )

        # Freeze θ_old + π_ref per-position log-probs once (θ_old = the pre-update model).
        frozen: list[tuple[object, float, torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for game, advantage in kept:
                old_logps, _ = _trajectory_logps(model, tokenizer, game, letter_ids, device)
                ref_logps, _ = _trajectory_logps(ref_model, tokenizer, game, letter_ids, device)
                frozen.append((game, advantage, old_logps, ref_logps))

        grad_norm = torch.zeros((), device=device)
        loss = torch.zeros((), device=device)
        total_kl = torch.zeros((), device=device)
        total_entropy = torch.zeros((), device=device)
        total_positions = 0
        for _ in range(grpo.inner_epochs):
            total_surrogate = torch.zeros((), device=device)
            total_kl = torch.zeros((), device=device)
            total_entropy = torch.zeros((), device=device)
            total_positions = 0
            for game, advantage, old_logps, ref_logps in frozen:
                cur_logps, entropy = _trajectory_logps(model, tokenizer, game, letter_ids, device)
                ratio = torch.exp(cur_logps - old_logps)  # 1 at epoch 0; ≠1 after a step
                clipped = torch.clamp(ratio, 1.0 - grpo.clip_eps, 1.0 + grpo.clip_eps)
                total_surrogate = (
                    total_surrogate + torch.minimum(ratio * advantage, clipped * advantage).sum()
                )
                log_ratio = ref_logps - cur_logps
                total_kl = total_kl + (torch.exp(log_ratio) - log_ratio - 1.0).sum()
                total_entropy = total_entropy + entropy
                total_positions += cur_logps.numel()
            loss = -total_surrogate / total_positions + grpo.kl_beta * total_kl / total_positions
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite GRPO loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grpo.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm: {grad_norm.item()}")
            optimizer.step()
    finally:
        model.train(was_training)

    return UpdateStats(
        reward_mean=reward_mean,
        advantage_var=float(torch.tensor(advantage_values).var(unbiased=False)),
        kl=float(total_kl.detach() / total_positions),
        entropy=float(total_entropy.detach() / total_positions),
        grad_norm=float(grad_norm),
        loss=float(loss.detach()),
        kept_secrets=kept_secrets,
        n_secrets=len(secrets),
        stepped=True,
        hard_secrets=tuple(hard_secrets),
    )


def eval_win_rate(
    model: WordleGenerator,
    tokenizer: Tokenizer,
    secrets: tuple[str, ...],
    *,
    device: str = "cpu",
) -> float:
    """Greedy held-out win rate over `secrets` (no sampling) — the learning-curve signal."""
    if not secrets:
        return 0.0
    wins = sum(play_game(model, tokenizer, s, sample=False, device=device).won for s in secrets)
    return wins / len(secrets)


def overfit_one_secret(
    model: WordleGenerator,
    ref_model: WordleGenerator,
    tokenizer: Tokenizer,
    secret: str,
    *,
    grpo: GRPOConfig,
    reward: RewardConfig,
    n_updates: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
    lr: float | None = None,
) -> list[UpdateStats]:
    """Layer-2 gate (Plan: X): run GRPO on ONE fixed secret and return the per-update stats.

    The decisive pre-flight check that the *full* loop actually learns: warm-start a model that can
    produce valid, varied guesses, then confirm mean group reward **rises** over updates and the
    model learns to solve this secret — if it can't improve on one word, it won't on the full set.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr if lr is not None else grpo.lr)
    return [
        grpo_update(
            model,
            ref_model,
            tokenizer,
            (secret,),
            grpo=grpo,
            reward=reward,
            optimizer=optimizer,
            group_size=grpo.group_size,
            device=device,
            generator=generator,
        )
        for _ in range(n_updates)
    ]


def train_grpo(
    model: WordleGenerator,
    ref_model: WordleGenerator,
    tokenizer: Tokenizer,
    curriculum: Curriculum,
    *,
    grpo: GRPOConfig,
    reward: RewardConfig,
    n_updates: int,
    eval_secrets: tuple[str, ...] = (),
    eval_every: int = 25,
    warmup_updates: int | None = None,
    device: str = "cpu",
    run_log: RunLog | None = None,
    seed: int = 0,
) -> list[UpdateStats]:
    """Run `n_updates` GRPO updates: curriculum sampling, LR warmup, telemetry, periodic eval."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=grpo.lr)
    rng = Random(seed)
    generator = torch.Generator().manual_seed(seed)
    warmup = (
        warmup_updates if warmup_updates is not None else max(1, int(n_updates * grpo.warmup_ratio))
    )
    history: list[UpdateStats] = []

    for update in range(n_updates):
        secrets = tuple(curriculum.sample(rng) for _ in range(grpo.secrets_per_update))
        lr = grpo.lr * min(1.0, (update + 1) / warmup)  # linear warmup
        for group in optimizer.param_groups:
            group["lr"] = lr
        stats = grpo_update(
            model,
            ref_model,
            tokenizer,
            secrets,
            grpo=grpo,
            reward=reward,
            optimizer=optimizer,
            group_size=grpo.group_size,
            device=device,
            generator=generator,
        )
        history.append(stats)
        for secret in stats.hard_secrets:
            curriculum.record_loss(secret)  # hard-word replay

        if run_log is not None:
            for tag, value in (
                ("reward_mean", stats.reward_mean),
                ("advantage_var", stats.advantage_var),
                ("kl", stats.kl),
                ("entropy", stats.entropy),
                ("grad_norm", stats.grad_norm),
                ("loss", stats.loss),
                ("kept_fraction", stats.kept_secrets / stats.n_secrets),
                ("lr", lr),
            ):
                run_log.log_scalar(f"grpo/{tag}", value, update)

        if eval_secrets and (update + 1) % eval_every == 0:
            win_rate = eval_win_rate(model, tokenizer, eval_secrets, device=device)
            promoted = curriculum.maybe_promote(win_rate)
            if run_log is not None:
                run_log.log_scalar("grpo/eval_win_rate", win_rate, update)
                run_log.log_scalar("grpo/tier", float(curriculum.tier_index), update)
            logger.info(
                "update %d: eval win_rate=%.3f tier=%d%s",
                update,
                win_rate,
                curriculum.tier_index,
                " (promoted)" if promoted else "",
            )

    return history
