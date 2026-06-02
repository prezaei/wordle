"""Typed configuration for every stage.

Defaults mirror the spec's hyperparameter table (docs/design/wordle-slm.md §13), tagged
I (invariant) / H (hypothesis — expect to change) / R (routine). These are *skeletons* for
S0; preset loading + CLI-override merge + resolved-config logging are implemented in step K.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Decoder-only transformer. Default ~3.2M params; target range 1–5M. R.

    Endpoints (pin to the speed+memory budget in Phase 1): small 128/5/4/512 ≈1.0M,
    default 256/4/8/1024 ≈3.2M, top 256/6/8/1024 ≈4.8M.
    """

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    context_len: int = 128  # a full 6-turn game is ~66 tokens
    dropout: float = 0.1

    def estimated_params(self, vocab_size: int = 34) -> int:
        """Rough parameter count with weight-tied embeddings.

        Dominated by the transformer blocks; the exact count is verified in step G.
        """
        per_layer = 4 * self.d_model**2 + 2 * self.d_model * self.d_ff
        return self.n_layers * per_layer + (vocab_size + self.context_len) * self.d_model


@dataclass
class TokenizerConfig:
    """Char-level vocab: 26 letters + 8 specials (~34 tokens). I."""

    # 26 a-z + <BOS> <EOS> <PAD> <SEP> <GUESS> <green> <yellow> <gray>
    vocab_size: int = 34


@dataclass
class RewardConfig:
    """Speed-dominant reward for the v3 restricted-action policy (spec §1.5; tune in Phase 3).

    Winning is near-automatic when the agent only plays still-consistent words, so the reward
    optimizes *guess count*: information gain per guess + a speed-scaled win bonus.
    """

    info_gain_weight: float = 1.0  # weight on log(|C_before| / |C_after|) per guess
    win_base: float = 1.0
    # Raised for the fewest-guesses priority: info-gain telescopes (constant across wins, so it
    # cancels in GRPO's group-relative advantage), making win_speed the real speed lever. Tunable.
    win_speed: float = 0.5  # extra per unused guess (faster = more)
    step_cost: float = 0.02  # per guess
    loss_penalty: float = 0.5  # subtracted on a loss


@dataclass
class SFTConfig:
    """Imitation head-start (spec §5.4-5.6). lr/wd are R; bars/blend are H."""

    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    cap_minutes: float = 15.0  # I: outcome-based stop, capped
    valid_word_bar: float = 0.95  # Phase-1 DoD
    clue_respect_bar: float = 0.80  # Phase-1 DoD (provisional)
    teacher_weak_frac: float = 0.70  # H: feedback-consistent
    teacher_strong_frac: float = 0.30  # H: near-optimal


@dataclass
class GRPOConfig:
    """GRPO (spec §6.1-6.3). algorithm/gamma are I; group/secrets/beta are H; rest R."""

    clip_eps: float = 0.2
    inner_epochs: int = 1  # K: grad steps per rollout batch
    group_size: int = 16  # G: rollouts per secret — pin via the Phase-0 memory benchmark
    secrets_per_update: int = 8
    lr: float = 1e-5
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    kl_beta: float = 0.01
    kl_estimator: str = "k3"  # unbiased, non-negative, low-variance
    advantage_norm: str = "mean_center"  # Dr. GRPO: no division by std
    filter_zero_variance: bool = True  # StarPO-S
    temperature: float = 1.0  # train; eval is greedy
    gamma: float = 1.0  # I: episodes <= 6 steps


@dataclass
class CurriculumConfig:
    """Performance-triggered widening + hard-word replay (spec §6.5). H."""

    # None marks the full train set (the final tier).
    tiers: tuple[int | None, ...] = (200, 500, 1000, None)
    promote_threshold: float = 0.60  # win rate on the current tier to widen
    replay_capacity: int = 256
    replay_prob: float = 0.10


@dataclass
class EvalConfig:
    """Two-tier eval + the measurable Phase-2 gate (spec §6.6-6.7). R defaults; gate H."""

    curve_subsample: int = 128  # cheap held-out subsample for the learning curve
    curve_cadence: int = 25  # updates between subsample evals
    full_cadence: int = 200  # updates between full held-out evals (checkpoint selection)
    gate_margin_pts: float = 10.0  # H: win rate must beat the floor by >= this
    gate_consecutive: int = 3  # H: over >= this many full-eval points
    gap_max_pts: float = 15.0  # H: generalization gap must stay below this


@dataclass
class DataConfig:
    """Word lists + seeded split (spec §4.1). split/seed are I."""

    train_frac: float = 0.80
    split_seed: int = 0
    data_dir: str = "data"


@dataclass
class RunConfig:
    """Top-level config composing every stage + run-wide settings."""

    seed: int = 0  # I: fixed (MPS is only approximately reproducible)
    device: str = "mps"  # CPU fallback for unsupported ops
    run_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)


__all__ = [
    "ModelConfig",
    "TokenizerConfig",
    "RewardConfig",
    "SFTConfig",
    "GRPOConfig",
    "CurriculumConfig",
    "EvalConfig",
    "DataConfig",
    "RunConfig",
]
