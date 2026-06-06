"""Typed configuration for every stage.

Defaults mirror the spec's hyperparameter table (docs/design/wordle-slm.md §13), tagged
I (invariant) / H (hypothesis — expect to change) / R (routine). These are *skeletons* for
S0; preset loading + CLI-override merge + resolved-config logging are implemented in step K.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Redesigned model presets (name -> d_model, n_layers, n_heads, d_ff, dropout).
# Philosophy: DEPTH over width (Wordle is multi-turn deduction -> sequential reasoning -> layers),
# size co-designed with the diverse curriculum (~14k secrets supports a bigger model without the
# memorization measured on the 2,315-answer set), and extra dropout at scale to fight overfitting.
MODEL_PRESETS: dict[str, tuple[int, int, int, int, float]] = {
    "tiny": (128, 6, 4, 512, 0.10),  # ~1.2M — smoke/tests
    "base": (320, 10, 8, 1280, 0.10),  # ~12M
    "large": (512, 16, 8, 2048, 0.15),  # ~50M — recommended WITH the diverse curriculum
    "xl": (640, 20, 10, 2560, 0.15),  # ~98M — depth-heavy, the upper end for MPS
}


@dataclass
class ModelConfig:
    """Decoder-only char transformer. Redesigned: DEPTH-emphasized, co-designed with data. R.

    Wordle is multi-turn logical deduction, which rewards DEPTH (sequential reasoning steps) over
    width — the presets stack layers at moderate width. Size is meant to scale WITH the data: the
    old 1–5M cap suited the 2,315-answer set, but the redesigned curriculum's ~14k diverse secrets
    support a larger model without the memorization we measured, so ``preset("large")`` ~50M (with
    extra dropout) is the recommended training config. The dataclass default stays small for
    tests/smoke; use ``ModelConfig.preset(name)`` for real runs.
    """

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    context_len: int = 128  # a full 6-turn game is ~66 tokens
    dropout: float = 0.1

    @classmethod
    def preset(cls, name: str) -> ModelConfig:
        """A redesigned, depth-emphasized config by name (see ``MODEL_PRESETS``)."""
        if name not in MODEL_PRESETS:
            raise ValueError(f"unknown model preset {name!r}; choose from {sorted(MODEL_PRESETS)}")
        d_model, n_layers, n_heads, d_ff, dropout = MODEL_PRESETS[name]
        return cls(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout)

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
    """Shaped per-guess reward for free generation (spec §6.4; tune in Phase 3).

    Rewards real, clue-respecting words so the generator learns to play. Dominance (must hold):
    ``p_invalid > b`` and ``q > b`` (any honest progress beats an invalid/stall), and the most
    progress a slow game can farm (≈ ``5*a + few*b``) ``< win_base`` (a win dominates farming).
    """

    a: float = 0.2  # new-green bonus (paid once per position)
    b: float = 0.1  # new-yellow bonus (only when it raises a known min-count)
    p_invalid: float = 0.5  # penalty for a non-word guess (consumes the turn, no progress)
    q: float = 0.5  # penalty for violating a confirmed clue (drops a green / reuses a known gray)
    repeat_penalty: float = 0.4  # penalty for re-emitting a previous (valid) guess (wasted turn)
    drop_present_penalty: float = 0.3  # penalty for omitting a known-present (yellow) letter
    c: float = 0.02  # per-guess step cost
    win_base: float = 3.0  # base win bonus (> max farmable progress)
    win_speed: float = 0.5  # extra per unused guess: win_base + win_speed*(max_guesses - t)
    loss_penalty: float = 1.0  # subtracted on a loss


@dataclass
class SFTConfig:
    """Imitation head-start (spec §5.4-5.6). lr/wd are R; bars/blend are H.

    ``aux_validity_lambda`` weights the auxiliary trie-validity loss (the empirically best lever:
    +3.4pts held-out, runs/sft_aux). At each guess-letter step it pushes probability mass onto the
    dictionary's valid next-letters, so free-generation learns to spell real words — the trie is a
    training signal only; inference stays unaided. Default-on; set 0.0 for the plain masked NLL.
    """

    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    aux_validity_lambda: float = 0.5  # H: auxiliary trie-validity loss weight (0.0 = off)
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
    """Difficulty-ordered, diversity-first curriculum + hard-word replay (redesigned; spec §6.5). H.

    Secrets are drawn from the FULL valid list (``build_curriculum_pool``), ordered easy->hard
    (common answers first, rarer valid words later) — 8x the answer-only pool, the change that
    matters for generalization. Tiers are cumulative pool sizes (None = full) that WIDEN as the
    policy improves; ``promote_patience`` force-widens after that many eval points even when the
    win-rate gate isn't cleared, so the curriculum always progresses (the old gate never fired).
    """

    # Cumulative pool sizes over the ~14k diverse pool; None marks the full pool (final tier).
    tiers: tuple[int | None, ...] = (2000, 6000, 10000, None)
    promote_threshold: float = 0.55  # win rate on the current tier to widen
    promote_patience: int = 6  # force-widen after this many evals on a tier (robust progress)
    replay_capacity: int = 512
    replay_prob: float = 0.15


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
    "MODEL_PRESETS",
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
