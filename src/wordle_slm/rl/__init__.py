"""GRPO trainer, reward, curriculum + replay, and the shared rollout.

Plan steps: H (reward), I (curriculum+replay), plus Y (play_game), Q (GRPO), X (overfit gate).
"""

from wordle_slm.rl.curriculum import Curriculum
from wordle_slm.rl.reward import RewardBreakdown, compute_reward

__all__ = ["Curriculum", "RewardBreakdown", "compute_reward"]
