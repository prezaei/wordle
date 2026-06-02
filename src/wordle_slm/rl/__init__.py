"""GRPO trainer, reward, curriculum + replay, and the shared play_game() rollout.

Plan steps: H (reward), I (curriculum+replay), Y (play_game), V (tracer bullet),
Q (GRPO trainer), X (single-secret overfit gate).
"""
