"""SFT training + overfit-gate tests (Plan: N, W; spec §5.5)."""

from __future__ import annotations

import torch

from wordle_slm.config import ModelConfig, SFTConfig
from wordle_slm.engine import Game
from wordle_slm.model import Tokenizer, WordleGenerator
from wordle_slm.model.serialization import encode_completed_game, guess_letter_target_positions
from wordle_slm.rl.rollout import letter_id_tensor, play_game
from wordle_slm.sft import load_checkpoint, make_batch, save_checkpoint, train_sft


def _small_model(seed: int = 0) -> tuple[WordleGenerator, Tokenizer]:
    torch.manual_seed(seed)
    tok = Tokenizer()
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)
    return WordleGenerator(cfg, tok.vocab_size), tok


def _game(secret: str = "crane", *guesses: str) -> Game:
    g = Game(secret)
    for w in guesses:
        g.guess(w)
    return g


# --- make_batch -------------------------------------------------------------------------------


def test_make_batch_shapes_mask_and_targets() -> None:
    _, tok = _small_model()
    g1 = _game("crane", "slate", "crane")  # 2 turns -> 10 guess-letter positions
    g2 = _game("money", "slate")  # 1 turn -> 5
    ids, target_idx, mask = make_batch([g1, g2], tok)
    assert ids.shape == target_idx.shape == mask.shape
    assert mask.sum().item() == 15  # 5 per completed guess
    # the masked targets recover the realized letters (index in the 26-letter space).
    seq = encode_completed_game(g1.turns, tok)
    letter_lo = tok.token_to_id("a")
    q = guess_letter_target_positions(seq, tok)[0]
    assert int(target_idx[0, q - 1]) == seq[q] - letter_lo


# --- W: overfit-one gate ----------------------------------------------------------------------


def test_overfit_one_transcript_drives_loss_to_zero_and_reproduces() -> None:
    model, tok = _small_model()
    teacher_game = _game("crane", "slate", "crane")  # the single transcript to memorize
    out = train_sft(model, [teacher_game], tok, SFTConfig(), epochs=250, batch_size=1)
    assert out["loss"] < 0.05  # CE -> ~0
    model.eval()
    played = play_game(model, tok, "crane", sample=False)
    assert [t.guess for t in played.turns] == [t.guess for t in teacher_game.turns]  # memorized


def test_adamw_optimizer_state_updates() -> None:
    # §12 smoke: the MPS AdamW bug — assert exp_avg_sq actually updates.
    model, tok = _small_model()
    out = train_sft(model, [_game("crane", "slate", "crane")], tok, SFTConfig(), epochs=3)
    state = out["optimizer"].state[next(model.parameters())]
    assert "exp_avg_sq" in state and float(state["exp_avg_sq"].abs().sum()) > 0.0


def test_sft_loss_is_masked_to_guess_letter_tokens() -> None:
    # The loss must depend ONLY on the guess-letter positions (board/feedback are context).

    model, tok = _small_model()
    g = _game("crane", "slate", "crane")
    ids, target_idx, mask = make_batch([g], tok)
    letter_ids = letter_id_tensor(tok)
    ids.requires_grad_(False)
    logits = model.forward(ids)
    logits.retain_grad()
    logp = torch.log_softmax(logits[:, :, letter_ids], dim=-1)
    nll = -logp.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)
    ((nll * mask).sum() / mask.sum()).backward()
    grad_positions = logits.grad[0].abs().sum(dim=1) > 0
    predict = torch.zeros_like(grad_positions)
    for q in guess_letter_target_positions(encode_completed_game(g.turns, tok), tok):
        predict[q - 1] = True
    assert torch.equal(grad_positions, predict)  # gradient exactly on the predict positions


# --- checkpoint round-trip --------------------------------------------------------------------


def test_checkpoint_round_trip_restores_greedy_play(tmp_path) -> None:
    model, tok = _small_model()
    out = train_sft(model, [_game("crane", "slate", "crane")], tok, SFTConfig(), epochs=10)
    model.eval()
    before = [t.guess for t in play_game(model, tok, "crane", sample=False).turns]
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, out["optimizer"], out["step"], SFTConfig())

    torch.manual_seed(999)
    fresh = WordleGenerator(
        ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0), tok.vocab_size
    )
    ckpt = load_checkpoint(path, fresh)
    fresh.eval()
    after = [t.guess for t in play_game(fresh, tok, "crane", sample=False).turns]
    assert ckpt["step"] == out["step"]
    assert after == before  # the reloaded model plays identically


def test_train_sft_respects_the_time_cap() -> None:
    model, tok = _small_model()
    games = [_game("crane", "slate", "crane") for _ in range(20)]
    out = train_sft(model, games, tok, SFTConfig(), epochs=100, batch_size=4, max_seconds=0.0)
    assert out["step"] == 1  # the 0s cap stops after the first batch
