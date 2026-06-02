"""Command-line entrypoints for wordle-slm.

Each subcommand resolves a config (preset + ``--set`` overrides, Plan: K) and logs it; the
per-phase logic is implemented in its build-plan wave (docs/design/wordle-slm-plan.md).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from wordle_slm.config import RunConfig
from wordle_slm.config.resolve import resolve, to_dict

logger = logging.getLogger(__name__)

_COMMANDS: dict[str, str] = {
    "phase0": "Phase 0: engine, data, baselines, floor/yardstick + speed budget",
    "benchmark": "Model-rollout + update micro-benchmark: pin the real budget / group size",
    "pretrain": "Spell warm-up: LM over the word list (learn to spell before SFT)",
    "sft": "Phase 1: imitation head-start (SFT) training",
    "rl": "Phase 2: GRPO reinforcement learning",
    "eval": "Evaluate a checkpoint on the held-out set",
    "play": "Play one live game in the terminal (Phase 4)",
}


def _not_implemented(name: str) -> int:
    print(f"`wordle-slm {name}` is not implemented yet — see docs/design/wordle-slm-plan.md")
    return 1


def _run_phase0(cfg: RunConfig) -> int:
    """Phase-0 run (Plan: L): floor + yardsticks + engine throughput + budget over held-out."""
    from wordle_slm.baselines.phase0 import run_phase0
    from wordle_slm.data import load_answers, load_valid_guesses, split
    from wordle_slm.telemetry.run_log import RunLog

    answers = load_answers()
    valid = load_valid_guesses()
    _, heldout = split(seed=cfg.data.split_seed, train_frac=cfg.data.train_frac)
    with RunLog(Path(cfg.run_dir) / "phase0", config=to_dict(cfg), seed=cfg.seed) as run_log:
        report = run_phase0(
            heldout,
            answers,
            valid,
            grpo=cfg.grpo,
            eval_cfg=cfg.eval,
            run_log=run_log,
            seed=cfg.seed,
        )
    b = report.budget
    avg = report.yardstick_answers.avg_guesses_on_wins
    avg_str = f"{avg:.3f}" if avg is not None else "n/a"  # None only if the yardstick won 0 games
    print(
        f"Phase 0 over {len(heldout)} held-out secrets:\n"
        f"  floor (answers): {report.floor_answers.win_rate * 100:.3f}%   "
        f"floor (valid): {report.floor_valid.win_rate * 100:.3f}%   (~0.26% / ~0.05% expected)\n"
        f"  yardstick (valid, §4.3): {report.yardstick_valid.win_rate * 100:.2f}%   "
        f"yardstick (answers, v3 floor): {report.yardstick_answers.win_rate * 100:.2f}% "
        f"(avg {avg_str} guesses)\n"
        f"  engine: {report.games_per_sec:.0f} games/sec   "
        f"budget: ~{b.n_updates:.0f} updates fit {b.rl_seconds / 60:.0f}-min RL at G="
        f"{cfg.grpo.group_size} (fits={b.fits}, max G={b.fitting_group_size}; engine is an "
        f"optimistic upper bound — model-rollout benchmark pins it in Plan O)"
    )
    return 0


def _checkpoint_path(cfg: RunConfig, override: str | None) -> Path:
    return Path(override) if override else Path(cfg.run_dir) / "sft.pt"


def _run_benchmark(cfg: RunConfig, args: argparse.Namespace) -> int:
    """Model-rollout + update benchmark (Plan: O): pin the real budget / group size on-device."""
    from wordle_slm.baselines.benchmark import recommend_group_size, run_benchmark
    from wordle_slm.data import load_answers
    from wordle_slm.model import Tokenizer, WordleGenerator
    from wordle_slm.rl.tracer import make_reference

    device = args.device or cfg.device
    tok = Tokenizer()
    model = WordleGenerator(cfg.model, tok.vocab_size).to(device)
    rows = run_benchmark(
        model,
        make_reference(model),
        tok,
        load_answers()[:16],
        grpo=cfg.grpo,
        reward=cfg.reward,
        group_sizes=(4, 8, 16),
        device=device,
    )
    print(
        f"Model-rollout benchmark on {device}: {rows[0].rollout_games_per_sec:.1f} games/sec, "
        f"{rows[0].peak_mem_mb:.0f} MB. Updates that fit the 45-min RL window "
        f"(update compute only; eval is extra):"
    )
    for r in rows:
        batch = cfg.grpo.secrets_per_update * r.group_size
        fit = "fits" if r.fits else "too few"
        print(
            f"  G={r.group_size:<3} {batch:>4} rollouts/upd  "
            f"{r.seconds_per_update:>6.1f}s/upd  ~{r.n_updates:>5.0f} updates  {fit}"
        )
    print(f"  → recommended group_size = {recommend_group_size(rows)}")
    return 0


def _run_pretrain(cfg: RunConfig, args: argparse.Namespace) -> int:
    """Spell warm-up: LM over the valid word list so the model learns to spell (migration §6)."""
    from wordle_slm.model import Tokenizer, WordleGenerator
    from wordle_slm.sft import pretrain_lm, pretrain_words, save_checkpoint
    from wordle_slm.telemetry.run_log import RunLog

    device = args.device or cfg.device
    tok = Tokenizer()
    model = WordleGenerator(cfg.model, tok.vocab_size).to(device)
    words = pretrain_words()
    if args.limit:
        words = words[: args.limit]
    with RunLog(Path(cfg.run_dir) / "pretrain", config=to_dict(cfg), seed=cfg.seed) as run_log:
        out = pretrain_lm(
            model,
            words,
            tok,
            cfg.sft,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            max_seconds=cfg.sft.cap_minutes * 60.0,
            run_log=run_log,
            seed=cfg.seed,
        )
    path = Path(cfg.run_dir) / "pretrain.pt"
    save_checkpoint(path, model, out["optimizer"], out["step"], cfg.sft)
    print(f"pretrain done: {out['step']} steps, loss {out['loss']:.4f} → checkpoint {path}")
    print("  next: wordle-slm sft --init " + str(path))
    return 0


def _run_sft(cfg: RunConfig, args: argparse.Namespace) -> int:
    """Phase-1 SFT (Plan: M+N): teacher data → masked imitation → reloadable checkpoint."""
    from wordle_slm.data import split
    from wordle_slm.model import Tokenizer, WordleGenerator
    from wordle_slm.sft import load_checkpoint, save_checkpoint, train_sft
    from wordle_slm.teacher import generate_transcripts
    from wordle_slm.telemetry.run_log import RunLog

    device = args.device or cfg.device
    train, _ = split(seed=cfg.data.split_seed, train_frac=cfg.data.train_frac)
    if args.limit:
        train = train[: args.limit]
    transcripts = generate_transcripts(train, weak_frac=cfg.sft.teacher_weak_frac, seed=cfg.seed)
    tok = Tokenizer()
    model = WordleGenerator(cfg.model, tok.vocab_size).to(device)
    if args.init:  # warm-start from the spell-warm-up checkpoint
        load_checkpoint(args.init, model)
    with RunLog(Path(cfg.run_dir) / "sft", config=to_dict(cfg), seed=cfg.seed) as run_log:
        out = train_sft(
            model,
            [t.game for t in transcripts],
            tok,
            cfg.sft,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            max_seconds=cfg.sft.cap_minutes * 60.0,
            run_log=run_log,
            seed=cfg.seed,
        )
    path = _checkpoint_path(cfg, args.checkpoint)
    save_checkpoint(path, model, out["optimizer"], out["step"], cfg.sft)
    print(f"SFT done: {out['step']} steps, loss {out['loss']:.4f} → checkpoint {path}")
    return 0


def _run_rl(cfg: RunConfig, args: argparse.Namespace) -> int:
    """Phase-2 GRPO (Plan: Q): init from the SFT checkpoint, train, eval on held-out."""
    from wordle_slm.data import split, train_probe
    from wordle_slm.model import Tokenizer, WordleGenerator
    from wordle_slm.rl.curriculum import Curriculum
    from wordle_slm.rl.grpo import eval_win_rate, train_grpo
    from wordle_slm.rl.tracer import make_reference
    from wordle_slm.sft import load_checkpoint
    from wordle_slm.telemetry.run_log import RunLog

    device = args.device or cfg.device
    path = _checkpoint_path(cfg, args.checkpoint)
    if not path.exists():
        print(f"no SFT checkpoint at {path} — run `wordle-slm sft` first")
        return 1
    train, heldout = split(seed=cfg.data.split_seed, train_frac=cfg.data.train_frac)
    if args.limit:
        train = train[: args.limit]
    tok = Tokenizer()
    model = WordleGenerator(cfg.model, tok.vocab_size).to(device)
    load_checkpoint(path, model)  # init the policy from SFT
    ref = make_reference(model)  # frozen π_ref = the SFT model
    curriculum = Curriculum(train, cfg.curriculum)
    eval_secrets = heldout[: cfg.eval.curve_subsample]
    probe = train_probe(seed=cfg.data.split_seed, train_frac=cfg.data.train_frac)
    best_path = Path(cfg.run_dir) / "best.pt"
    with RunLog(Path(cfg.run_dir) / "rl", config=to_dict(cfg), seed=cfg.seed) as run_log:
        history = train_grpo(
            model,
            ref,
            tok,
            curriculum,
            grpo=cfg.grpo,
            reward=cfg.reward,
            n_updates=args.updates,
            eval_secrets=eval_secrets,
            probe_secrets=probe,  # generalization-gap telemetry (spec §6.7)
            best_checkpoint=best_path,  # keep the best-by-held-out model, not the last
            eval_every=cfg.eval.curve_cadence,
            device=device,
            run_log=run_log,
            seed=cfg.seed,
        )
    stepped = sum(h.stepped for h in history)
    summary = f"RL done: {len(history)} updates ({stepped} stepped)"
    if eval_secrets:
        win = eval_win_rate(model, tok, eval_secrets, device=device)
        summary += f", held-out win rate {win * 100:.1f}% (best checkpoint → {best_path})"
    print(summary)
    return 0


def _run_eval(cfg: RunConfig, args: argparse.Namespace) -> int:
    """Phase-1 readiness eval (Plan: P): valid-word + green-retention bars on the SFT checkpoint."""
    from wordle_slm.data import split
    from wordle_slm.eval.phase1 import evaluate_phase1
    from wordle_slm.model import Tokenizer, WordleGenerator
    from wordle_slm.sft import load_checkpoint

    device = args.device or cfg.device
    path = _checkpoint_path(cfg, args.checkpoint)
    if not path.exists():
        print(f"no checkpoint at {path} — run `wordle-slm sft` first")
        return 1
    _, heldout = split(seed=cfg.data.split_seed, train_frac=cfg.data.train_frac)
    secrets = heldout[: args.limit] if args.limit else heldout[: cfg.eval.curve_subsample]
    tok = Tokenizer()
    model = WordleGenerator(cfg.model, tok.vocab_size).to(device)
    load_checkpoint(path, model)
    report = evaluate_phase1(model, tok, secrets, device=device)
    passed = report.passes(cfg.sft)
    vw_bar = cfg.sft.valid_word_bar * 100
    cr_bar = cfg.sft.clue_respect_bar * 100
    verdict = "PASS — ready for RL" if passed else "FAIL — keep training SFT"
    print(
        f"Phase-1 eval over {report.n_games} held-out games: "
        f"valid-word {report.valid_word_rate * 100:.1f}% (bar {vw_bar:.0f}%), "
        f"green-retention {report.green_retention * 100:.1f}% (bar {cr_bar:.0f}%) → {verdict}"
    )
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wordle-slm", description="Wordle SLM toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in _COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--preset", default="default", help="config preset name")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config field, e.g. --set grpo.group_size=16 (repeatable)",
        )
        p.add_argument("--device", default=None, help="override the device (cpu/mps)")
        p.add_argument("--limit", type=int, default=None, help="cap train secrets (quick runs)")
        p.add_argument("--epochs", type=int, default=50, help="SFT epochs (or until the time cap)")
        p.add_argument("--batch-size", type=int, default=64, help="SFT batch size")
        p.add_argument("--updates", type=int, default=50, help="RL updates")
        p.add_argument(
            "--checkpoint", default=None, help="SFT checkpoint path (default run_dir/sft.pt)"
        )
        p.add_argument(
            "--init", default=None, help="warm-start SFT from this checkpoint (pretrain)"
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = resolve(preset=args.preset, overrides=args.overrides)
    except (KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))  # clean usage error + exit 2, not a raw traceback
    logger.info(
        "resolved config for %s: %s", args.command, json.dumps(to_dict(cfg), sort_keys=True)
    )
    if args.command == "phase0":
        return _run_phase0(cfg)
    if args.command == "benchmark":
        return _run_benchmark(cfg, args)
    if args.command == "pretrain":
        return _run_pretrain(cfg, args)
    if args.command == "sft":
        return _run_sft(cfg, args)
    if args.command == "rl":
        return _run_rl(cfg, args)
    if args.command == "eval":
        return _run_eval(cfg, args)
    return _not_implemented(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
