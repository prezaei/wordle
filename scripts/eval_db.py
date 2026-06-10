"""Honest free-gen eval harness with a VERSIONED SQLite store + per-turn / CoT capture.

Runs the CLEAN protocol (greedy free-generation over {26 letters, THINK, commit}; NO dictionary, trie,
or consistency filter at inference; a non-word is counted as a turn but NOT fed back) on the held-out
TEST split, exactly reusing ``validity_max.play``'s decode loop. For every game it records the target,
outcome, each turn's guess/feedback/clue-respect, the model's ephemeral CoT candidate words, and the
size of the still-consistent answer set before each turn (an OFFLINE diagnostic — never seen by the
model). Results land in ``runs/evals.db`` as an immutable, versioned run row + child games/turns.

Honesty guards (assert at runtime):
  * the captured win/valid rate must equal ``validity_max.evaluate`` on the same set (faithful decode)
  * targets are the held-out split only (``split(seed=0)`` -> held[96:]); never train secrets

Usage:
    uv run python scripts/eval_db.py                      # best model, full TEST -> new version
    uv run python scripts/eval_db.py --checkpoint runs/x.pt --label "x" --limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validity_max as vm  # noqa: E402  reuse the EXACT honest protocol + model/tokenizer/split

from wordle_slm.engine import Color, Game, Status  # noqa: E402

SCHEMA_VERSION = 1
DB_PATH = Path("evals/evals.db")  # tracked in-repo (runs/ is gitignored)
COLOR_CH = {Color.GREEN: "G", Color.YELLOW: "Y", Color.GRAY: "X"}
RARE_LETTERS = set("jqxzvkwfy")


# --------------------------------------------------------------------------------------------------
# DB schema
# --------------------------------------------------------------------------------------------------
def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL,
            label           TEXT NOT NULL,
            checkpoint      TEXT NOT NULL,
            checkpoint_sha  TEXT NOT NULL,
            git_sha         TEXT NOT NULL,
            schema_version  INTEGER NOT NULL,
            protocol        TEXT NOT NULL,        -- JSON: decode rules (proves honesty)
            split_seed      INTEGER NOT NULL,
            n_targets       INTEGER NOT NULL,
            win_rate        REAL NOT NULL,
            valid_rate      REAL NOT NULL,
            avg_turns_win   REAL,
            clue_respect    REAL NOT NULL,
            notes           TEXT
        );
        CREATE TABLE IF NOT EXISTS eval_games (
            game_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id               INTEGER NOT NULL REFERENCES eval_runs(run_id),
            target               TEXT NOT NULL,
            solved               INTEGER NOT NULL,
            outcome              TEXT NOT NULL,    -- won | lost_runout | lost_nonword
            n_valid_turns        INTEGER NOT NULL,
            final_consistent_set INTEGER NOT NULL, -- |answers still consistent| at end (offline)
            target_dup_letters   INTEGER NOT NULL,
            target_rare_letters  INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS eval_turns (
            turn_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id              INTEGER NOT NULL REFERENCES eval_games(game_id),
            turn_index           INTEGER NOT NULL,
            emitted              TEXT NOT NULL,    -- raw word the model emitted (may be <5 chars)
            valid                INTEGER NOT NULL,
            feedback             TEXT,             -- e.g. "GGYXX" (null if invalid)
            n_green              INTEGER NOT NULL,
            n_yellow             INTEGER NOT NULL,
            consistent_w_prior   INTEGER NOT NULL, -- did this guess respect all prior clues?
            consistent_set_before INTEGER NOT NULL,-- |consistent answers| before this guess (offline)
            cot_candidates       TEXT NOT NULL     -- JSON list of the model's CoT words this turn
        );
        CREATE INDEX IF NOT EXISTS ix_games_run ON eval_games(run_id);
        CREATE INDEX IF NOT EXISTS ix_turns_game ON eval_turns(game_id);
        """
    )
    return conn


# --------------------------------------------------------------------------------------------------
# Faithful capture-play (mirrors validity_max.play byte-for-byte; only ADDS recording)
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def play_capture(model: Any, secret: str) -> dict[str, Any]:
    g = Game(secret)
    visible: list[Any] = []
    records: list[dict[str, Any]] = []
    while g.status is Status.ONGOING:
        seq = vm.board_only(visible)
        guess: list[int] = []
        cot_tokens: list[int] = []
        committed = False
        for _ in range(60):
            nxt = int(
                vm.ALLOWED_GEN[
                    int(
                        torch.argmax(
                            model.forward(torch.tensor([seq], device=vm.DEV))[0, -1][vm.ALLOWED_GEN]
                        )
                    )
                ]
            )
            seq.append(nxt)
            if committed:
                if nxt in vm.LETTER_SET:
                    guess.append(nxt)
                if len(guess) >= 5:
                    break
            elif nxt == vm.tok.guess_id:
                committed = True
            else:
                cot_tokens.append(nxt)  # THINK markers + the CoT candidate letters
        word = "".join(vm.tok.id_to_token(t) for t in guess[:5])
        played = word if len(word) == 5 else "zzzzz"
        turn = g.guess(played)
        records.append(
            {
                "emitted": word,
                "played": played,
                "valid": turn.valid,
                "feedback": turn.feedback,
                "consistent_w_prior": vm.is_consistent(played, visible) if turn.valid else False,
                "cot_candidates": _parse_cot(cot_tokens),
            }
        )
        if turn.valid:
            visible.append(turn)
        else:
            break  # non-word: identical to validity_max.play (lost)
    return {"game": g, "records": records}


def _parse_cot(cot_tokens: list[int]) -> list[str]:
    """Split the pre-commit token stream into candidate words (each THINK starts one)."""
    words: list[str] = []
    cur: list[int] = []
    for t in cot_tokens:
        if t == vm.THINK:
            if cur:
                words.append("".join(vm.tok.id_to_token(x) for x in cur[:5]))
            cur = []
        elif t in vm.LETTER_SET:
            cur.append(t)
    if cur:
        words.append("".join(vm.tok.id_to_token(x) for x in cur[:5]))
    return words


# --------------------------------------------------------------------------------------------------
# Offline diagnostics
# --------------------------------------------------------------------------------------------------
def consistent_trajectory(records: list[dict[str, Any]], answers: list[str]) -> list[int]:
    """|answers still consistent| BEFORE each turn, narrowing incrementally over valid turns.

    Pure offline analysis over the answer pool (the full set of possible secrets); the model never
    sees this. We narrow against the answer set so 'set size 1' means 'the answer is uniquely
    determined among possible secrets' (the deduction-difficulty metric).
    """
    sizes: list[int] = []
    cands = list(answers)
    history: list[Any] = []
    for rec in records:
        sizes.append(len(cands))
        if rec["valid"]:
            history.append(_mk_turn(rec))
            cands = [c for c in cands if vm.is_consistent(c, history[-1:])]
    return sizes


def _mk_turn(rec: dict[str, Any]) -> Any:
    from wordle_slm.engine.game import Turn

    return Turn(guess=rec["played"], feedback=rec["feedback"], valid=rec["valid"])


def classify(g: Any, records: list[dict[str, Any]]) -> str:
    if g.won:
        return "won"
    if records and not records[-1]["valid"]:
        return "lost_nonword"
    return "lost_runout"


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/validity_max_v4.pt", type=Path)
    ap.add_argument("--label", default="validity_max_v4 (best honest)")
    ap.add_argument("--limit", default=None, type=int, help="cap #targets (default: full TEST)")
    ap.add_argument("--db", default=DB_PATH, type=Path)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    train, held = vm.split(seed=0)
    test = tuple(held[96:])
    targets = test[: args.limit] if args.limit else test
    assert not (set(targets) & set(train)), "HELD-OUT CONTAMINATION: target in train set"
    answers = sorted(
        set(train) | set(held)
    )  # universe of possible secrets (offline diagnostic only)

    model = vm.WordleGenerator(vm.CFG, vm.VOCAB).to(vm.DEV)
    vm.load_checkpoint(str(args.checkpoint), model)
    model.eval()

    print(
        f"[eval_db] checkpoint={args.checkpoint} targets={len(targets)} (held-out TEST) dev={vm.DEV}",
        flush=True,
    )
    t0 = time.time()
    games_data: list[dict[str, Any]] = []
    for i, secret in enumerate(targets, 1):
        cap = play_capture(model, secret)
        cap["target"] = secret
        cap["traj"] = consistent_trajectory(cap["records"], answers)
        games_data.append(cap)
        if i % 50 == 0 or i == len(targets):
            won = sum(1 for d in games_data if d["game"].won)
            print(
                f"  {i}/{len(targets)}  win-so-far {won / i:.3f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    # ---- aggregate + FIDELITY CHECK against the canonical evaluate() ----
    wins = [d for d in games_data if d["game"].won]
    win_rate = len(wins) / len(games_data)
    n_turns = sum(len(d["game"].turns) for d in games_data)
    n_valid = sum(t.valid for d in games_data for t in d["game"].turns)
    valid_rate = n_valid / n_turns
    avg_turns_win = (sum(d["game"].guesses_used for d in wins) / len(wins)) if wins else None
    respect_turns = [r for d in games_data for r in d["records"] if r["valid"]]
    clue_respect = sum(r["consistent_w_prior"] for r in respect_turns) / max(len(respect_turns), 1)

    ref = vm.evaluate(model, targets)
    assert abs(ref["win"] - win_rate) < 1e-9, (
        f"FIDELITY FAIL: capture win {win_rate} != evaluate {ref['win']}"
    )
    print(
        f"[eval_db] FIDELITY OK: capture win {win_rate:.4f} == evaluate {ref['win']:.4f}",
        flush=True,
    )

    # ---- persist a new immutable version ----
    conn = connect(args.db)
    protocol = {
        "decode": "greedy free-gen over {26 letters, THINK, commit}",
        "dictionary_at_inference": False,
        "trie_or_consistency_filter_at_inference": False,
        "nonword": "counted as a turn, NOT fed back (break)",
        "max_guesses": 6,
        "cot": "ephemeral, model-generated",
    }
    cur = conn.execute(
        """INSERT INTO eval_runs(created_at,label,checkpoint,checkpoint_sha,git_sha,schema_version,
           protocol,split_seed,n_targets,win_rate,valid_rate,avg_turns_win,clue_respect,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now(UTC).isoformat(),
            args.label,
            str(args.checkpoint),
            sha16(args.checkpoint),
            git_sha(),
            SCHEMA_VERSION,
            json.dumps(protocol),
            0,
            len(targets),
            win_rate,
            valid_rate,
            avg_turns_win,
            clue_respect,
            args.notes,
        ),
    )
    run_id = cur.lastrowid
    for d in games_data:
        g = d["game"]
        target = d["target"]
        cur = conn.execute(
            """INSERT INTO eval_games(run_id,target,solved,outcome,n_valid_turns,final_consistent_set,
               target_dup_letters,target_rare_letters) VALUES(?,?,?,?,?,?,?,?)""",
            (
                run_id,
                target,
                int(g.won),
                classify(g, d["records"]),
                sum(t.valid for t in g.turns),
                d["traj"][-1] if d["traj"] else 0,
                int(len(set(target)) < 5),
                sum(1 for ch in set(target) if ch in RARE_LETTERS),
            ),
        )
        game_id = cur.lastrowid
        for idx, (rec, size_before) in enumerate(zip(d["records"], d["traj"])):
            fb = "".join(COLOR_CH[c] for c in rec["feedback"]) if rec["feedback"] else None
            conn.execute(
                """INSERT INTO eval_turns(game_id,turn_index,emitted,valid,feedback,n_green,n_yellow,
                   consistent_w_prior,consistent_set_before,cot_candidates) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_id,
                    idx,
                    rec["emitted"],
                    int(rec["valid"]),
                    fb,
                    (fb.count("G") if fb else 0),
                    (fb.count("Y") if fb else 0),
                    int(rec["consistent_w_prior"]),
                    size_before,
                    json.dumps(rec["cot_candidates"]),
                ),
            )
    conn.commit()

    outcomes = Counter(classify(d["game"], d["records"]) for d in games_data)
    print(
        f"\n[eval_db] RUN #{run_id} stored in {args.db}\n"
        f"  win {win_rate:.3f} ({len(wins)}/{len(games_data)}) | valid {valid_rate:.3f} | "
        f"clue-respect {clue_respect:.3f} | avg-turns(win) {avg_turns_win}\n"
        f"  outcomes: {dict(outcomes)}",
        flush=True,
    )
    conn.close()


if __name__ == "__main__":
    main()
