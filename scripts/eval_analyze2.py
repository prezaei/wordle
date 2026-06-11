"""Deeper failure-pattern mining over the eval DB (see eval_db.py), aimed at TRAINING levers.

Beyond the first-pass breakdown (eval_analyze.py), this digs to the letter/word level:

  1. Non-word taxonomy: how far is each emitted non-word from a real word (Hamming over 5-letter
     dict)? and does it respect the prior clues? -> spelling near-miss vs constraint-violation vs chimera.
  2. Constraint-violation decomposition: of valid guesses that break the clues, WHICH clue type
     (dropped green / missing yellow / reused gray)? -> which aux signal to strengthen.
  3. Hard-core vs recoverable targets across ALL runs: lost-everywhere (greedy + every sampling seed)
     vs won-only-by-sampling -> separates vocabulary gaps from spelling variance.
  4. Vocabulary probe: does the target word EVER appear in the model's own output (guess or CoT) across
     all runs? If never -> the model can't produce that word at all (a spelling/vocab gap, trainable).
  5. CoT-at-determined: when the answer is pinned (<=3 left) and the model misses, what did it emit/think?

Usage: uv run python scripts/eval_analyze2.py   (writes evals/analysis2.md)
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from wordle_slm.data import load_valid_guesses, split

DB = Path("evals/evals.db")
VALID5 = [w for w in load_valid_guesses() if len(w) == 5]
VALID_SET = set(VALID5)
_train, _held = split(seed=0)
ANSWERS = set(_train) | set(_held)


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (-)"


def hamming5(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def min_hamming_to_real(word: str) -> int:
    """Fewest letter changes to reach SOME real 5-letter word (0 if it is one)."""
    if word in VALID_SET or len(word) != 5:
        return 0 if word in VALID_SET else 5
    best = 5
    for w in VALID5:
        d = hamming5(word, w)
        if d < best:
            best = d
            if best == 1:
                return 1
    return best


def constraints_from(prior: list[tuple[str, str]]):
    """greens(pos->letter), present(letters that must appear), absent(letters that must not)."""
    greens: dict[int, str] = {}
    present: set[str] = set()
    for guess, fb in prior:
        for i, (c, f) in enumerate(zip(guess, fb)):
            if f == "G":
                greens[i] = c
                present.add(c)
            elif f == "Y":
                present.add(c)
    absent: set[str] = set()
    for guess, fb in prior:
        for c, f in zip(guess, fb):
            if f == "X" and c not in present:
                absent.add(c)
    return greens, present, absent


def violation_kinds(guess: str, greens, present, absent) -> set[str]:
    kinds = set()
    for pos, let in greens.items():
        if guess[pos] != let:
            kinds.add("dropped_green")
    for c in present:
        if c not in guess:
            kinds.add("missing_present")
    for c in guess:
        if c in absent:
            kinds.add("reused_gray")
    return kinds


def load_runs(conn):
    conn.row_factory = sqlite3.Row
    runs = {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM eval_runs")}
    games = defaultdict(list)
    for r in conn.execute("SELECT * FROM eval_games"):
        games[r["run_id"]].append(dict(r))
    turns = defaultdict(list)
    for r in conn.execute("SELECT t.*, g.run_id, g.target FROM eval_turns t JOIN eval_games g ON t.game_id=g.game_id ORDER BY t.turn_index"):
        turns[r["game_id"]].append(dict(r))
    for rid in games:
        for g in games[rid]:
            g["turns"] = turns[g["game_id"]]
    return runs, games


def main():
    conn = sqlite3.connect(DB)
    runs, games = load_runs(conn)
    greedy_rid = next(rid for rid, r in runs.items() if "best honest" in r["label"])
    out: list[str] = []
    w = out.append
    w("# Deeper failure analysis (training levers)\n")
    w(f"Greedy run #{greedy_rid}; {len(runs)} runs total in DB.\n")

    # ---- 1. Non-word taxonomy (greedy run) ----
    w("## 1. Non-word taxonomy — spelling near-miss vs constraint-violation vs chimera")
    nonwords = []  # (emitted, prior_turns)
    for g in games[greedy_rid]:
        prior = []
        for t in g["turns"]:
            if not t["valid"]:
                nonwords.append((t["emitted"], list(prior)))
            else:
                prior.append((t["emitted"], t["feedback"]))
    ham = Counter()
    respects = 0
    near_consistent = 0  # non-word is hamming-1 from a real word that is ALSO consistent w/ prior
    for nw, prior in nonwords:
        if len(nw) != 5:
            ham["not-5-letters"] += 1
            continue
        d = min_hamming_to_real(nw)
        ham[f"hamming-{d}"] += 1
        greens, present, absent = constraints_from(prior)
        if not violation_kinds(nw, greens, present, absent):
            respects += 1
        # is there a real CONSISTENT word within 1 edit of the non-word?
        if d <= 1:
            for w5 in VALID5:
                if hamming5(nw, w5) <= 1 and not violation_kinds(w5, greens, present, absent):
                    near_consistent += 1
                    break
    tot = len([n for n in nonwords if len(n[0]) == 5])
    w(f"- non-words analyzed: {len(nonwords)} ({tot} are 5 letters)")
    for k in sorted(ham):
        w(f"  - {k}: {ham[k]}")
    w(f"- non-words that RESPECT all prior clues (pure spelling failure, not constraint): {_pct(respects, len(nonwords))}")
    w(f"- non-words within 1 letter of a real word that is ALSO clue-consistent (a 1-edit spelling fix away from a legal move): {_pct(near_consistent, tot)}")
    w("")

    # ---- 2. Constraint-violation decomposition (greedy) ----
    w("## 2. Which clue type do VALID-but-inconsistent guesses break?")
    kind_counts = Counter()
    n_violations = 0
    for g in games[greedy_rid]:
        prior = []
        for t in g["turns"]:
            if t["valid"]:
                if not t["consistent_w_prior"] and prior:
                    greens, present, absent = constraints_from(prior)
                    ks = violation_kinds(t["emitted"], greens, present, absent)
                    n_violations += 1
                    for k in ks:
                        kind_counts[k] += 1
                prior.append((t["emitted"], t["feedback"]))
    w(f"- valid guesses that violate prior clues: {n_violations}")
    for k, c in kind_counts.most_common():
        w(f"  - {k}: {c} ({100 * c / max(n_violations, 1):.0f}% of violations)")
    w("")

    # ---- 3. Hard-core vs recoverable targets across ALL runs ----
    w("## 3. Hard-core (lost everywhere) vs recoverable (won only by sampling)")
    solved_by = defaultdict(set)  # target -> set of run_ids that solved it
    all_targets = set()
    sampling_rids = [rid for rid, r in runs.items() if "sample" in r["label"]]
    for rid in [greedy_rid] + sampling_rids:
        for g in games[rid]:
            all_targets.add(g["target"])
            if g["solved"]:
                solved_by[g["target"]].add(rid)
    hardcore = sorted(t for t in all_targets if not solved_by[t])
    greedy_solved = {g["target"] for g in games[greedy_rid] if g["solved"]}
    recoverable = sorted(t for t in all_targets if solved_by[t] and t not in greedy_solved)
    w(f"- targets evaluated: {len(all_targets)}; runs considered: greedy + {len(sampling_rids)} sampling")
    w(f"- **hard-core (never solved in ANY run): {_pct(len(hardcore), len(all_targets))}**")
    w(f"- recoverable (won by >=1 sampling run, lost by greedy): {_pct(len(recoverable), len(all_targets))}")
    w(f"  examples hard-core: {', '.join(hardcore[:15])}")
    w(f"  examples recoverable: {', '.join(recoverable[:15])}")
    w("")

    # ---- 4. Vocabulary probe: does the target EVER appear in the model's output? ----
    w("## 4. Vocabulary gap — does the model ever PRODUCE the target (guess or CoT), any run?")
    produced = defaultdict(bool)
    for rid in [greedy_rid] + sampling_rids:
        for g in games[rid]:
            tgt = g["target"]
            for t in g["turns"]:
                cands = json.loads(t["cot_candidates"])
                if t["emitted"] == tgt or tgt in cands:
                    produced[tgt] = True
    never_produced = sorted(t for t in all_targets if not produced[t])
    hc_never = [t for t in hardcore if not produced[t]]
    w(f"- targets the model NEVER produced anywhere (pure vocab gap): {_pct(len(never_produced), len(all_targets))}")
    w(f"- of the hard-core targets, how many were never produced: {_pct(len(hc_never), len(hardcore))}")
    w(f"  -> hard-core that the model CAN produce somewhere (deduction/commit gap, not vocab): "
      f"{_pct(len(hardcore) - len(hc_never), len(hardcore))}")
    w(f"  examples never-produced: {', '.join(never_produced[:15])}")
    # rare-letter share of never-produced vs all
    def rare_share(words):
        rare = set("jqxzvkwf")
        return sum(any(c in rare for c in w) for w in words) / max(len(words), 1)
    w(f"- rare-letter share: never-produced {rare_share(never_produced):.2f} vs all targets {rare_share(all_targets):.2f}")
    w("")

    # ---- 5. CoT-at-determined: what does it emit/think when the answer is pinned? ----
    w("## 5. When the answer is pinned (<=3 consistent) and the turn is missed — what happened?")
    det_emit = Counter()
    det_total = 0
    for g in games[greedy_rid]:
        for t in g["turns"]:
            if t["consistent_set_before"] <= 3 and t["emitted"] != g["target"]:
                det_total += 1
                if not t["valid"]:
                    det_emit["non-word"] += 1
                elif not t["consistent_w_prior"]:
                    det_emit["valid but violates clues"] += 1
                else:
                    det_emit["valid + consistent but wrong word"] += 1
    w(f"- determined-state misses: {det_total}")
    for k, c in det_emit.most_common():
        w(f"  - {k}: {c} ({100 * c / max(det_total, 1):.0f}%)")
    w("")

    report = "\n".join(out)
    print(report)
    Path("evals/analysis2.md").write_text(report + "\n", encoding="utf-8")
    print("\n[eval_analyze2] wrote evals/analysis2.md")
    conn.close()


if __name__ == "__main__":
    main()
