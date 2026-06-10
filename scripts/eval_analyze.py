"""Deep failure-pattern analysis over a stored eval version (see eval_db.py).

Reads one run from ``evals/evals.db`` and breaks the games down by failure mode, turn position,
deduction difficulty (consistent-set size), clue-respect, spelling-under-constraints, target shape,
and CoT quality. Prints a report and writes ``evals/analysis_run<id>.md``.

Usage:
    uv run python scripts/eval_analyze.py                 # latest run
    uv run python scripts/eval_analyze.py --run-id 3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wordle_slm.data import load_valid_guesses

DB_PATH = Path("evals/evals.db")
VALID = set(load_valid_guesses())


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0 (-)"


def _bucket(size: int) -> str:
    if size <= 1:
        return "1 (determined)"
    if size <= 3:
        return "2-3"
    if size <= 10:
        return "4-10"
    if size <= 50:
        return "11-50"
    return "51+"


def load_run(conn: sqlite3.Connection, run_id: int | None) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    if run_id is None:
        row = conn.execute("SELECT * FROM eval_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    else:
        row = conn.execute("SELECT * FROM eval_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise SystemExit("no run found")
    run = dict(row)
    games = [
        dict(r) for r in conn.execute("SELECT * FROM eval_games WHERE run_id=?", (run["run_id"],))
    ]
    turns_by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in conn.execute(
        "SELECT t.* FROM eval_turns t JOIN eval_games g ON t.game_id=g.game_id WHERE g.run_id=? ORDER BY t.turn_index",
        (run["run_id"],),
    ):
        turns_by_game[r["game_id"]].append(dict(r))
    for g in games:
        g["turns"] = turns_by_game[g["game_id"]]
    return {"run": run, "games": games}


def analyze(data: dict[str, Any]) -> str:
    run, games = data["run"], data["games"]
    out: list[str] = []
    w = out.append

    won = [g for g in games if g["solved"]]
    lost = [g for g in games if not g["solved"]]
    nonword = [g for g in lost if g["outcome"] == "lost_nonword"]
    runout = [g for g in lost if g["outcome"] == "lost_runout"]

    w(f"# Failure analysis — run #{run['run_id']}: {run['label']}")
    w("")
    w(f"- checkpoint `{run['checkpoint']}` (sha {run['checkpoint_sha']}, git {run['git_sha']})")
    w(
        f"- protocol: `{json.loads(run['protocol'])['decode']}`, dict@inference="
        f"{json.loads(run['protocol'])['dictionary_at_inference']}"
    )
    avg_turns = f"{run['avg_turns_win']:.2f}" if run["avg_turns_win"] is not None else "n/a"
    w(
        f"- **win {run['win_rate']:.3f} ({len(won)}/{len(games)})** · valid {run['valid_rate']:.3f} · "
        f"clue-respect {run['clue_respect']:.3f} · avg-turns(win) {avg_turns}"
    )
    w("")

    # ---- §1 outcome split ----
    w("## 1. Where games go")
    w(f"- **won**: {_pct(len(won), len(games))}")
    w(
        f"- **lost — non-word (spelling)**: {_pct(len(nonword), len(games))}  "
        f"= {_pct(len(nonword), len(lost))} of losses"
    )
    w(
        f"- **lost — ran out of 6 valid turns (deduction)**: {_pct(len(runout), len(games))}  "
        f"= {_pct(len(runout), len(lost))} of losses"
    )
    w("")

    # ---- §2 spelling failures: when + under what pressure ----
    w("## 2. Spelling failures (non-word emitted -> instant loss)")
    nw_turn = Counter(
        g["n_valid_turns"] for g in nonword
    )  # valid turns before the killing non-word
    w("Valid turns survived before the non-word that ended the game:")
    for k in sorted(nw_turn):
        w(f"  - after {k} valid turn(s): {nw_turn[k]} games")
    # constraint tightness at the non-word turn
    tight = []
    for g in nonword:
        bad = [t for t in g["turns"] if not t["valid"]]
        if bad:
            tight.append(bad[-1]["consistent_set_before"])
    if tight:
        w(
            f"- consistent-set size when it failed to spell: median {int(statistics.median(tight))}, "
            f"min {min(tight)}, max {max(tight)}"
        )
        w(
            f"  - failed with <=3 answers left (tight): {_pct(sum(1 for x in tight if x <= 3), len(tight))}"
        )
    w("")

    # ---- §3 deduction failures: was the answer pin-down-able? ----
    w("## 3. Deduction failures (ran out of valid turns)")
    if runout:
        buckets = Counter(_bucket(g["final_consistent_set"]) for g in runout)
        for b in ["1 (determined)", "2-3", "4-10", "11-50", "51+"]:
            if buckets.get(b):
                w(f"  - final consistent set {b}: {buckets[b]} games")
        det = sum(1 for g in runout if g["final_consistent_set"] <= 3)
        w(
            f"- **answer narrowed to <=3 but still missed: {_pct(det, len(runout))}** (the commit/deduction wall)"
        )
    else:
        w("  - none")
    w("")

    # ---- §4 spelling vs constraint tightness (all turns) ----
    w("## 4. Does tightness cause the spelling failures? P(non-word | consistent-set before guess)")
    by_bucket_total: Counter[str] = Counter()
    by_bucket_bad: Counter[str] = Counter()
    for g in games:
        for t in g["turns"]:
            b = _bucket(t["consistent_set_before"])
            by_bucket_total[b] += 1
            if not t["valid"]:
                by_bucket_bad[b] += 1
    for b in ["51+", "11-50", "4-10", "2-3", "1 (determined)"]:
        tot = by_bucket_total.get(b, 0)
        if tot:
            w(f"  - set {b:15s}: non-word rate {_pct(by_bucket_bad.get(b, 0), tot)}")
    w("")

    # ---- §5 clue-respect by turn ----
    w("## 5. Clue-respect by turn (valid guesses consistent with all prior feedback)")
    rt_tot: Counter[int] = Counter()
    rt_ok: Counter[int] = Counter()
    for g in games:
        for t in g["turns"]:
            if t["valid"]:
                rt_tot[t["turn_index"]] += 1
                rt_ok[t["turn_index"]] += t["consistent_w_prior"]
    for i in sorted(rt_tot):
        w(f"  - turn {i}: respect {_pct(rt_ok[i], rt_tot[i])}")
    w("")

    # ---- §6 target shape ----
    w("## 6. Win rate by target shape")
    for key, label in [
        ("target_dup_letters", "duplicate letters"),
        ("target_rare_letters", "rare letters (count)"),
    ]:
        grp_tot: Counter[int] = Counter()
        grp_win: Counter[int] = Counter()
        for g in games:
            grp_tot[g[key]] += 1
            grp_win[g[key]] += g["solved"]
        w(f"- by {label}:")
        for v in sorted(grp_tot):
            w(f"    {v}: win {_pct(grp_win[v], grp_tot[v])}")
    w("")

    # ---- §7 CoT quality ----
    w("## 7. CoT quality (the model's ephemeral candidate words)")
    n_cot_turns = n_with_cands = 0
    cand_counts = []
    valid_frac = []
    consistent_frac = []
    final_in_cands = 0
    answer_in_cands_when_det = answer_det_turns = 0
    for g in games:
        target = g["target"]
        for t in g["turns"]:
            cands = json.loads(t["cot_candidates"])
            n_cot_turns += 1
            if cands:
                n_with_cands += 1
                cand_counts.append(len(cands))
                valid_frac.append(sum(c in VALID for c in cands) / len(cands))
                final_in_cands += int(t["emitted"] in cands)
            if t["consistent_set_before"] <= 3:
                answer_det_turns += 1
                answer_in_cands_when_det += int(target in cands)
    w(f"- turns with >=1 CoT candidate: {_pct(n_with_cands, n_cot_turns)}")
    if cand_counts:
        w(f"- candidates per turn: mean {statistics.mean(cand_counts):.2f}")
        w(f"- CoT candidates that are real words: mean {100 * statistics.mean(valid_frac):.1f}%")
        w(f"- final guess appeared in its own CoT: {_pct(final_in_cands, n_with_cands)}")
    if answer_det_turns:
        w(
            f"- when answer was determined (<=3 left), the answer was among CoT candidates: "
            f"{_pct(answer_in_cands_when_det, answer_det_turns)}"
        )
    w("")

    # ---- §8 opener ----
    w("## 8. First guess (opener) diversity")
    openers = Counter(g["turns"][0]["emitted"] for g in games if g["turns"])
    for word, c in openers.most_common(8):
        w(f"  - {word}: {c}")
    w("")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, default=None)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    data = load_run(conn, args.run_id)
    report = analyze(data)
    print(report)
    out_path = Path(f"evals/analysis_run{data['run']['run_id']}.md")
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\n[eval_analyze] wrote {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
