"""Parallel teacher-game generation -> pickle (CPU-only, NO torch) so the GPU isn't left idle during the
single-threaded, consistency-scan-heavy teacher rollouts. Builds the same secret set as format_sweep
(incl. the VM_EXPAND common-word lever, held-out excluded) and dumps games for format_sweep to load via
VM_GAMES_PKL. Run this first; then run format_sweep with VM_GAMES_PKL pointed at the output.

Env: VM_SECRETS, VM_EXPAND, VM_TEACHER(2), VM_NPROC(12), VM_GAMES_PKL(out path).
"""

from __future__ import annotations

import os
import pickle
from multiprocessing import get_context

import teacher_gen
from wordle_slm.data import load_valid_guesses, split


def main():
    train, held = split(seed=0)
    VALID = load_valid_guesses()
    cap = int(os.environ.get("VM_SECRETS", str(len(train))))
    secrets = tuple(train[:cap])
    held_set = set(held)
    guess_pool = list(VALID)  # words the teacher may guess
    if os.environ.get("VM_EXPAND") == "1":
        import wordfreq
        common = [w for w in wordfreq.top_n_list("en", 100000) if len(w) == 5 and w.isalpha() and w.isascii()]
        secrets = tuple(w for w in common if w in set(VALID) and w not in held_set)[:cap]
        assert not (set(secrets) & held_set), "HELD-OUT LEAKED"
        print(f"EXPAND: {len(secrets)} common+valid secrets (held-out excluded); "
              f"train-answers={len(set(secrets) & set(train))}, new={len(set(secrets) - set(train) - held_set)}", flush=True)
    if os.environ.get("VM_FULLDICT") == "1":  # DATA LEVER #2: deduction over the FULL dict (not just answers)
        import random as _r
        pool = [w for w in VALID if w not in held_set]  # held-out (VAL+TEST) excluded from secrets AND guesses
        secrets = tuple(_r.Random(0).sample(pool, min(cap, len(pool))))
        guess_pool = pool  # the teacher can NEVER guess a held-out word -> held-out is never a training target
        assert not (set(secrets) & held_set), "HELD-OUT LEAKED INTO SECRETS"
        assert not (set(guess_pool) & held_set), "HELD-OUT LEAKED INTO GUESS POOL"
        print(f"FULLDICT: {len(secrets)} secrets from the full dict; held-out {len(held)} excluded from BOTH "
              f"secrets and the teacher guess pool; answers-in-set={len(set(secrets) & set(train))}", flush=True)
    safe = tuple(o for o in ("salet", "crane", "slate", "trace", "stare", "raise", "crate") if o not in held_set)
    TEACHER = int(os.environ.get("VM_TEACHER", "2"))
    nproc = min(int(os.environ.get("VM_NPROC", "12")), os.cpu_count() or 4)
    chunks = [secrets[i::nproc] for i in range(nproc)]
    tasks = [(c, 300 + s * 1000 + ci, safe, guess_pool, secrets, 0.5)
             for s in range(TEACHER) for ci, c in enumerate(chunks) if c]
    print(f"gen {len(secrets)} secrets x{TEACHER} passes across {nproc} procs ({len(tasks)} tasks) …", flush=True)
    import time
    t0 = time.time()
    with get_context("spawn").Pool(nproc) as pool:
        games = [g for r in pool.map(teacher_gen.gen_chunk, tasks) for g in r]
    out = os.environ.get("VM_GAMES_PKL", "runs/games.pkl")
    with open(out, "wb") as f:
        pickle.dump(games, f)
    print(f"wrote {len(games)} games -> {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
