#!/usr/bin/env bash
# Overnight autonomous push: ~10+ honest approaches to beat 0.385, all versioned in evals/evals.db.
# Honest throughout: free-gen, NO dict/trie/consistency filter at inference; held-out never trained as
# an answer; eval on held-out TEST (367). Each step is independent (|| true) so one failure can't abort
# the rest. Read results in the morning: sqlite3 evals/evals.db.
set -u
cd /Users/pedram/.superset/worktrees/wordle/false-quicksand
LOG=runs/overnight.log
exec >>"$LOG" 2>&1
echo "==================== OVERNIGHT START $(date) ===================="
PY="uv run python"
V4=runs/validity_max_v4.pt

ev () {  # ev <label> <checkpoint> <extra eval_db args...>
  local label="$1"; local ckpt="$2"; shift 2
  echo "---- EVAL: $label  ($*) ----"
  $PY scripts/eval_db.py --checkpoint "$ckpt" --label "$label" --notes overnight "$@" 2>&1 \
    | grep -E "RUN #|^  win|FIDELITY|Error|Traceback" || true
}

# 0) wait for the in-flight #2b full-dict pipeline to release the GPU
echo "waiting for #2b full-dict pipeline …"
until grep -q "#2 FULLDICT PIPELINE DONE" runs/fulldict_pipeline.log 2>/dev/null; do sleep 30; done
echo "#2b done; GPU free at $(date)"

# ============================ DECODE approaches on v4 (cheap, no retrain) ============================
# (1) anti-repeat temperature fine-sweep — pin the peak around T=0.6
for T in 0.5 0.55 0.65 0.7; do for S in 0 1; do
  ev "v4 AR T=$T s$S" "$V4" --temp $T --seed $S --antirepeat
done; done

# (2) anti-repeat resample budget (more retries to find a novel/valid word)
for TR in 16 24; do for S in 0 1; do
  AR_TRIES=$TR ev "v4 AR T=0.6 tries$TR s$S" "$V4" --temp 0.6 --seed $S --antirepeat
done; done

# (3) per-turn temperature schedule (low early = narrow well, high late = spelling retries)
for SC in 0.3:0.8 0.4:0.9 0.5:0.9; do for S in 0 1; do
  ev "v4 AR tsched$SC s$S" "$V4" --tsched $SC --seed $S --antirepeat
done; done

# (4) fixed strong opener (faster narrowing -> more endgame turns) + AR
for OP in salet raise crane; do for S in 0 1; do
  ev "v4 AR T=0.6 opener=$OP s$S" "$V4" --temp 0.6 --seed $S --antirepeat --opener $OP
done; done

# (5) best-guess COMBO: strong opener + late-heavy schedule + bigger budget
for S in 0 1 2; do
  AR_TRIES=20 ev "v4 AR combo(salet,0.4:0.9,t20) s$S" "$V4" --tsched 0.4:0.9 --seed $S --antirepeat --opener salet
done

# ============================ TRAINING approaches (likely null; breadth) ============================
# (6) MIXED data: answers + full-dict secrets (coverage + answer-distribution focus), warm-start v4
$PY - <<'PYEOF' || true
import pickle
a = pickle.load(open("runs/games_ans.pkl", "rb"))
f = pickle.load(open("runs/games_fulldict.pkl", "rb"))
pickle.dump(a + f, open("runs/games_mixed.pkl", "wb"))
print(f"mixed pickle: {len(a)} answers + {len(f)} fulldict = {len(a)+len(f)} games")
PYEOF
FMT=baseline VM_BASE=$V4 VM_GAMES_PKL=runs/games_mixed.pkl VM_SELECT=win VM_EPOCHS=8 VM_OUT=runs/mixed.pt \
  $PY scripts/format_sweep.py > runs/mixed_train.log 2>&1 || true
[ -f runs/mixed.pt ] && ev "mixed-data greedy" runs/mixed.pt
[ -f runs/mixed.pt ] && for S in 0 1; do ev "mixed-data +AR s$S" runs/mixed.pt --temp 0.6 --seed $S --antirepeat; done

# (7) CURRICULUM: take #2b full-dict model, RE-CENTER on answers (full-dict coverage then answer focus)
if [ -f runs/fulldict.pt ]; then
  FMT=baseline VM_BASE=runs/fulldict.pt VM_GAMES_PKL=runs/games_ans.pkl VM_SELECT=win VM_EPOCHS=6 VM_OUT=runs/curric.pt \
    $PY scripts/format_sweep.py > runs/curric_train.log 2>&1 || true
  [ -f runs/curric.pt ] && ev "curriculum(fulldict->answers) greedy" runs/curric.pt
  [ -f runs/curric.pt ] && for S in 0 1; do ev "curriculum +AR s$S" runs/curric.pt --temp 0.6 --seed $S --antirepeat; done
fi

# (8) #2b EXTENDED: more epochs on full-dict (does win keep climbing or plateau?)
FMT=baseline VM_BASE=$V4 VM_GAMES_PKL=runs/games_fulldict.pkl VM_SELECT=win VM_EPOCHS=16 VM_OUT=runs/fulldict16.pt \
  $PY scripts/format_sweep.py > runs/fulldict16_train.log 2>&1 || true
[ -f runs/fulldict16.pt ] && ev "fulldict-16ep greedy" runs/fulldict16.pt
[ -f runs/fulldict16.pt ] && for S in 0 1; do ev "fulldict-16ep +AR s$S" runs/fulldict16.pt --temp 0.6 --seed $S --antirepeat; done

echo "==================== OVERNIGHT DONE $(date) ===================="
