# Deeper failure analysis (training levers)

Greedy run #1; 10 runs total in DB.

## 1. Non-word taxonomy — spelling near-miss vs constraint-violation vs chimera
- non-words analyzed: 228 (228 are 5 letters)
  - hamming-1: 226
  - hamming-2: 2
- non-words that RESPECT all prior clues (pure spelling failure, not constraint): 177/228 (77.6%)
- non-words within 1 letter of a real word that is ALSO clue-consistent (a 1-edit spelling fix away from a legal move): 134/228 (58.8%)

## 2. Which clue type do VALID-but-inconsistent guesses break?
- valid guesses that violate prior clues: 190
  - missing_present: 97 (51% of violations)
  - dropped_green: 62 (33% of violations)
  - reused_gray: 58 (31% of violations)

## 3. Hard-core (lost everywhere) vs recoverable (won only by sampling)
- targets evaluated: 367; runs considered: greedy + 9 sampling
- **hard-core (never solved in ANY run): 79/367 (21.5%)**
- recoverable (won by >=1 sampling run, lost by greedy): 166/367 (45.2%)
  examples hard-core: aback, abyss, apart, array, avoid, begat, belch, canoe, cargo, chute, cloak, cynic, dummy, duvet, ebony
  examples recoverable: abort, actor, allot, aloft, altar, amiss, amply, annul, apply, awash, awful, banal, baste, batch, baton

## 4. Vocabulary gap — does the model ever PRODUCE the target (guess or CoT), any run?
- targets the model NEVER produced anywhere (pure vocab gap): 75/367 (20.4%)
- of the hard-core targets, how many were never produced: 75/79 (94.9%)
  -> hard-core that the model CAN produce somewhere (deduction/commit gap, not vocab): 4/79 (5.1%)
  examples never-produced: aback, abyss, apart, array, avoid, belch, canoe, cargo, chute, cloak, cynic, duvet, ebony, edict, etude
- rare-letter share: never-produced 0.49 vs all targets 0.38

## 5. When the answer is pinned (<=3 consistent) and the turn is missed — what happened?
- determined-state misses: 497
  - non-word: 201 (40%)
  - valid + consistent but wrong word: 156 (31%)
  - valid but violates clues: 140 (28%)

