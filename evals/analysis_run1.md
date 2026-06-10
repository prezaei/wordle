# Failure analysis — run #1: validity_max_v4 (best honest free-gen)

- checkpoint `runs/validity_max_v4.pt` (sha cbf5787bdfd2ed37, git e67b0db)
- protocol: `greedy free-gen over {26 letters, THINK, commit}`, dict@inference=False
- **win 0.332 (122/367)** · valid 0.855 · clue-respect 0.859 · avg-turns(win) 4.32

## 1. Where games go
- **won**: 122/367 (33.2%)
- **lost — non-word (spelling)**: 228/367 (62.1%)  = 228/245 (93.1%) of losses
- **lost — ran out of 6 valid turns (deduction)**: 17/367 (4.6%)  = 17/245 (6.9%) of losses

## 2. Spelling failures (non-word emitted -> instant loss)
Valid turns survived before the non-word that ended the game:
  - after 1 valid turn(s): 4 games
  - after 2 valid turn(s): 62 games
  - after 3 valid turn(s): 85 games
  - after 4 valid turn(s): 53 games
  - after 5 valid turn(s): 24 games
- consistent-set size when it failed to spell: median 1, min 1, max 16
  - failed with <=3 answers left (tight): 201/228 (88.2%)

## 3. Deduction failures (ran out of valid turns)
  - final consistent set 1 (determined): 8 games
  - final consistent set 2-3: 7 games
  - final consistent set 4-10: 2 games
- **answer narrowed to <=3 but still missed: 15/17 (88.2%)** (the commit/deduction wall)

## 4. Does tightness cause the spelling failures? P(non-word | consistent-set before guess)
  - set 51+            : non-word rate 0/571 (0.0%)
  - set 11-50          : non-word rate 11/166 (6.6%)
  - set 4-10           : non-word rate 16/227 (7.0%)
  - set 2-3            : non-word rate 54/243 (22.2%)
  - set 1 (determined) : non-word rate 147/365 (40.3%)

## 5. Clue-respect by turn (valid guesses consistent with all prior feedback)
  - turn 0: respect 367/367 (100.0%)
  - turn 1: respect 346/363 (95.3%)
  - turn 2: respect 217/299 (72.6%)
  - turn 3: respect 126/188 (67.0%)
  - turn 4: respect 67/90 (74.4%)
  - turn 5: respect 31/37 (83.8%)

## 6. Win rate by target shape
- by duplicate letters:
    0: win 88/259 (34.0%)
    1: win 34/108 (31.5%)
- by rare letters (count):
    0: win 62/175 (35.4%)
    1: win 48/155 (31.0%)
    2: win 11/34 (32.4%)
    3: win 1/3 (33.3%)

## 7. CoT quality (the model's ephemeral candidate words)
- turns with >=1 CoT candidate: 1572/1572 (100.0%)
- candidates per turn: mean 1.95
- CoT candidates that are real words: mean 85.3%
- final guess appeared in its own CoT: 1570/1572 (99.9%)
- when answer was determined (<=3 left), the answer was among CoT candidates: 111/608 (18.3%)

## 8. First guess (opener) diversity
  - stare: 367

