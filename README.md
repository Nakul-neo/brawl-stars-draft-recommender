# Brawl Stars Draft Recommender and Win Predictor

A machine learning tool that provides data-backed pick recommendations during the Brawl Stars ranked drafting phase, and predicts win probability once a draft is complete.

**Live demo:** https://brawl-stars-draft-recommender.onrender.com

> Note: the demo runs on a free hosting tier and sleeps after inactivity. The first request may take up to 60 seconds to wake the instance.

---

## Overview

In Brawl Stars ranked mode, two teams alternate picking characters ("brawlers") while seeing the opponent's picks. Picks interact through counters, synergies and map suitability, but players make these decisions in seconds with no data available to them.

This project tests whether draft composition measurably affects match outcome, and if so, surfaces that signal as a usable tool.

**The hypothesis was testable and it held.** A model trained to predict the winner at nine different stages of the draft improves monotonically as more picks are revealed — from 0.575 AUC-ROC with one brawler on the board to 0.695 with the full draft visible. That climb is attributable to draft information alone, since nothing else about the input changes.

---

## Results

Evaluated on a held-out test set of 17,686 matches, split at the match level before any feature generation.

| Draft state (my picks, opponent picks) | Test rows | AUC-ROC | Accuracy |
|---|---|---|---|
| (1, 0) | 35,372 | 0.575 | 0.548 |
| (0, 1) | 35,372 | 0.574 | 0.547 |
| (1, 1) | 35,372 | 0.626 | 0.582 |
| (2, 1) | 35,372 | 0.641 | 0.592 |
| (1, 2) | 35,372 | 0.644 | 0.595 |
| (2, 2) | 35,372 | 0.662 | 0.609 |
| (3, 2) | 35,372 | 0.678 | 0.622 |
| (2, 3) | 35,372 | 0.676 | 0.620 |
| **(3, 3) — full draft** | **35,372** | **0.6950** | **0.6350** |

Pooled across all draft states: **0.6443 AUC-ROC**.

The full-draft figure is reported as the headline because it corresponds to the state the application actually displays to a user. The pooled figure averages in early-draft states where almost nothing is revealed and prediction is close to unanswerable.

### Interpreting these numbers

Draft-only outcome prediction has a low ceiling in team games. The draft is a real but minor factor next to player skill and in-match execution, neither of which this model observes. Published draft-only models for Dota 2 and League of Legends typically report 0.60 to 0.65 AUC. A substantially higher figure on this task would indicate data leakage rather than better modelling.

---

## Dataset

| Property | Value |
|---|---|
| Source | Brawl Stars public API battle logs |
| Raw rows | 519,325 |
| After filtering and de-duplication | 117,905 unique matches |
| Date range | 16 December 2021 to 11 February 2023 |
| Brawlers | 64 |
| Maps | 43 |
| Game modes | 6 (gemGrab, brawlBall, knockout, heist, bounty, hotZone) |

### Cleaning pipeline

Raw battle logs are recorded per player, so a single match can appear up to six times, and the outcome is recorded from the logging player's perspective rather than at team level. Three steps resolve this:

1. **Filtering.** Ranked matches only (casual modes have no draft phase), 3v3 modes only, parseable team structures, decisive results. This reduces 519,325 rows to 290,570.
2. **De-duplication.** Each match is fingerprinted by timestamp, mode and the unordered set of all six player tags. An unordered set is required because different players' logs list the teams in different orders. This reduces 290,570 rows to 117,905 unique matches.
3. **Outcome reconstruction.** Team-level winner is inferred exactly by checking whether the logging player's ID appears in team A's tag set and combining that with their individual result.

---

## Methodology

### Feature engineering

231 features per row:

| Block | Count | Contents |
|---|---|---|
| Brawler one-hot (own side) | 64 | Presence indicator per brawler |
| Brawler one-hot (opponent) | 64 | Presence indicator per brawler |
| Map one-hot | 43 | |
| Mode one-hot | 6 | |
| Role counts (own / opponent) | 14 | Tallies across 7 hand-assigned roles |
| Engineered scalars | 40 | Win rates, counter and synergy aggregates, power, trophies, time |

The scalar block carries most of the predictive power. The three highest-ranked features by gain are `mode_diff`, `counter_mean` and `map_diff` — all constructed differences that do not exist in the raw data. Explicit own-minus-opponent differences are included because tree models split on a single feature at a time and cannot directly compare two columns.

### Counter and synergy modelling

Five win-rate tables are built from training matches only: overall, per-map, per-mode, per-matchup (directional, across teams) and per-synergy (symmetric, within a team).

Three corrections are applied:

**Empirical Bayes shrinkage.** Raw pair win rates are dominated by sampling noise — one brawler pairing in the dataset shows a 78.8% win rate from only 85 matches. Rather than choosing shrinkage constants by hand, each table's constant is derived by decomposing observed variance into signal and sampling noise, then setting `k = p(1-p) / var_signal`. Estimated values:

| Table | Constant |
|---|---|
| Overall | 166 |
| Map | 104 |
| Mode | 117 |
| Matchup | 58 |
| Synergy | 70 |

Applied to the example above, 0.788 from 85 matches shrinks to 0.658, while a brawler's overall win rate of 0.572 from 11,335 matches barely moves, to 0.570.

**Residualisation.** A pair's win rate largely reflects the individual strength of both brawlers, which the model already receives separately. Subtracting that expectation isolates the interaction: the pair above resolves to a residual synergy of +0.121 over what solo strength predicts.

**Mean, minimum and maximum.** A full board produces nine cross-team matchups. Collapsing these to a single mean hides the one hard counter that often decides a draft. All counter and synergy blocks carry mean, min and max, in both raw and residualised form.

### Preventing leakage

Two structural safeguards, both added after an earlier version of this project was found to be leaking (see below):

- **Match-level splitting.** Each match generates 18 training rows (9 draft states x 2 team perspectives). The train/validation/test split is performed on match IDs *before* any rows are generated, so all rows derived from one match remain on the same side of the split.
- **Out-of-fold target encoding.** Every win-rate feature is derived from match outcomes. Training rows are built from statistics fitted on the other four of five folds, so no match contributes to the statistics used to predict itself. Validation and test rows use statistics fitted on the full training set, which they never contributed to.

### Draft state coverage

The recommender evaluates a candidate pick by constructing the board state that would follow it, which produces states where the user's revealed picks exceed the opponent's. Training covers all nine reachable states:

```
DRAFT_STATES = [(1,0), (0,1), (1,1), (2,1), (1,2), (2,2), (3,2), (2,3), (3,3)]
```

The `(0,0)` state is deliberately excluded. With nothing revealed, both perspectives of a match produce identical feature vectors with opposite labels, making it unlearnable by construction with a theoretical ceiling of 0.500 AUC.

### Model

XGBoost gradient-boosted trees, trained on 1,485,594 rows.

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 3000 cap, stopped at 2100 | Early stopping on validation logloss |
| `max_depth` | 6 | Deep enough for interactions, shallow enough to stay a weak learner |
| `learning_rate` | 0.03 | Small corrections, many trees |
| `min_child_weight` | 60 | Prevents leaves fitted to sparse, noisy pair statistics |
| `reg_lambda` | 5.0 | L2 penalty on leaf values |
| `subsample` | 0.8 | Row sampling for decorrelation |
| `colsample_bytree` | 0.7 | Feature sampling for decorrelation |

Probabilities are calibrated with isotonic regression fitted on the validation split. Calibration left AUC unchanged at 0.6443 and moved the Brier score from 0.23115 to 0.23118, indicating the model was already well calibrated — expected given the exactly balanced label construction and the regularisation applied.

The shipped model is refit on all 117,905 matches after evaluation is complete, using the tree count determined by early stopping on the honest split and retaining out-of-fold encoding. No reported metric comes from the refit model.

---

## Methodology correction from an earlier version

An initial version of this project reported **0.730 AUC at a full draft** — higher than the 0.695 reported here. That figure was inflated by two forms of data leakage:

1. **Row-level train/test splitting.** All training rows were generated before splitting, so approximately 12 of each match's 14 rows landed in training and 2 in test. The model was evaluated on matches it had effectively memorised.
2. **Statistics computed across the full dataset.** All win-rate tables were built over every match, including test matches, before any split. Features describing a test match were therefore partly computed from that match's own outcome.

A third defect was functional rather than statistical: the earlier version trained only on states where `n_my <= n_opp`, leaving three of six real draft turns — including every first pick — unsupported. The visible symptom was that all first-pick recommendations returned within a fraction of a percent of 50%. After the fix, first-pick recommendations span a 24.8 point range (35.1% to 59.9%).

The current version reports a lower headline figure because the higher one was measuring the flaws above.

---

## Repository contents

```
app.py                 Inference-only Gradio application
model_bundle.joblib    Calibrated model, shrunk statistics tables, encoders (8.3 MB)
requirements.txt       Pinned dependencies
runtime.txt            Python version pin
```

The application performs no training. It loads the serialised bundle at startup and builds a single feature vector per candidate pick. The encoder dictionaries are serialised alongside the model because the model learned positional meaning within the feature vector; rebuilding the encoding independently would produce silently incorrect predictions.

---

## Running locally

```bash
git clone https://github.com/Nakul-neo/brawl-stars-draft-recommender.git
cd brawl-stars-draft-recommender
pip install -r requirements.txt
python app.py
```

The application starts on port 7860 by default, or on the port specified by the `PORT` environment variable.

Python 3.11 is required. Python 3.13 and later remove the `audioop` module, which a transitive dependency of the pinned Gradio version imports.

---

## Deployment

Deployed as a web service on Render's free tier, built directly from this repository.

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `python app.py` |
| Python version | 3.11.9 (set via `PYTHON_VERSION`) |

All dependencies are pinned to exact versions, including transitive ones. The pinned Gradio release is incompatible with current releases of `huggingface_hub`, `fastapi`, `starlette`, `pydantic` and `jinja2`, so these are constrained explicitly rather than resolved automatically.

---

## Limitations

- **Training data is from December 2021 to February 2023.** The game has been rebalanced since, and brawlers released after this window are absent entirely. Recommendations reflect the meta of that period, not the current one.
- **Part of the predictive signal is not about drafting.** Power level and trophy features rank highly in feature importance. These are proxies for account investment and player skill rather than draft quality, so a portion of model performance reflects detecting that one team is more developed.
- **Recommendations are greedy, not strategic.** Each pick is scored on the board state immediately following it, without modelling the opponent's response. Proper drafting requires lookahead; a pick that is locally optimal may invite an obvious counter on the next turn.
- **Thin data on rare combinations.** Shrinkage handles small samples responsibly by pulling them toward neutral, but the model is close to agnostic about genuinely rare pairings. New brawlers with no history read as average everywhere.
- **Random rather than time-based split.** The model may learn from later matches to predict earlier ones, which is not possible in deployment. A time-based split would give a more conservative and more realistic estimate.

---

## Possible extensions

- One-ply lookahead in the recommender, scoring each candidate against the opponent's best response rather than the immediate board state.
- Brawler embeddings to generalise across rare pairings, at the cost of the current approach's interpretability.
- A variant trained without power and trophy features to isolate pure draft signal.
- Scheduled retraining triggered by game balance patches, with full-draft AUC monitored on recent matches as a drift signal.

---

## Acknowledgements

Match data originates from the Brawl Stars public API. Brawl Stars is a trademark of Supercell. This project is unaffiliated with and unendorsed by Supercell.
