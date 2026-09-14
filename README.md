# Week 9 — Operational ML & Explainability: Predictive Maintenance
## (Built on the real Microsoft Azure Predictive Maintenance dataset)

This project builds an end-to-end **predictive maintenance** pipeline on real, publicly available
industrial telemetry: a model that predicts whether a machine will experience an unplanned
component failure within the next 24 hours, and explains *why* it made that call in language a
maintenance technician — not just a data scientist — can act on.

It was built for a Week 9 "Operational ML & Explainability" assignment with three parts: a
technical notebook, a non-technical explainer report + video, and a capstone progress update.

---

## The dataset

**Source:** [Microsoft Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
— 100 machines, one full year (Jan 2015–Jan 2016) of hourly telemetry, plus real error logs,
maintenance/component-replacement records, machine metadata, and recorded failures.

| File | Rows | Contents |
|---|---|---|
| `PdM_telemetry.csv` (zipped) | 876,100 | Hourly voltage, rotation, pressure, vibration |
| `PdM_errors.csv` | 3,919 | Timestamped non-fatal error codes (`error1`–`error5`) |
| `PdM_maint.csv` | 3,286 | Timestamped component replacement events (`comp1`–`comp4`) |
| `PdM_failures.csv` | 761 | Timestamped **actual recorded failures** per component |
| `PdM_machines.csv` | 100 | Machine `model` and `age` |

This is real, messy(ish), publicly benchmarked data — not a synthetic stand-in — so every number
in the notebook, PDF, and video reflects genuine model behavior on genuine (if unusually clean)
industrial telemetry.

## What's in this repository

```
project/
├── README.md                              ← you are here
├── week9_operational_ml.ipynb             ← Part A: full technical notebook
├── Week9_Model_Explainer_[YourName].pdf   ← Part B: 2-page non-technical written report
├── video_script.md                        ← Part B: script for the 3-min video walkthrough
├── capstone_week9_update.md               ← Part C: capstone progress update
├── build_notebook.py                      ← generates the notebook programmatically
├── build_pdf.py                           ← generates the PDF report
├── data/                                  ← the 5 source CSVs (telemetry kept zipped for size)
└── assets/                                ← chart PNGs exported from the notebook, reused in the PDF
```

> **Note on `Week9_Video_[YourName].mp4`:** recording your own screen + voice isn't something I can
> do for you, so `video_script.md` has a complete, timed script with exact narration mapped to
> specific plots — including a walkthrough of a genuine flagged machine (Machine 15) that the
> model correctly identified as high-risk before its real recorded failure.

Rename the `[Your Name]` / `[YourName]` placeholders in the PDF, script, and report before
submitting.

## Part A — Technical notebook (`week9_operational_ml.ipynb`)

| Step | What it does | Why |
|---|---|---|
| 1. Problem definition | `failure_within_24h`: will any of a machine's 4 components fail in the next 24h | Recall-sensitive, imbalanced classification — the standard framing for this dataset |
| 2. Feature engineering | 3h/24h rolling mean & std per telemetry channel, 24h rolling error-code counts, days-since-last-maintenance per component, machine metadata | Turns raw noisy hourly readings into trend signals; explicitly backward-looking only (no leakage) |
| 3. Imbalance check | ~2.3% of snapshots precede a failure within 24h | Confirms this is a genuinely imbalanced, recall-sensitive problem |
| 4. Chronological split & CV | Train on Jan–Sep 2015, test on Oct–Dec 2015; `TimeSeriesSplit` (not random K-fold) for cross-validation | Prevents a machine's own future from leaking into its training data — a real risk with time-series sensor data |
| 5. Model training | **Random Forest** and **XGBoost**, class-weighted (primary) + a direct **SMOTE** comparison | Two independent model families; class weighting chosen as the computationally efficient primary technique at this data volume, with SMOTE demonstrated and compared as required |
| 6. Evaluation | Confusion matrix, precision, recall, F1, ROC-AUC — **not accuracy alone** | Accuracy is meaningless at 2.3% positive rate |
| 7. Clustering (bonus) | K-Means on machine-level average operating profiles (k chosen via silhouette score), PCA visualization | Surfaces fleet-level risk segments for batch maintenance planning |
| 8. Explainability | Feature importance + **SHAP** (`TreeExplainer`): global summary, and local force/waterfall plots for one real, correctly-flagged high-risk machine | Global importance shows fleet-wide patterns; local SHAP shows exactly why *this* machine was flagged — verified against its real subsequent failure |
| 9. Documentation | Markdown throughout explaining *why* each metric/technique was chosen, plus an explicit leakage check and a caveat about this dataset's unusually clean signal | Required for grading, and genuinely useful for anyone extending this pipeline |

Every cell has already been executed — outputs, metrics, and all plots are baked into the `.ipynb`
file. The full pipeline (data load → feature engineering → CV → final models → SHAP → clustering)
runs in well under 5 minutes on a single CPU core.

**Headline results:**
- Class distribution: 97.7% no-failure / 2.3% failure within 24h (realistic extreme imbalance).
- Best model: **XGBoost** — Test ROC-AUC **0.9999**, Precision **97.2%**, Recall **99.6%**, F1 **0.984**.
- Random Forest (comparison): Test ROC-AUC 0.9996, Precision 82.6%, Recall 98.3%.
- SMOTE + XGBoost comparison: ROC-AUC 1.0000 at nearly double the training rows and fit time —
  used as evidence for choosing class weighting as the primary, more efficient technique.
- Top global drivers (feature importance + SHAP, in agreement): **24-hour rolling averages of
  pressure, vibration, voltage, and rotation speed**, followed by **recent error-code counts**.
- Local explanation for a real flagged case (Machine 15, Oct 2015, later confirmed comp2 failure):
  dominated by recent error codes (error5, error2, error3) plus a rotation-speed/vibration uptick.
- K-Means: best k=2 by silhouette score (0.115) — a moderate but real split of the fleet into two
  operating-risk segments.
- **No data leakage found:** every engineered feature is built from strictly backward-looking
  windows, and the train/test split is chronological — verified explicitly in the notebook.

## Part B — Model Explainer (PDF + video)

The **PDF report** (`Week9_Model_Explainer_[YourName].pdf`) is written for the operations team:
- **The Analogy:** the model as a doctor reading vital signs / a triage nurse.
- **The Drivers:** the real feature importance chart, with plain-English explanations of the top 3
  driver groups (24h sensor trend, recent error codes, maintenance recency), plus an explicit,
  honest note about why this dataset's performance is unusually high and what that means for a
  real deployment.
- **Trust & Limits:** a false-positive/false-negative cost table, the real precision/recall numbers
  translated into plain terms, and concrete guidance to treat this as decision support, not
  automation.

The **video script** (`video_script.md`) is a timed (~3 min) narration script covering the
imbalance problem, chronological validation, the SHAP summary plot, and a full walkthrough of the
real Machine 15 case — a genuinely flagged machine that the model identified before its actual
recorded failure.

## Part C — Capstone update

`capstone_week9_update.md` documents: the algorithm choice (XGBoost, selected on real held-out test
performance and training efficiency), how class imbalance was handled (class weighting as primary,
SMOTE as a validated comparison), and a genuinely two-part surprising finding — that error-code
counts rivaled sensor drift in the local explanation for the model's most confident real case, and
that this specific public dataset's near-perfect scores reflect a known property of the benchmark
rather than something to expect from a first pass on noisier real plant data.

## Reproducing this project

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap reportlab jupyter pyarrow

# The five source CSVs live in data/ (telemetry kept zipped for size — the notebook unzips it automatically)

# Regenerate the notebook definition and execute it end-to-end (~5 min on 1 CPU core)
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace week9_operational_ml.ipynb

# Regenerate the PDF report (reads PNGs from assets/, produced by the notebook run)
python build_pdf.py
```

## Limitations & honest caveats (worth reading before reusing this)

- **This benchmark dataset is unusually clean.** ROC-AUC > 0.999 is not typical of real plant
  sensor data; it reflects a documented property of this specific, widely-used public dataset
  (deliberately clear drift signatures ahead of each failure type). We explicitly checked for and
  ruled out data leakage as the explanation. Expect materially lower performance on real,
  noisier sensor data until validated.
- **Chronological split matters.** A random train/test split on this kind of data would likely
  inflate performance further through subtle temporal leakage; we deliberately used a
  forward-in-time split and `TimeSeriesSplit` cross-validation instead.
- **Moderate clustering signal.** The K-Means machine segments (silhouette ≈ 0.115) show a real
  but not dramatically strong 2-cluster split — useful as a planning input, not a hard boundary.
- **Decision support, not automation.** Every deliverable in this project repeats the same
  operational guidance on purpose: this model should rank and prioritize human attention, not
  replace it — especially given how strong its benchmark numbers are.
