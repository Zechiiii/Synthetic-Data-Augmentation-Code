# Synthetic Data Augmentation for Imbalanced Learning

Code accompanying the paper **“Synthetic Augmentation in Imbalanced Learning: When It Helps, When It Hurts, and How Much to Add.”**

This repository contains simulation studies, hypothesis-test experiments, a theory-guided procedure for selecting the amount of synthetic minority data, and retrospective applications to MIMIC-III clinical data.

## Overview

Synthetic oversampling can reduce variance in imbalanced classification, but adding too much—or using a biased generator—can increase prediction error. The code studies three local regimes:

- **Local symmetry:** little or no synthetic augmentation is expected to help.
- **Local asymmetry with a realistic generator:** augmentation may help, and theory can guide how much to add.
- **Local asymmetry with a biased generator:** the size search must be widened and validation is especially important.

The repository implements **TG-VTSS** (theory-guided, validation-tuned synthetic size). It diagnoses the local regime from real training data, evaluates generator quality with cross-fitting when needed, constructs candidate synthetic sample sizes, and selects among them with a balanced validation criterion.

## Repository structure

| Path | Description |
| --- | --- |
| `tg_vtss.py` | Reusable TG-VTSS diagnostics, candidate construction, and reporting utilities. |
| `TGVTSS.ipynb` | End-to-end synthetic-size experiments with Gaussian-mixture/SMOTE and non-Gaussian/jitter settings. |
| `Local-Symmetry.ipynb` | Simulation studies for locally symmetric data-generating processes. |
| `Asymmetry-Alignment-Realistic.ipynb` | Locally asymmetric experiment with a realistic/aligned generator. |
| `Asymmetry-Alignment-Biased.ipynb` | Locally asymmetric experiment with a deliberately biased generator. |
| `hypo-test.ipynb` | Null calibration, Q–Q, and power studies for the local-symmetry test. |
| `TG-nonsyn.ipynb` | Non-synthetic baselines and TG-VTSS comparisons using a PyTorch MLP. |
| `Real-Data/Data-Construction/` | SQL and Python scripts for retrospective MIMIC-III cohorts. |
| `Real-Data/Mortality/mortality.py` | Hospital-mortality experiment. |
| `Real-Data/Sepsis/sepsis.py` | Severe-sepsis experiment. |
| `Real-Data/Septicshock/septicshock.py` | Septic-shock experiment. |

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/Zechiiii/Synthetic-Data-Augmentation-Code.git
cd Synthetic-Data-Augmentation-Code

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib scikit-learn imbalanced-learn joblib jupyter torch duckdb
```

PyTorch is needed by `TG-nonsyn.ipynb`; DuckDB is needed only for real-data construction.

## Quick start

Start Jupyter from the repository root so notebooks can import `tg_vtss.py`:

```bash
jupyter lab
```

Open a notebook and run its cells in order. Experiment settings appear near the beginning of each notebook. For a smoke test, reduce the number of repetitions and test-set size before running a full paper-scale configuration.

## TG-VTSS API

```python
from tg_vtss import choose_tg_vtss_candidates

decision = choose_tg_vtss_candidates(
    X,
    y,
    estimator,
    generator,
    alpha=0.05,
    folds=5,
    seed=42,
)

print(decision.regime)
print(decision.generator_decision)
print(decision.candidate_sizes)
```

Input conventions:

- `X` and `y` contain real training observations with binary labels `0` and `1`.
- Class `0` is the majority class and class `1` the minority class.
- `estimator` supports `fit` and `predict_proba`, including sample weights directly or through the final pipeline step.
- `generator(X_real, y_real, n_synthetic, seed)` returns exactly `n_synthetic` minority feature vectors.
- At least two real observations per class are required.

The returned `TGVtssDecision` records the symmetry statistic and cutoff, generator quality score (GQS), gradient diagnostics, theory target, and candidate sizes. Use `print_tg_vtss_decision` for individual runs and `print_tg_vtss_summary` for repeated experiments.

Default candidate grids are:

- local symmetry: 0–20% of the class-balance gap;
- realistic generator: 80–120% neighborhoods around balance and theory targets;
- biased generator: 0–200% of the class-balance gap.

Candidate construction is the theory-guided stage; notebooks then tune the final size with their validation criterion.

## Simulation studies

A suggested order is:

1. `hypo-test.ipynb` — null calibration and power of the symmetry diagnostic.
2. `Local-Symmetry.ipynb` — the symmetric regime.
3. `Asymmetry-Alignment-Realistic.ipynb` and `Asymmetry-Alignment-Biased.ipynb` — aligned and misspecified generators under asymmetry.
4. `TGVTSS.ipynb` — end-to-end synthetic-size selection.
5. `TG-nonsyn.ipynb` — comparisons with raw ERM, class reweighting, and tuned cost-sensitive learning.

The checked-in notebooks include executed outputs. Full configurations can be computationally expensive because they use repeated experiments, large balanced test sets, cross-validation, and, in some cases, neural-network training.

## Real-data applications

The real-data code uses **MIMIC-III**, which is not redistributed here. Obtain authorized access separately and prepare `mimic3.db` with a `mimic3` schema containing the tables referenced by `hmm-data.sql`.

The retrospective admission-level cohort:

- retains each patient’s latest adult admission;
- summarizes available vitals and labs during the first 120 hours;
- studies same-admission hospital mortality, severe sepsis/septic shock diagnosis, and septic-shock diagnosis;
- is intended for retrospective identification, not post-120-hour landmark prediction, because ICD-9 diagnosis records do not define onset time.

### Construct datasets

Place `mimic3.db` in the working directory and run:

```bash
python Real-Data/Data-Construction/mortalitydataset.py
python Real-Data/Data-Construction/sepsisdataset.py
python Real-Data/Data-Construction/septicshockdataset.py
```

The scripts write `mortality.csv`, `sepsis.csv`, and `septicshock.csv` under `Real-Data/Data-Construction/`. Missing values are preserved; downstream pipelines fit imputation during modeling.

### Run outcome experiments

Copy each CSV into its corresponding experiment directory. Run each analysis from that directory because its input path is relative:

```bash
cp Real-Data/Data-Construction/mortality.csv Real-Data/Mortality/
cp Real-Data/Data-Construction/sepsis.csv Real-Data/Sepsis/
cp Real-Data/Data-Construction/septicshock.csv Real-Data/Septicshock/

(cd Real-Data/Mortality && python mortality.py)
(cd Real-Data/Sepsis && python sepsis.py)
(cd Real-Data/Septicshock && python septicshock.py)
```

On Windows, use `Copy-Item` and `Set-Location`. Each analysis writes serialized results, long-form metrics, and summary CSV files beside its script.

## Reproducibility

Seeds and principal settings are defined in the notebooks and scripts. Run notebook cells in order in a fresh kernel; preserve data splits; fit preprocessing and generators only on training data; and record dependency versions and hardware. Floating-point results may vary slightly across platforms, package versions, and parallel or GPU backends.

## Data and privacy

No MIMIC-III patient data are included. Users must comply with PhysioNet credentialing, the data-use agreement, institutional requirements, and applicable privacy rules. Do not commit databases or derived patient-level CSV files to a public repository.

## Citation

If you use this code, please cite:

> *Synthetic Augmentation in Imbalanced Learning: When It Helps, When It Hurts, and How Much to Add.*

A complete BibTeX entry can be added when the paper’s publication metadata is available.

