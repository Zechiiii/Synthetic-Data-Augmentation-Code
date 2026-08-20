# Synthetic Augmentation in Imbalanced Learning

Official simulation and real-data code for:

> **Synthetic Augmentation in Imbalanced Learning: When It Helps, When It Hurts, and How Much to Add**  
> Zhengchi Ma and Anru R. Zhang, Duke University

This repository accompanies the paper’s simulations, hypothesis-test experiments, and retrospective MIMIC-III applications. It also provides the reference implementation of **TG-VTSS**: Theory-Guided Validation-Tuned Synthetic Size.

## Research question and main findings

Synthetic oversampling can improve minority-class learning, but it is not automatically beneficial. The paper asks:

1. **When does synthetic augmentation help?**
2. **How many synthetic observations should be added?**

The analysis uses a balanced population risk, which weights both classes equally. The value of augmentation depends on the learning geometry, generator mismatch, directional alignment of that mismatch, and synthetic sample size.

### Local asymmetry

The classes have different first-order influence near the balanced-risk optimum. Imbalance can be a leading source of error, so augmentation may help.

- An ideal generator supports balancing.
- With a realistic, consistent generator, balancing is a useful default, but directional alignment may favor a nearby theory-guided size.
- With a biased generator, balancing may be inconsistent. Another size can sometimes cancel leading bias; in other geometries, changing quantity alone cannot remove mismatch.

### Local symmetry

The classes already have equal first-order influence in relevant directions near the balanced optimum. Imbalance is not the leading bottleneck. Realistic synthetic augmentation cannot improve the learning rate and generator mismatch can worsen performance, so little or no augmentation is preferred.

## TG-VTSS

**Theory-Guided Validation-Tuned Synthetic Size** operationalizes the theory:

1. Fit a balanced-risk pilot model on real training observations.
2. Test local symmetry using a Wald statistic.
3. Under local symmetry, consider only zero or small augmentation.
4. Under local asymmetry, estimate a cross-fitted generator-quality score (GQS).
5. For a realistic generator, search near the balancing and directionally adjusted theory targets.
6. For a potentially biased generator, use a wider range.
7. Choose the final size by minimizing balanced validation loss.

The diagnostics use real training data and cross-fitted synthetic observations. Validation observations are reserved for candidate selection.

Let (n_0-n_1) be the class-balance gap, with class 0 the majority:

| Regime | Default candidate sizes |
| --- | --- |
| Local symmetry | (mathrm{round}(gamma(n_0-n_1))), (gammain{0,.05,.10,.15,.20}) |
| Asymmetry, realistic generator | 80%-120% neighborhoods around balance and theory targets |
| Asymmetry, potentially biased generator | (mathrm{round}(gamma(n_0-n_1))), (gammain{0,.05,ldots,2}) |

The default GQS threshold is 10. The grids are configurable and can reflect validation noise and computational budget.

## Repository map

| Path | Paper role |
| --- | --- |
| `tg_vtss.py` | Symmetry diagnostic, cross-fitted GQS, candidate construction, and reporting. |
| `hypo-test.ipynb` | Null calibration and local-power studies for the symmetry test. |
| `Asymmetry-Alignment-Biased.ipynb` | Biased aligned generator; balancing versus a bias-canceling size. |
| `Asymmetry-Alignment-Realistic.ipynb` | Consistent aligned generator; TG-VTSS versus balancing. |
| `Local-Symmetry.ipynb` | Mean-shift and logistic local-symmetry experiments. |
| `TGVTSS.ipynb` | End-to-end linear/SMOTE and nonlinear/jitter evaluations. |
| `TG-nonsyn.ipynb` | Comparison with reweighting and cost-sensitive learning. |
| `Real-Data/Data-Construction/` | MIMIC-III SQL extraction and outcome-specific construction. |
| `Real-Data/{Mortality,Sepsis,Septicshock}/` | The three clinical analyses. |

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/Zechiiii/Synthetic-Data-Augmentation-Code.git
cd Synthetic-Data-Augmentation-Code
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib scikit-learn \
  imbalanced-learn joblib jupyter torch duckdb
```

PyTorch is used by `TG-nonsyn.ipynb`; DuckDB is needed only for clinical cohort construction. Versions are not pinned, so record the environment for exact replication.

## Quick start

Start Jupyter at the repository root so notebooks can import `tg_vtss.py`:

```bash
jupyter lab
```

Run cells in order in a fresh kernel. Paper configurations can be expensive: several use 100 repetitions, large balanced test sets, repeated cross-validation, parallel execution, or neural networks. Reduce repetitions and test sizes for a smoke test.

## TG-VTSS API

```python
from sklearn.linear_model import LogisticRegression
from tg_vtss import choose_tg_vtss_candidates, print_tg_vtss_decision

def generator(X_real, y_real, n_synthetic, seed):
    # Fit/use only the supplied real training observations.
    ...
    return X_synthetic  # exactly n_synthetic rows

decision = choose_tg_vtss_candidates(
    X, y,
    LogisticRegression(max_iter=2000),
    generator,
    alpha=0.05,
    tau_gqs=10.0,
    folds=5,
    seed=42,
)
print_tg_vtss_decision(1, decision, random_seed=42)
```

Requirements:

- `X` is a two-dimensional feature array and `y` contains labels 0 and 1.
- Class 0 is the majority and class 1 the minority.
- At least two real observations per class are required.
- The estimator implements `fit` and `predict_proba` and accepts sample weights directly or through its final pipeline step.
- `generator(X_real, y_real, n_synthetic, seed)` returns exactly the requested number of minority vectors.

`TGVtssDecision` records the regime, generator decision, symmetry statistic and cutoff, GQS, discrepancy norms and cosine, theory target, and candidate sets. Candidate construction does not fit the final model: the notebooks validate the candidates, generate the selected amount, and refit.

## Simulation studies

### Local-symmetry test — `hypo-test.ipynb`

- Centered squared-loss model in (d=5).
- Exact local-symmetry null.
- Imbalance settings include (n_0/n_1=20).
- (n_1in{50,250,500}).
- 5,000 datasets per null-calibration configuration.
- Local power at imbalance ratios 5 and 20.
- Empirical rejection probabilities compared with noncentral chi-square theory.

Set `PAPER_MODE = True` for the full design.

### Biased aligned generator — `Asymmetry-Alignment-Biased.ipynb`

A two-dimensional Gaussian squared-loss experiment:

- majority (N(0,I_2));
- true minority (N((1,0),I_2));
- biased aligned synthetic minority (N((0.5,0),I_2));
- imbalance ratio 20;
- minority sizes 6 through 3,200;
- 5,000 fresh test observations per class;
- 100 Monte Carlo repetitions.

It compares balancing, (widetilde n=n_0-n_1), with the bias-canceling choice (widetilde n=4(n_0-n_1)). The latter drives parameter error and balanced excess risk toward zero; balancing approaches a nonzero floor.

### Realistic aligned generator — `Asymmetry-Alignment-Realistic.ipynb`

The synthetic mean converges to the true minority mean while remaining aligned.

- (n_1in{10,100,200,300,400,500}), imbalance ratio 20;
- 100 repetitions;
- three-fold GQS cross-fitting;
- five-fold stratified validation;
- no-intercept ridge least squares with ridge (10^{-6}).

TG-VTSS is compared with balancing using population excess risk and parameter error.

### Local symmetry — `Local-Symmetry.ipynb`

The main mean-shift study uses (d=20), (n_1=100), (n_0=2{,}000), 20 multipliers over ([0,4]), and 100 repetitions. Noise families are uniform cube, Rademacher, and uniform sphere. Generators are Oracle, SMOTE (five neighbors), Gaussian-fit, and Semi-Oracle.

Realistic imperfect generators generally do not improve balanced excess risk and often worsen it as augmentation grows; the perfect Oracle can improve monotonically. The notebook also includes a sigmoid-Bernoulli logistic example.

### End-to-end TG-VTSS — `TGVTSS.ipynb`

**Linear configuration:** (d=5), (n_1=200), (n_0=4{,}000); Gaussian majority, two-component Gaussian-mixture minority; SMOTE plus logistic regression; multipliers 0-2 by 0.05; balanced test set of 10,000 per class.

**Nonlinear configuration:** (d=5), the same class counts; non-Gaussian majority mixture and noisy-ring minority; bootstrap plus Gaussian jitter; logistic regression on 100 random Fourier RBF features; five-fold diagnostics and validation; balanced test set of 10,000 per class.

The paper configuration uses 100 repetitions and compares the TG-VTSS-selected loss with the complete fixed-size loss curve.

### Synthetic versus non-synthetic — `TG-nonsyn.ipynb`

A (d=10) nonlinear problem with 800 majority and 100 minority observations compares:

- raw ERM;
- balanced reweighting;
- tuned cost-sensitive learning;
- TG-VTSS with SMOTE, local jitter, or ADASYN.

All use the same standardized neural network with ReLU layers of widths 64 and 32, trained using full-batch AdamW. The paper repeats the experiment 100 times. This is an existence demonstration, not a universal superiority claim: outcomes vary substantially by generator.

## MIMIC-III real-data applications

### Access and interpretation

MIMIC-III v1.4 is not distributed here. Access requires PhysioNet credentialing, training, and a data-use agreement: <https://physionet.org/content/mimiciii/1.4/>.

Prepare `mimic3.db` with a `mimic3` schema containing the tables referenced by `hmm-data.sql`. The cohort retains each patient’s latest adult admission (age greater than 15) to avoid patient overlap.

| Retrospective outcome | Positive | Negative | Ratio |
| --- | ---: | ---: | ---: |
| In-hospital mortality | 5,744 | 32,893 | 5.726 |
| Severe sepsis or septic shock | 3,197 | 35,440 | 11.09 |
| Septic shock | 2,028 | 36,609 | 18.05 |

Mortality is any in-hospital death in the admission. Sepsis uses ICD-9 99592 or 78552; septic shock uses 78552. The first 120 hours define the covariate window, not a prediction landmark. Since diagnosis onset times are unavailable, sepsis tasks are retrospective admission-level identification.

### Forty clinical features

Four vital signs are averaged into 20 six-hour bins:

- heart rate, respiratory rate, temperature, systolic blood pressure.

Six labs are averaged into five 24-hour bins:

- white blood cells, platelets, creatinine, INR, lactate, total bilirubin.

Each variable is summarized by mean, standard deviation, last value, and hourly linear trend, yielding 40 covariates. Trends with fewer than two observed bins remain missing.

### Build datasets

The scripts open `mimic3.db` using a relative path. Run from the directory containing it or adjust that path.

```bash
python Real-Data/Data-Construction/mortalitydataset.py
python Real-Data/Data-Construction/sepsisdataset.py
python Real-Data/Data-Construction/septicshockdataset.py
```

They create `mortality.csv`, `sepsis.csv`, and `septicshock.csv` under `Real-Data/Data-Construction/`. Identifiers are retained for auditing but excluded from modeling; missing values remain unfilled.

### Run analyses

Copy each CSV to its analysis directory, then run there because input paths are relative:

```bash
cp Real-Data/Data-Construction/mortality.csv Real-Data/Mortality/
cp Real-Data/Data-Construction/sepsis.csv Real-Data/Sepsis/
cp Real-Data/Data-Construction/septicshock.csv Real-Data/Septicshock/
(cd Real-Data/Mortality && python mortality.py)
(cd Real-Data/Sepsis && python sepsis.py)
(cd Real-Data/Septicshock && python septicshock.py)
```

PowerShell users can use `Copy-Item` and `Push-Location`/`Pop-Location`.

Each logistic-regression analysis compares SMOTE, ADASYN, Borderline-SMOTE, and minority jitter. The paper uses 100 train-test splits, balanced excess risk, repeated five-fold validation with five repeats, balanced log loss for size selection, and empirical 95% intervals.

### Leakage-safe preprocessing

For every outer split and every model-selection fold:

1. Fit mean imputation only on real training observations.
2. Fit standardization only on those training observations.
3. Apply the fitted transformations unchanged to validation/test data.
4. Fit the synthetic generator only on the transformed training fold.

Each analysis writes a pickle, long-form metric CSV, and summary CSV beside its script.

## Reproducibility

- Run notebooks at repository root, in order, with a fresh kernel.
- Preserve configured seeds and restore full repetition counts for paper results.
- Keep train, validation, and test data separate.
- Fit preprocessing and generators only on the relevant training split.
- Record package versions, hardware, and parallel backend.
- Expect small floating-point differences across environments.

Checked-in notebook outputs and figures provide reference results.

## Privacy

Do not commit `mimic3.db`, derived patient-level CSVs, or protected data. Users must comply with PhysioNet requirements, institutional approval, the MIMIC-III agreement, and applicable privacy rules.

## Citation

```text
Ma, Zhengchi, and Anru R. Zhang.
“Synthetic Augmentation in Imbalanced Learning:
When It Helps, When It Hurts, and How Much to Add.”
```

Add a BibTeX entry when publication or preprint metadata are available.

## Contact

The manuscript lists Anru R. Zhang as corresponding author: `anru.zhang@duke.edu`.

