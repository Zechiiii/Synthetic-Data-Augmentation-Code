"""Repeated-CV theory-guided VTSS analysis for retrospective septic shock."""

import os
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    jaccard_score, matthews_corrcoef, precision_score, recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_ROOT = SCRIPT_DIR.parents[2]
TG_MODULE_DIR = REVISION_ROOT
if str(TG_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(TG_MODULE_DIR))

from tg_vtss import (  # noqa: E402
    choose_tg_vtss_candidates,
    print_tg_vtss_summary,
)

DATA_PATH = "septicshock.csv"
TARGET = "septic_shock"
ID_COLUMNS = ["subject_id", "hadm_id"]
RANDOM_STATE = 42
K_FOLDS = 5
N_REPEATS = 5
N_SPLITS = 100
N_JOBS = 6
TAU_GQS = 10.0
GQS_FOLDS = 3
SWEEP_GAMMAS = np.arange(0.0, 2.0001, 0.05)
GENERATOR_NAMES = [
    "SMOTE", "BorderlineSMOTE", "ADASYN", "Minority jitter"
]


def make_imputer():
    try:
        return SimpleImputer(strategy="mean", keep_empty_features=True)
    except TypeError:  # compatibility with older scikit-learn
        return SimpleImputer(strategy="mean")


def make_logreg(seed=RANDOM_STATE, warm_start=False):
    # lbfgs is substantially faster than liblinear for this dense 40-feature dataset.
    return LogisticRegression(
        solver="lbfgs", max_iter=300, tol=1e-3,
        warm_start=bool(warm_start), random_state=int(seed),
    )


def diagnostic_estimator(seed):
    return Pipeline([
        ("imputer", make_imputer()),
        ("scaler", StandardScaler()),
        ("clf", make_logreg(seed)),
    ])


def balanced_log_loss(y_true, p_pos, eps=1e-15):
    y_true = np.asarray(y_true, dtype=int)
    p_pos = np.clip(np.asarray(p_pos, dtype=float), eps, 1.0 - eps)
    losses = np.where(y_true == 1, -np.log(p_pos), -np.log1p(-p_pos))
    return float(0.5 * losses[y_true == 0].mean() + 0.5 * losses[y_true == 1].mean())


def youden_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return sensitivity + specificity - 1.0


def compute_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "BalancedLogLoss": balanced_log_loss(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions),
        "Jaccard": jaccard_score(y_true, predictions, zero_division=0),
        "Youden": youden_score(y_true, predictions),
        "BalancedAccuracy": balanced_accuracy_score(y_true, predictions),
    }


def balanced_test_subset(X, y, seed):
    y = np.asarray(y, dtype=int)
    index0, index1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    n = min(len(index0), len(index1))
    rng = np.random.default_rng(seed)
    keep = np.r_[rng.choice(index0, n, replace=False), rng.choice(index1, n, replace=False)]
    rng.shuffle(keep)
    return X[keep], y[keep]


def safe_neighbors(y, default=10):
    return min(int(default), max(1, int(np.sum(np.asarray(y) == 1)) - 1))


def _adaptive_sampler_points(sampler_class, Z, y, n_synthetic, seed, **kwargs):
    n1 = int(np.sum(y == 1))
    requested = int(n_synthetic)
    generated = np.empty((0, Z.shape[1]))
    for attempt in range(20):
        sampler = sampler_class(
            sampling_strategy={1: n1 + requested},
            random_state=int(seed) + attempt,
            **kwargs,
        )
        try:
            Z_resampled, _ = sampler.fit_resample(Z, y)
        except ValueError as exc:
            if "No samples will be generated" not in str(exc):
                raise
            requested = max(2 * requested, int(np.ceil(n1 / 2)))
            continue
        generated = Z_resampled[len(Z):]
        if len(generated) >= n_synthetic:
            return generated[:n_synthetic]
        requested = max(
            requested + (n_synthetic - len(generated)) + 2,
            int(np.ceil(1.35 * requested)),
        )
    raise RuntimeError(
        f"{sampler_class.__name__} produced {len(generated)} of {n_synthetic} requested points."
    )


def fit_generator_pool(generator_name, Z, y, n_synthetic, seed):
    """Fit one generator at maximum size and return a reusable synthetic pool."""
    n_synthetic = int(n_synthetic)
    if n_synthetic <= 0:
        return np.empty((0, Z.shape[1]))
    y = np.asarray(y, dtype=int)
    minority = Z[y == 1]
    k = safe_neighbors(y)

    if generator_name == "SMOTE":
        sampler = SMOTE(
            sampling_strategy={1: len(minority) + n_synthetic},
            k_neighbors=k, random_state=int(seed),
        )
        Z_resampled, _ = sampler.fit_resample(Z, y)
        return Z_resampled[len(Z):len(Z) + n_synthetic]
    if generator_name == "BorderlineSMOTE":
        return _adaptive_sampler_points(
            BorderlineSMOTE, Z, y, n_synthetic, seed, k_neighbors=k
        )
    if generator_name == "ADASYN":
        return _adaptive_sampler_points(ADASYN, Z, y, n_synthetic, seed, n_neighbors=k)

    rng = np.random.default_rng(seed)
    if generator_name == "Minority jitter":
        base = minority[rng.integers(0, len(minority), size=n_synthetic)]
        scale = np.nan_to_num(np.std(minority, axis=0, ddof=1), nan=0.0)
        return base + rng.normal(size=base.shape) * scale
    raise KeyError(f"Unknown generator: {generator_name}")


def generate_raw(generator_name, X_real, y_real, n_synthetic, seed):
    """Fit preprocessing only on supplied real data and return exact synthetic rows."""
    X_real = np.asarray(X_real, dtype=float)
    imputer = make_imputer().fit(X_real)
    X_imputed = imputer.transform(X_real)
    scaler = StandardScaler().fit(X_imputed)
    Z = scaler.transform(X_imputed)
    Z_syn = fit_generator_pool(generator_name, Z, y_real, n_synthetic, seed)
    return scaler.inverse_transform(Z_syn)



def repeated_cv_select_size(X, y, candidates, generator_name, splits, full_gap, seed):
    """Average repeated-CV loss using one nested synthetic pool per fold."""
    candidates = np.asarray(sorted(set(int(n) for n in candidates)), dtype=int)
    losses = np.full((len(candidates), len(splits)), np.nan)
    for fold, (train_index, validation_index) in enumerate(splits):
        X_train, y_train = X[train_index], y[train_index]
        X_validation, y_validation = X[validation_index], y[validation_index]

        # Fit preprocessing once, using only this fold's real training rows.
        imputer = make_imputer().fit(X_train)
        X_train_imputed = imputer.transform(X_train)
        scaler = StandardScaler().fit(X_train_imputed)
        Z_train = scaler.transform(X_train_imputed)
        Z_validation = scaler.transform(imputer.transform(X_validation))

        fold_gap = int(np.sum(y_train == 0) - np.sum(y_train == 1))
        fold_sizes = np.asarray([
            max(0, int(round(size * fold_gap / full_gap))) if full_gap > 0 else 0
            for size in candidates
        ])
        max_size = int(np.max(fold_sizes))
        pool = fit_generator_pool(
            generator_name, Z_train, y_train, max_size, seed + 1009 * fold
        ) if max_size > 0 else np.empty((0, Z_train.shape[1]))

        # Candidates are nested prefixes of one seeded pool. Warm starts avoid
        # resolving nearly identical logistic problems from scratch.
        classifier = make_logreg(seed + 1009 * fold, warm_start=True)
        for candidate_index, fold_size in enumerate(fold_sizes):
            Z_fit = np.vstack([Z_train, pool[:fold_size]]) if fold_size else Z_train
            y_fit = np.r_[y_train, np.ones(fold_size, dtype=int)] if fold_size else y_train
            classifier.fit(Z_fit, y_fit)
            probabilities = classifier.predict_proba(Z_validation)[:, 1]
            losses[candidate_index, fold] = balanced_log_loss(y_validation, probabilities)

    mean_losses = np.nanmean(losses, axis=1)
    best = int(np.nanargmin(mean_losses))
    return int(candidates[best]), float(mean_losses[best]), candidates, mean_losses


def run_one_split(rep, X, y, split_seed):
    warnings.filterwarnings("ignore")
    train_index, test_index = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=split_seed
    )
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = balanced_test_subset(X[test_index], y[test_index], split_seed)
    n0, n1 = int(np.sum(y_train == 0)), int(np.sum(y_train == 1))
    gap = n0 - n1
    repeated_splits = list(RepeatedStratifiedKFold(
        n_splits=K_FOLDS, n_repeats=N_REPEATS, random_state=split_seed
    ).split(X_train, y_train))

    # Reuse outer-training preprocessing for every final model and sweep point.
    outer_imputer = make_imputer().fit(X_train)
    X_train_imputed = outer_imputer.transform(X_train)
    outer_scaler = StandardScaler().fit(X_train_imputed)
    Z_train = outer_scaler.transform(X_train_imputed)
    Z_test = outer_scaler.transform(outer_imputer.transform(X_test))
    raw_classifier = make_logreg(split_seed).fit(Z_train, y_train)
    raw_metrics = compute_metrics(y_test, raw_classifier.predict_proba(Z_test)[:, 1])
    results = [("Raw LR", raw_metrics)]
    decisions, selected_sizes, selected_cv_losses = {}, {}, {}
    sweeps = {}

    for generator_index, generator_name in enumerate(GENERATOR_NAMES):
        generator_seed = split_seed + 100_000 * (generator_index + 1)
        def diagnostic_generator(Xr, yr, _requested, diagnostic_seed, name=generator_name):
            # Balance the current GQS fold-training subset, as required by cross-fitting.
            fold_balance = max(0, int(np.sum(np.asarray(yr) == 0) - np.sum(np.asarray(yr) == 1)))
            return generate_raw(name, Xr, yr, fold_balance, diagnostic_seed)
        decision = choose_tg_vtss_candidates(
            X_train, y_train, diagnostic_estimator(generator_seed), diagnostic_generator,
            tau_gqs=TAU_GQS, folds=GQS_FOLDS, seed=generator_seed,
        )
        selected_size, cv_loss, _, _ = repeated_cv_select_size(
            X_train, y_train, decision.candidate_sizes, generator_name,
            repeated_splits, gap, generator_seed + 10_000,
        )
        decisions[generator_name] = decision
        selected_sizes[generator_name] = selected_size
        selected_cv_losses[generator_name] = cv_loss

        # One maximum seeded pool supplies nested prefixes for the balanced fit,
        # TG-VTSS fit, and entire 0--2 plotting sweep. Expensive generators such
        # the generator is therefore fitted once rather than once per synthetic size.
        sweep_sizes = np.asarray([max(0, int(round(gamma * gap))) for gamma in SWEEP_GAMMAS])
        max_pool_size = max(int(np.max(sweep_sizes)), gap, selected_size)
        pool = fit_generator_pool(
            generator_name, Z_train, y_train, max_pool_size, generator_seed + 20_000
        )

        def probabilities_for_size(size, classifier_seed):
            size = int(size)
            Z_fit = np.vstack([Z_train, pool[:size]]) if size else Z_train
            y_fit = np.r_[y_train, np.ones(size, dtype=int)] if size else y_train
            classifier = make_logreg(classifier_seed).fit(Z_fit, y_fit)
            return classifier.predict_proba(Z_test)[:, 1]

        balanced_probabilities = probabilities_for_size(gap, generator_seed + 30_000)
        tg_probabilities = probabilities_for_size(selected_size, generator_seed + 40_000)
        results.append((f"{generator_name}-balanced LR", compute_metrics(
            y_test, balanced_probabilities
        )))
        results.append((f"{generator_name} + TG-VTSS", compute_metrics(
            y_test, tg_probabilities
        )))

        sweep = np.empty(len(SWEEP_GAMMAS))
        sweep_classifier = make_logreg(generator_seed + 50_000, warm_start=True)
        for gamma_index, size in enumerate(sweep_sizes):
            Z_fit = np.vstack([Z_train, pool[:size]]) if size else Z_train
            y_fit = np.r_[y_train, np.ones(size, dtype=int)] if size else y_train
            sweep_classifier.fit(Z_fit, y_fit)
            sweep[gamma_index] = balanced_log_loss(
                y_test, sweep_classifier.predict_proba(Z_test)[:, 1]
            )
        sweeps[generator_name] = sweep

    # Estimate the unknown optimal balanced risk by an empirical ERM fitted and
    # evaluated on the same balanced test sample. This deliberately overfits the
    # test sample and is used only as the subtracted reference-risk term; it is
    # never used for TG-VTSS diagnosis, size tuning, or method prediction.
    oracle_classifier = LogisticRegression(
        solver="lbfgs", C=1e6, max_iter=1000, tol=1e-6,
        random_state=split_seed + 900_000,
    ).fit(Z_test, y_test)
    oracle_probabilities = oracle_classifier.predict_proba(Z_test)[:, 1]
    oracle_loss = balanced_log_loss(y_test, oracle_probabilities)

    for _, metrics in results:
        metrics["EmpiricalBalancedExcessRisk"] = (
            metrics["BalancedLogLoss"] - oracle_loss
        )
    print(f"[outer split {rep + 1}/{N_SPLITS}] completed", flush=True)
    return {
        "rep": rep, "results": results, "decisions": decisions,
        "selected_sizes": selected_sizes, "selected_cv_losses": selected_cv_losses,
        "gap": gap, "sweeps": sweeps, "oracle_loss": oracle_loss,
    }

def summarize_metrics(outputs):
    rows = []
    for output in outputs:
        for method, metrics in output["results"]:
            rows.append({"rep": output["rep"], "method": method, **metrics})
    long = pd.DataFrame(rows)
    summary_rows = []
    metric_columns = [column for column in long.columns if column not in {"rep", "method"}]
    for method, part in long.groupby("method", sort=False):
        for metric in metric_columns:
            values = part[metric].to_numpy(float)
            mean = float(np.mean(values))
            se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            summary_rows.append({
                "method": method, "metric": metric, "mean": mean,
                "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
            })
    return long, pd.DataFrame(summary_rows)


def main():
    warnings.filterwarnings("ignore")
    total_start = time.perf_counter()
    print(
        f"Starting septic-shock TG-VTSS: {N_SPLITS} outer train/test splits",
        flush=True,
    )
    df = pd.read_csv(DATA_PATH)
    required = set(ID_COLUMNS + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in septicshock.csv: {missing}")
    if df["subject_id"].duplicated().any():
        raise ValueError("septicshock.csv must contain only one admission per subject_id.")

    feature_columns = [column for column in df.columns if column not in required]
    if len(feature_columns) != 40:
        raise ValueError(
            f"Expected 40 longitudinal summary covariates, found {len(feature_columns)}."
        )
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    if np.any(np.all(np.isnan(X), axis=0)):
        empty = [feature_columns[i] for i in np.flatnonzero(np.all(np.isnan(X), axis=0))]
        raise ValueError(f"Entirely missing covariates cannot be mean-imputed: {empty}")

    rng = np.random.default_rng(RANDOM_STATE)
    split_seeds = rng.integers(0, 10**9, size=N_SPLITS)
    outputs = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(run_one_split)(rep, X, y, int(split_seeds[rep]))
        for rep in range(N_SPLITS)
    )
    outputs = sorted(outputs, key=lambda item: item["rep"])

    with open(SCRIPT_DIR / "septicshock_tg_vtss_outputs.pkl", "wb") as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    long, summary = summarize_metrics(outputs)
    long.to_csv(SCRIPT_DIR / "septicshock_tg_vtss_metrics.csv", index=False)
    summary.to_csv(SCRIPT_DIR / "septicshock_tg_vtss_summary.csv", index=False)

    for generator_name in GENERATOR_NAMES:
        generator_decisions = [
            output["decisions"][generator_name] for output in outputs
        ]
        print(f"\nGenerator: {generator_name}")
        print_tg_vtss_summary(generator_decisions)

    for generator_name in GENERATOR_NAMES:
        curves = np.column_stack([output["sweeps"][generator_name] for output in outputs])
        oracle_losses = np.asarray([output["oracle_loss"] for output in outputs])
        excess_curves = curves - oracle_losses[None, :]
        selected_gamma = np.asarray([
            output["selected_sizes"][generator_name] / output["gap"]
            if output["gap"] > 0 else 0.0 for output in outputs
        ])
        selected_excess = np.asarray([
            next(metrics["EmpiricalBalancedExcessRisk"] for method, metrics in output["results"]
                 if method == f"{generator_name} + TG-VTSS")
            for output in outputs
        ])
        mean_curve = np.nanmean(excess_curves, axis=1)
        curve_q025, curve_q975 = np.nanquantile(
            excess_curves, [0.025, 0.975], axis=1
        )
        tg_mean = float(np.nanmean(selected_excess))
        tg_q025, tg_q975 = np.nanquantile(selected_excess, [0.025, 0.975])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(SWEEP_GAMMAS, mean_curve, label=f"{generator_name} excess-risk curve")
        ax.fill_between(
            SWEEP_GAMMAS, curve_q025, curve_q975,
            alpha=0.20, label="Curve empirical 95% interval",
        )
        ax.axhspan(
            tg_q025, tg_q975, color="black", alpha=0.12,
            label="TG-VTSS empirical 95% interval",
        )
        ax.axhline(tg_mean, linestyle="--", color="black", label="TG-VTSS mean")
        ax.axvline(np.mean(selected_gamma), linestyle=":", color="tab:red", label="Mean selected size")
        ax.set_xlim(0.0, 2.0)
        ax.set_xlabel(r"Synthetic-size ratio $\tilde n/(n_0-n_1)$", fontsize=14)
        ax.set_ylabel("Empirical balanced excess risk", fontsize=14)
        ax.set_title(f"Septic shock: {generator_name}", fontsize=16)
        ax.legend(fontsize=10)
        fig.tight_layout()
        slug = generator_name.lower().replace(" ", "_")
        fig.savefig(SCRIPT_DIR / f"septicshock_{slug}_tg_vtss.pdf", bbox_inches="tight")
        fig.savefig(SCRIPT_DIR / f"septicshock_{slug}_tg_vtss.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(
        f"Finished all work in {time.perf_counter()-total_start:.1f}s. "
        f"Saved tables, serialized output, and figures to {SCRIPT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()












