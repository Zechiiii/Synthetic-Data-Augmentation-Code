"""Theory-guided validation-tuned synthetic-size (TG-VTSS) utilities.

The implementation follows Algorithm 1.  Diagnostics
are computed only from real training observations and cross-fitted synthetic
observations; the returned candidate sizes are subsequently tuned using the
caller's balanced validation criterion.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


TAU_GQS = 10.0
GQS_FOLDS = 3
GAMMA_SYMMETRIC = np.arange(0.0, 0.2001, 0.05)
GAMMA_LOCAL = np.arange(0.80, 1.2001, 0.05)
GAMMA_WIDE = np.arange(0.0, 2.0001, 0.05)


@dataclass(frozen=True)
class TGVtssDecision:
    regime: str
    generator_decision: str
    t_sym: float
    symmetry_cutoff: float
    gqs: float
    g_phi_norm: float
    g_psi_norm: float
    cosine: float
    theory_denominator: float
    theory_target: float
    balance_candidates: tuple
    theory_candidates: tuple
    candidate_sizes: tuple


def _fit_balanced(estimator, X, y):
    fitted = clone(estimator)
    n0 = max(1, int(np.sum(y == 0)))
    n1 = max(1, int(np.sum(y == 1)))
    weights = np.where(y == 0, 0.5 / n0, 0.5 / n1)
    # Rescale to mean one; this is immaterial for an unpenalized ERM and is
    # numerically friendlier for sklearn estimators.
    weights *= len(y)
    if hasattr(fitted, "steps"):
        final_step = fitted.steps[-1][0]
        fitted.fit(X, y, **{f"{final_step}__sample_weight": weights})
    else:
        fitted.fit(X, y, sample_weight=weights)
    return fitted


def _scores_and_hessians(model, X, y):
    
    p = np.clip(model.predict_proba(X)[:, 1], 1e-10, 1 - 1e-10)
    if hasattr(model, "steps"):
        features = model[:-1].transform(X)
        final_estimator = model.steps[-1][1]
        if hasattr(final_estimator, "coef_"):
            Z = np.column_stack([features, np.ones(len(features))])
        else:
            Z = np.ones((len(X), 1))
    elif hasattr(model, "coef_") and np.asarray(model.coef_).ndim == 2:
        Z = np.column_stack([X, np.ones(len(X))])
    else:
        Z = np.ones((len(X), 1))
    scores = (p - y)[:, None] * Z
    hessians = p[:, None, None] * (1 - p)[:, None, None]
    hessians = hessians * Z[:, :, None] * Z[:, None, :]
    return scores, hessians


def _cov(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.zeros((x.shape[1], x.shape[1]))
    return np.cov(x, rowvar=False, ddof=1).reshape(x.shape[1], x.shape[1])


def _stable_inverse(matrix, ridge=1e-8):
    matrix = np.asarray(matrix, dtype=float)
    scale = max(1.0, float(np.trace(matrix)) / max(1, matrix.shape[0]))
    return np.linalg.pinv(matrix + ridge * scale * np.eye(matrix.shape[0]))


def _symmetry_diagnostic(estimator, X, y, alpha):
    pilot = _fit_balanced(estimator, X, y)
    s0, h0_i = _scores_and_hessians(pilot, X[y == 0], y[y == 0])
    s1, h1_i = _scores_and_hessians(pilot, X[y == 1], y[y == 1])
    m0, m1 = s0.mean(0), s1.mean(0)
    H0, H1 = h0_i.mean(0), h1_i.mean(0)
    H = 0.5 * (H0 + H1)
    G = H0 - H1
    H_inv = _stable_inverse(H)
    p = H.shape[0]
    A0 = np.eye(p) - 0.5 * G @ H_inv
    A1 = -np.eye(p) - 0.5 * G @ H_inv
    V = A0 @ _cov(s0) @ A0.T / len(s0) + A1 @ _cov(s1) @ A1.T / len(s1)
    g_phi = m0 - m1
    t_sym = float(g_phi @ _stable_inverse(V) @ g_phi)
    cutoff = float(chi2.ppf(1 - alpha, df=p))
    return pilot, g_phi, H, t_sym, cutoff


def _cross_fitted_gqs(estimator, X, y, generator, n_balance, folds, seed):
    n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
    splitter = StratifiedKFold(
        n_splits=min(folds, n0, n1), shuffle=True, random_state=seed
    )
    deltas, hessians, cov0, cov1, cov_syn = [], [], [], [], []
    for fold, (train, heldout) in enumerate(splitter.split(X, y)):
        X_train, y_train = X[train], y[train]
        X_hold, y_hold = X[heldout], y[heldout]
        pilot = _fit_balanced(estimator, X_train, y_train)
        X_syn = generator(X_train, y_train, n_balance, seed + 1009 * (fold + 1))
        if len(X_syn) == 0:
            raise ValueError("The generator returned no samples for the GQS diagnostic.")

        s0, h0 = _scores_and_hessians(pilot, X_hold[y_hold == 0], y_hold[y_hold == 0])
        s1, h1 = _scores_and_hessians(pilot, X_hold[y_hold == 1], y_hold[y_hold == 1])
        ss, _ = _scores_and_hessians(pilot, X_syn, np.ones(len(X_syn), dtype=int))
        deltas.append(ss.mean(0) - s1.mean(0))
        hessians.append(0.5 * (h0.mean(0) + h1.mean(0)))
        cov0.append(_cov(s0))
        cov1.append(_cov(s1))
        cov_syn.append(_cov(ss))

    delta = np.mean(deltas, axis=0)
    H = np.mean(hessians, axis=0)
    sigma_aug = (
        0.5 * np.mean(cov0, axis=0)
        + (n1 / (2 * n0)) * np.mean(cov1, axis=0)
        + ((n0 - n1) / (2 * n0)) * np.mean(cov_syn, axis=0)
    )
    H_inv = _stable_inverse(H)
    numerator = (n_balance ** 2 / (2 * n0)) * float(delta @ H_inv @ delta)
    denominator = float(np.trace(H_inv @ sigma_aug))
    gqs = numerator / max(denominator, np.finfo(float).eps)
    return float(gqs), delta


def _rounded_sizes(gammas, target):
    return {max(0, int(round(float(gamma) * float(target)))) for gamma in gammas}


def choose_tg_vtss_candidates(
    X,
    y,
    estimator,
    generator,
    *,
    alpha=0.05,
    tau_gqs=TAU_GQS,
    folds=GQS_FOLDS,
    seed=0,
):
    """Select the theory-guided candidate synthetic sizes for one run.

    ``generator(X_real, y_real, n_synthetic, seed)`` must fit/use the
    generator on the supplied real data and return exactly the requested
    minority synthetic observations.
    """
    X, y = np.asarray(X), np.asarray(y, dtype=int)
    n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
    n_balance = max(0, n0 - n1)
    if n0 < 2 or n1 < 2:
        raise ValueError("TG-VTSS requires at least two real observations per class.")

    _, g_phi, _, t_sym, cutoff = _symmetry_diagnostic(estimator, X, y, alpha)
    if t_sym <= cutoff:
        candidates = _rounded_sizes(GAMMA_SYMMETRIC, n_balance)
        return TGVtssDecision(
            "local symmetry", "not evaluated", t_sym, cutoff, np.nan,
            float(np.linalg.norm(g_phi)), np.nan, np.nan, np.nan, np.nan,
            tuple(sorted(candidates)), tuple(), tuple(sorted(candidates)),
        )

    gqs, g_psi = _cross_fitted_gqs(
        estimator, X, y, generator, n_balance, folds, seed
    )
    if gqs <= tau_gqs:
        g_phi_norm = float(np.linalg.norm(g_phi))
        g_psi_norm = float(np.linalg.norm(g_psi))
        cosine = float(g_phi @ g_psi) / max(
            g_phi_norm * g_psi_norm, np.finfo(float).eps
        )
        cosine = float(np.clip(cosine, -1.0, 1.0))
        denominator = 1.0 - 2.0 * (g_psi_norm / max(
            g_phi_norm, np.finfo(float).eps
        )) * cosine
        theory_target = n_balance / denominator if denominator > 0 else np.nan
        balance_candidates = _rounded_sizes(GAMMA_LOCAL, n_balance)
        theory_candidates = set()
        if np.isfinite(theory_target) and theory_target >= 0:
            theory_candidates = _rounded_sizes(GAMMA_LOCAL, theory_target)
        candidates = balance_candidates | theory_candidates
        decision = "realistic generator"
    else:
        g_phi_norm = float(np.linalg.norm(g_phi))
        g_psi_norm = float(np.linalg.norm(g_psi))
        cosine = float(g_phi @ g_psi) / max(
            g_phi_norm * g_psi_norm, np.finfo(float).eps
        )
        cosine = float(np.clip(cosine, -1.0, 1.0))
        denominator = np.nan
        theory_target = np.nan
        candidates = _rounded_sizes(GAMMA_WIDE, n_balance)
        balance_candidates = candidates
        theory_candidates = set()
        decision = "biased generator"

    return TGVtssDecision(
        "local asymmetry", decision, t_sym, cutoff, gqs,
        g_phi_norm, g_psi_norm, cosine, denominator, theory_target,
        tuple(sorted(balance_candidates)), tuple(sorted(theory_candidates)),
        tuple(sorted(candidates)),
    )


def print_tg_vtss_decision(
    run, decision, tau_gqs=TAU_GQS, random_seed=None
):
    gqs = "not evaluated" if np.isnan(decision.gqs) else f"{decision.gqs:.6g}"
    target = (
        "not used"
        if np.isnan(decision.theory_target)
        else str(int(round(decision.theory_target)))
    )
    theory = (
        "not used"
        if np.isnan(decision.cosine)
        else (
            f"||g_phi||={decision.g_phi_norm:.6g}, "
            f"||g_psi||={decision.g_psi_norm:.6g}, "
            f"cos={decision.cosine:.6g}, "
            f"denominator={decision.theory_denominator:.6g}"
        )
    )
    seed_text = "" if random_seed is None else f" | random_seed={random_seed}"
    print(
        f"run={run}{seed_text} | regime={decision.regime} | "
        f"generator={decision.generator_decision} | "
        f"T_sym={decision.t_sym:.6g} (cutoff={decision.symmetry_cutoff:.6g}) | "
        f"GQS={gqs} (tau_GQS={tau_gqs:g}) | {theory} | "
        f"theory_target={target} | "
        f"around_balance={list(decision.balance_candidates)} | "
        f"around_theory={list(decision.theory_candidates)} | "
        f"candidates={list(decision.candidate_sizes)}"
    )


def print_tg_vtss_summary(decisions):
    """Print regime/generator percentages using all repetitions as denominator."""
    decisions = list(decisions)
    total = len(decisions)
    if total == 0:
        print("TG-VTSS summary: no completed repetitions.")
        return

    def count(attribute, value):
        return sum(getattr(decision, attribute) == value for decision in decisions)

    n_sym = count("regime", "local symmetry")
    n_asym = count("regime", "local asymmetry")
    n_real = count("generator_decision", "realistic generator")
    n_bias = count("generator_decision", "biased generator")
    n_not_evaluated = count("generator_decision", "not evaluated")

    def percentage(n):
        return 100.0 * n / total

    print(f"\nTG-VTSS final diagnosis summary ({total} repetitions):")
    print(f"  Local symmetry:       {percentage(n_sym):6.2f}% ({n_sym}/{total})")
    print(f"  Local asymmetry:      {percentage(n_asym):6.2f}% ({n_asym}/{total})")
    print(f"  Realistic generator:  {percentage(n_real):6.2f}% ({n_real}/{total})")
    print(f"  Biased generator:     {percentage(n_bias):6.2f}% ({n_bias}/{total})")
    print(
        f"  Generator not tested: {percentage(n_not_evaluated):6.2f}% "
        f"({n_not_evaluated}/{total})"
    )
