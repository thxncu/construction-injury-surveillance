#!/usr/bin/env python3
"""Prepare the administrative register and estimate coverage-audit models.

Usage
-----
python analysis_main.py --data CSI_analytic_deidentified.csv --output outputs

The source CSV is read as UTF-8 (a byte-order mark is accepted).  All analyses are deterministic.
The primary estimand is an additive interaction in the probability that a
*reported acute-traumatic construction incident* was fatal, conditional on a
recurring event configuration and measured project characteristics.  It is not
an estimate of the total effect of plan coverage on accident incidence or
fatalities.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

WORKFORCE_ORDER = ["19인 이하", "20~49인", "50~99인", "100~299인", "300~499인", "500인 이상"]
WORKFORCE_LABELS = ["<20", "20–49", "50–99", "100–299", "300–499", "500+"]
EXCLUDED_PRIMARY_TYPES = {"질병", "온열질환", "없음", "미입력", "분류불능"}
CONTRACT_SMALL_LEVELS = {
    "1,000만원 미만",
    "1,000만 ~ 2,000만원 미만",
    "2,000만 ~ 4,000만원 미만",
    "4,000만 ~ 1억원 미만",
    "1억 ~ 2억원 미만",
    "2억 ~ 3억원 미만",
    "3억 ~ 5억원 미만",
    "5억 ~ 10억원 미만",
    "10억 ~ 20억원 미만",
    "20억 ~ 50억원 미만",
}
HEIGHT_DETAIL_ORDER = [
    "떨어짐(2미터 미만)",
    "떨어짐(2미터 이상 ~ 3미터 미만)",
    "떨어짐(3미터 이상 ~ 5미터 미만)",
    "떨어짐(5미터 이상 ~ 10미터 미만)",
    "떨어짐(10미터 이상)",
    "떨어짐(분류불능)",
]
HEIGHT_LABELS = ["<2 m", "2–<3 m", "3–<5 m", "5–<10 m", "≥10 m", "Unclassified"]
TYPE_ENGLISH = {
    "넘어짐": "Slip or trip",
    "떨어짐": "Fall from height",
    "물체에 맞음": "Struck by object",
    "끼임": "Caught in or between",
    "부딪힘": "Struck against",
    "기타": "Other",
    "절단, 베임": "Cut or laceration",
    "깔림": "Crushed or buried",
    "질병": "Disease",
    "찔림": "Puncture",
    "분류불능": "Unclassifiable",
    "없음": "None",
    "화재": "Fire",
    "감전": "Electrocution",
    "질식": "Asphyxiation",
    "무너짐": "Collapse",
    "온열질환": "Heat illness",
    "미입력": "Not entered",
}


@dataclass
class FitResult:
    result: object
    data: pd.DataFrame
    y: pd.Series
    X: pd.DataFrame
    group: pd.Series
    fe_means: pd.Series | None = None


@dataclass
class SimpleWaldResult:
    statistic: float
    pvalue: float
    df_denom: int


class SimpleOLSResult:
    """Minimal statsmodels-compatible result used by this replication script."""

    def __init__(self, names: Sequence[str], beta: np.ndarray, cov: np.ndarray, nobs: int):
        self.params = pd.Series(np.asarray(beta, dtype=float), index=list(names), dtype=float)
        self._cov = pd.DataFrame(np.asarray(cov, dtype=float), index=list(names), columns=list(names))
        self.bse = pd.Series(np.sqrt(np.maximum(0.0, np.diag(cov))), index=list(names), dtype=float)
        z = self.params.to_numpy() / self.bse.replace(0, np.nan).to_numpy()
        self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(z)), index=list(names), dtype=float)
        self.nobs = int(nobs)

    def cov_params(self):
        return self._cov.copy()

    def wald_test(self, R: np.ndarray, scalar: bool = True) -> SimpleWaldResult:
        R = np.asarray(R, dtype=float)
        q = R @ self.params.to_numpy()
        V = R @ self._cov.to_numpy() @ R.T
        rank = int(np.linalg.matrix_rank(V, tol=max(V.shape) * np.finfo(float).eps * max(1.0, np.linalg.norm(V, 2))))
        if rank == 0:
            return SimpleWaldResult(float("nan"), float("nan"), 0)
        stat = float(q.T @ np.linalg.pinv(V) @ q)
        return SimpleWaldResult(stat, float(stats.chi2.sf(stat, rank)), rank)


def manual_ols_result(
    y: np.ndarray,
    X: np.ndarray,
    names: Sequence[str],
    cov_type: str,
    clusters: np.ndarray | None = None,
) -> SimpleOLSResult:
    """OLS with HC1 or one-way cluster-robust covariance."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    pinvX = np.linalg.pinv(X)
    beta = pinvX @ y
    resid = y - X @ beta
    bread = pinvX @ pinvX.T
    n, p = X.shape
    rank = int(np.linalg.matrix_rank(X))
    if cov_type == "cluster":
        if clusters is None:
            raise ValueError("clusters are required for cluster covariance")
        codes = pd.Categorical(clusters).codes
        G = int(codes.max()) + 1
        scores = np.zeros((G, p), dtype=float)
        np.add.at(scores, codes, X * resid[:, None])
        meat = scores.T @ scores
        correction = (G / (G - 1)) * ((n - 1) / (n - p))
        cov = bread @ meat @ bread * correction
    elif cov_type == "HC1":
        meat = X.T @ ((resid**2)[:, None] * X)
        cov = bread @ meat @ bread * (n / (n - rank))
    else:
        sigma2 = float(resid @ resid) / (n - rank)
        cov = bread * sigma2
    return SimpleOLSResult(names, beta, cov, n)


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def ci_from_beta(beta: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    z = stats.norm.ppf(1 - alpha / 2)
    return beta - z * se, beta + z * se


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return center - half, center + half


def prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = [
        "사고일시", "사망자", "부상자", "인적사고종류(대분류)", "인적사고종류",
        "사고객체(대분류)", "사고객체(소분류)", "작업프로세스", "공정율", "공사비",
        "작업자수", "안전관리계획", "공공민간구분", "낙찰율", "지상층수", "지하층수", "공사종류",
    ]
    ensure_columns(df, required)
    df = df.copy()
    df["date"] = pd.to_datetime(df["사고일시"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"Could not parse {int(df['date'].isna().sum())} accident dates")
    df["fatal"] = (pd.to_numeric(df["사망자"], errors="coerce").fillna(0) > 0).astype(int)
    df["casualty"] = (
        pd.to_numeric(df["사망자"], errors="coerce").fillna(0)
        + pd.to_numeric(df["부상자"], errors="coerce").fillna(0)
        > 0
    )
    df["plan_complete"] = df["안전관리계획"].isin(["대상현장(1/2종)", "대상현장(1/2종외)", "비대상현장"])
    df["workforce_complete"] = df["작업자수"].isin(WORKFORCE_ORDER)
    df["covered"] = df["안전관리계획"].isin(["대상현장(1/2종)", "대상현장(1/2종외)"]).astype(int)
    df["exempt"] = (df["안전관리계획"] == "비대상현장").astype(int)
    df["small50"] = df["작업자수"].isin(["19인 이하", "20~49인"]).astype(int)
    df["small20"] = (df["작업자수"] == "19인 이하").astype(int)
    df["small100"] = df["작업자수"].isin(["19인 이하", "20~49인", "50~99인"]).astype(int)
    df["event_cell"] = (
        df["인적사고종류"].astype(str) + " || "
        + df["사고객체(소분류)"].astype(str) + " || "
        + df["작업프로세스"].astype(str)
    )
    df["event_type_object_broad"] = df["인적사고종류"].astype(str) + " || " + df["사고객체(대분류)"].astype(str)
    df["event_type_process"] = df["인적사고종류"].astype(str) + " || " + df["작업프로세스"].astype(str)
    df["event_type_object_detail"] = df["인적사고종류"].astype(str) + " || " + df["사고객체(소분류)"].astype(str)
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["accident_date"] = df["date"].dt.date.astype(str)
    df["acute_type"] = ~df["인적사고종류(대분류)"].isin(EXCLUDED_PRIMARY_TYPES)
    df["primary_eligible"] = df["workforce_complete"] & df["plan_complete"] & df["casualty"] & df["acute_type"]
    df["workforce_ordered"] = pd.Categorical(df["작업자수"], categories=WORKFORCE_ORDER, ordered=True)
    df["small_contract"] = df["공사비"].isin(CONTRACT_SMALL_LEVELS).astype(int)
    df["contract_complete"] = ~df["공사비"].isin(["미입력", "분류불능"])
    return df


def retain_recurring(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sizes = df.groupby(group_col, observed=True).size()
    keep = sizes[sizes >= 2].index
    retained = df[df[group_col].isin(keep)].copy()
    excluded = df[~df[group_col].isin(keep)].copy()
    return retained, excluded


def design_matrix(
    data: pd.DataFrame,
    main_cols: Sequence[str],
    categorical_controls: Sequence[str] = (),
    drop_first: bool = True,
) -> pd.DataFrame:
    X = data[list(main_cols)].astype(float).copy()
    for c in categorical_controls:
        d = pd.get_dummies(data[c].astype(str), prefix=c, drop_first=drop_first, dtype=float)
        X = pd.concat([X, d], axis=1)
    return X


def fit_ols(
    data: pd.DataFrame,
    main_cols: Sequence[str],
    categorical_controls: Sequence[str] = (),
    fe_col: str | None = None,
    cluster_col: str | None = None,
    cov_type: str = "cluster",
) -> FitResult:
    work = data.copy()
    X = design_matrix(work, main_cols, categorical_controls)
    y = work["fatal"].astype(float)
    fe_means = None
    if fe_col is not None:
        group = work[fe_col].astype(str)
        y_dm = y - y.groupby(group).transform("mean")
        X_dm = X - X.groupby(group).transform("mean")
        vary = X_dm.abs().sum(axis=0) > 1e-12
        X_dm = X_dm.loc[:, vary]
        cluster = work[cluster_col or fe_col].astype(str).to_numpy() if cov_type == "cluster" else None
        result = manual_ols_result(
            y_dm.to_numpy(dtype=float), X_dm.to_numpy(dtype=float), list(X_dm.columns), cov_type, cluster
        )
        xb = X[result.params.index].to_numpy() @ result.params.to_numpy()
        fe_means = pd.Series(y.to_numpy() - xb, index=work.index).groupby(group).mean()
        return FitResult(result=result, data=work, y=y, X=X, group=group, fe_means=fe_means)
    Xc = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
    cluster = work[cluster_col or "event_cell"].astype(str).to_numpy() if cov_type == "cluster" else None
    result = manual_ols_result(y.to_numpy(dtype=float), Xc.to_numpy(dtype=float), list(Xc.columns), cov_type, cluster)
    return FitResult(result=result, data=work, y=y, X=Xc, group=work[cluster_col or "event_cell"].astype(str))


def coef_row(fit: FitResult, term: str, label: str, scale: float = 100.0) -> dict:
    r = fit.result
    b = float(r.params[term])
    se = float(r.bse[term])
    lo, hi = ci_from_beta(b, se)
    return {
        "Specification": label,
        "n": int(r.nobs),
        "Coefficient_pp": b * scale,
        "CI_low_pp": lo * scale,
        "CI_high_pp": hi * scale,
        "p": float(r.pvalues[term]),
    }


def linear_contrast(fit: FitResult, weights: dict[str, float], label: str, scale: float = 100.0) -> dict:
    r = fit.result
    names = list(r.params.index)
    w = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    b = float(w @ r.params.to_numpy())
    cov = np.asarray(r.cov_params())
    se = float(math.sqrt(max(0.0, w @ cov @ w)))
    lo, hi = ci_from_beta(b, se)
    p = 2 * stats.norm.sf(abs(b / se)) if se > 0 else np.nan
    return {
        "Contrast": label,
        "Estimate_pp": b * scale,
        "CI_low_pp": lo * scale,
        "CI_high_pp": hi * scale,
        "p": p,
    }


def standardized_probability(fit: FitResult, assignments: dict[str, float], label: str) -> dict:
    if fit.fe_means is None:
        raise ValueError("Standardized predictions require an absorbed fixed-effect fit")
    Xcf = fit.X.copy()
    for term, value in assignments.items():
        if term in Xcf.columns:
            Xcf[term] = value
    # Ensure the interaction is consistent if both components are assigned.
    if "small" in Xcf.columns and "exempt" in Xcf.columns and "interaction" in Xcf.columns:
        Xcf["interaction"] = Xcf["small"] * Xcf["exempt"]
    params = fit.result.params
    pred = Xcf[params.index].to_numpy() @ params.to_numpy()
    fe = fit.group.map(fit.fe_means).to_numpy()
    estimate = float(np.mean(pred + fe))
    # Delta-method variance for slopes; recovered FE treated as nuisance estimates.
    xbar = Xcf[params.index].mean(axis=0).to_numpy()
    cov = np.asarray(fit.result.cov_params())
    se = float(math.sqrt(max(0.0, xbar @ cov @ xbar)))
    lo, hi = ci_from_beta(estimate, se)
    return {"Group": label, "Probability": estimate, "CI_low": lo, "CI_high": hi}


def bootstrap_standardized_probabilities(
    fit: FitResult,
    assignments: Sequence[tuple[str, dict[str, float]]],
    reps: int = 500,
    seed: int = 20260828,
) -> pd.DataFrame:
    """Cluster-bootstrap standardized probabilities by event configuration.

    The bootstrap resamples event configurations with replacement.  To keep
    the workflow practical, the fixed-effect normal equations and the
    counterfactual prediction contributions are pre-aggregated by cluster.
    This is algebraically equivalent to rebuilding the duplicated bootstrap
    sample, but it avoids solving a large least-squares problem in every draw.
    Percentile intervals are reported.
    """
    if fit.fe_means is None:
        raise ValueError("Bootstrap predictions require an absorbed fixed-effect fit")
    names = list(fit.result.params.index)
    X = fit.X[names].to_numpy(dtype=float)
    y = fit.y.to_numpy(dtype=float)
    codes = pd.Categorical(fit.group).codes
    unique_codes = np.unique(codes)
    group_indices = [np.flatnonzero(codes == g) for g in unique_codes]
    G = len(group_indices)
    p = X.shape[1]
    rng = np.random.default_rng(seed)

    n_g = np.empty(G, dtype=float)
    ysum_g = np.empty(G, dtype=float)
    xsum_g = np.empty((G, p), dtype=float)
    xtx_g = np.empty((G, p, p), dtype=float)
    xty_g = np.empty((G, p), dtype=float)
    for g, idx in enumerate(group_indices):
        Xg = X[idx]
        yg = y[idx]
        n = len(idx)
        xm = Xg.mean(axis=0)
        ym = yg.mean()
        Xc = Xg - xm
        yc = yg - ym
        n_g[g] = n
        ysum_g[g] = yg.sum()
        xsum_g[g] = Xg.sum(axis=0)
        xtx_g[g] = Xc.T @ Xc
        xty_g[g] = Xc.T @ yc

    # q[g, s, :] is the change in the group sum of X under setting s.
    q = np.empty((G, len(assignments), p), dtype=float)
    for j, (_, setting) in enumerate(assignments):
        xcf_sum = xsum_g.copy()
        for term, value in setting.items():
            if term in names:
                xcf_sum[:, names.index(term)] = value * n_g
        if all(term in names for term in ("small", "exempt", "interaction")):
            sval = setting.get("small")
            eval_ = setting.get("exempt")
            if sval is not None and eval_ is not None:
                xcf_sum[:, names.index("interaction")] = (sval * eval_) * n_g
        q[:, j, :] = xcf_sum - xsum_g

    beta0 = fit.result.params.to_numpy(dtype=float)
    total_n = n_g.sum()
    point = (ysum_g.sum() + np.einsum("gsp,p->s", q, beta0)) / total_n

    draws = np.empty((reps, len(assignments)), dtype=float)
    for r in range(reps):
        sampled = rng.integers(0, G, G)
        w = np.bincount(sampled, minlength=G).astype(float)
        A = np.tensordot(w, xtx_g, axes=(0, 0))
        b = w @ xty_g
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
        denom = float(w @ n_g)
        ysum = float(w @ ysum_g)
        qsum = np.einsum("g,gsp->sp", w, q)
        draws[r, :] = (ysum + qsum @ beta) / denom

    rows = []
    for j, (label, _) in enumerate(assignments):
        rows.append({
            "Group": label,
            "Probability": float(point[j]),
            "CI_low": float(np.quantile(draws[:, j], 0.025)),
            "CI_high": float(np.quantile(draws[:, j], 0.975)),
            "Bootstrap_reps": reps,
            "Bootstrap_seed": seed,
        })
    return pd.DataFrame(rows)


def binomial_table(data: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    out = (
        data.groupby(list(group_cols), observed=True)["fatal"]
        .agg(Accidents="size", Deaths="sum", Fatality="mean")
        .reset_index()
    )
    cis = [wilson_ci(int(k), int(n)) for k, n in zip(out["Deaths"], out["Accidents"])]
    out["CI_low"] = [x[0] for x in cis]
    out["CI_high"] = [x[1] for x in cis]
    return out


def make_model_ladder(primary_frame: pd.DataFrame, primary_fe: pd.DataFrame) -> tuple[pd.DataFrame, FitResult]:
    primary_fe = primary_fe.copy()
    primary_frame = primary_frame.copy()
    for d in (primary_frame, primary_fe):
        d["small"] = d["small50"].astype(float)
        d["interaction"] = d["small"] * d["exempt"]
    rows: list[dict] = []
    fit0 = fit_ols(primary_frame, ["small", "exempt", "interaction"], fe_col=None, cluster_col="event_cell")
    fits = [
        (fit0, "1 No fixed effect or project controls"),
        (fit_ols(primary_fe, ["small", "exempt", "interaction"], fe_col="event_cell"), "2 Event-configuration fixed effect"),
        (fit_ols(primary_fe, ["small", "exempt", "interaction"], ["공정율"], fe_col="event_cell"), "3 Adds progress rate"),
        (fit_ols(primary_fe, ["small", "exempt", "interaction"], ["공사비"], fe_col="event_cell"), "4 Adds contract value"),
        (fit_ols(primary_fe, ["small", "exempt", "interaction"], ["공정율", "공사비"], fe_col="event_cell"), "5 Adds progress and contract value"),
        (fit_ols(primary_fe, ["small", "exempt", "interaction"], ["공정율", "공사비", "공공민간구분"], fe_col="event_cell"), "6 Adds ownership (primary)"),
        (fit_ols(primary_fe, ["small", "exempt", "interaction"], ["공정율", "공사비", "공공민간구분", "year"], fe_col="event_cell"), "7 Adds calendar year"),
    ]
    for fit, label in fits:
        r = fit.result
        row = {"Specification": label, "n": int(r.nobs)}
        for term, prefix in [("small", "Small"), ("exempt", "Exempt"), ("interaction", "Interaction")]:
            b = float(r.params[term]) * 100
            se = float(r.bse[term]) * 100
            lo, hi = ci_from_beta(b, se)
            row[f"{prefix}_pp"] = b
            row[f"{prefix}_low"] = lo
            row[f"{prefix}_high"] = hi
            row[f"{prefix}_p"] = float(r.pvalues[term])
        rows.append(row)
    return pd.DataFrame(rows), fits[5][0]


def alternative_contract_fit(primary_frame: pd.DataFrame) -> tuple[FitResult, pd.DataFrame]:
    d = primary_frame[primary_frame["contract_complete"]].copy()
    d, _ = retain_recurring(d, "event_cell")
    d["small"] = d["small_contract"].astype(float)
    d["interaction"] = d["small"] * d["exempt"]
    fit = fit_ols(
        d, ["small", "exempt", "interaction"], ["공정율", "작업자수", "공공민간구분"], fe_col="event_cell"
    )
    return fit, d


def numeric_floor(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.where(~series.eq("미입력")), errors="coerce")


def label_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    basement = numeric_floor(frame["지하층수"]) >= 3
    stories = numeric_floor(frame["지상층수"]) >= 10
    railway = frame["공사종류"].str.contains("철도", na=False)
    plant = frame["사고객체(소분류)"].isin(["타워크레인", "항타 및 항발기", "천공기"])
    conditions = {
        "Three or more basement floors (proxy for excavation ≥10 m)": basement,
        "Ten or more storeys above ground": stories,
        "Railway or urban-railway work": railway,
        "Tower crane, pile driver/extractor, or drilling rig": plant,
    }
    rows = []
    for label, mask in conditions.items():
        n = int(mask.sum())
        ex = int((mask & frame["exempt"].eq(1)).sum())
        rows.append({"Condition": label, "Records": n, "Exempt_records": ex, "Exempt_percent": 100 * ex / n if n else np.nan})
    any_cond = np.logical_or.reduce([m.to_numpy() for m in conditions.values()])
    any_cond = pd.Series(any_cond, index=frame.index)
    exempt_total = int(frame["exempt"].sum())
    candidate = any_cond & frame["exempt"].eq(1)
    rows.append({
        "Condition": "Any reconstructible condition among exempt records",
        "Records": exempt_total,
        "Exempt_records": int(candidate.sum()),
        "Exempt_percent": 100 * candidate.sum() / exempt_total,
    })
    rows.append({
        "Condition": "Complete analytic frame (baseline)",
        "Records": int(len(frame)),
        "Exempt_records": exempt_total,
        "Exempt_percent": 100 * exempt_total / len(frame),
    })
    return pd.DataFrame(rows), candidate


def same_spec_robustness(
    all_df: pd.DataFrame, primary_frame: pd.DataFrame, primary_fit: FitResult, candidate_all: pd.Series
) -> pd.DataFrame:
    rows = [coef_row(primary_fit, "interaction", "Reference pooled binary sample")]

    full = all_df[all_df["workforce_complete"] & all_df["plan_complete"]].copy()
    full, _ = retain_recurring(full, "event_cell")
    full["small"] = full["small50"].astype(float)
    full["interaction"] = full["small"] * full["exempt"]
    fit_full = fit_ols(full, ["small", "exempt", "interaction"], ["공정율", "공사비", "공공민간구분"], fe_col="event_cell")
    rows.append(coef_row(fit_full, "interaction", "Full administrative register"))

    no_other = primary_frame[primary_frame["인적사고종류(대분류)"] != "기타"].copy()
    no_other, _ = retain_recurring(no_other, "event_cell")
    no_other["small"] = no_other["small50"].astype(float)
    no_other["interaction"] = no_other["small"] * no_other["exempt"]
    fit_no_other = fit_ols(no_other, ["small", "exempt", "interaction"], ["공정율", "공사비", "공공민간구분"], fe_col="event_cell")
    rows.append(coef_row(fit_no_other, "interaction", "Acute-traumatic sample excluding broad 'Other'"))

    # Label-cleaned, post-amendment, and combined restrictions, always with primary controls.
    candidate_primary = candidate_all.reindex(primary_frame.index).fillna(False)
    cleaned_frame = primary_frame[~candidate_primary].copy()
    for label, d in [
        ("Candidate label inconsistencies removed", cleaned_frame),
        ("From February 2021", primary_frame[primary_frame["date"] >= pd.Timestamp("2021-02-01")].copy()),
        (
            "Both label cleaning and February 2021 restriction",
            cleaned_frame[cleaned_frame["date"] >= pd.Timestamp("2021-02-01")].copy(),
        ),
    ]:
        d, _ = retain_recurring(d, "event_cell")
        d["small"] = d["small50"].astype(float)
        d["interaction"] = d["small"] * d["exempt"]
        fit = fit_ols(d, ["small", "exempt", "interaction"], ["공정율", "공사비", "공공민간구분"], fe_col="event_cell")
        rows.append(coef_row(fit, "interaction", label))

    award = primary_frame.copy()
    award, _ = retain_recurring(award, "event_cell")
    award["small"] = award["small50"].astype(float)
    award["interaction"] = award["small"] * award["exempt"]
    fit_award = fit_ols(
        award,
        ["small", "exempt", "interaction"],
        ["공정율", "공사비", "공공민간구분", "낙찰율"],
        fe_col="event_cell",
    )
    rows.append(coef_row(fit_award, "interaction", "Adds award-rate category"))
    return pd.DataFrame(rows)


def threshold_sensitivity(primary_fe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [("small20", "<20 workers"), ("small50", "<50 workers"), ("small100", "<100 workers")]:
        d = primary_fe.copy()
        d["small"] = d[col].astype(float)
        d["interaction"] = d["small"] * d["exempt"]
        fit = fit_ols(d, ["small", "exempt", "interaction"], ["공정율", "공사비", "공공민간구분"], fe_col="event_cell")
        rows.append(coef_row(fit, "interaction", label))
    return pd.DataFrame(rows)


FE_LEVELS = [
    ("인적사고종류", "Detailed accident type"),
    ("event_type_object_broad", "Type × broad object"),
    ("event_type_process", "Type × work process"),
    ("event_type_object_detail", "Type × detailed object"),
    ("event_cell", "Type × detailed object × process"),
]


def manual_within_cluster_term(
    data: pd.DataFrame,
    main_cols: Sequence[str],
    categorical_controls: Sequence[str],
    fe_col: str,
    cluster_col: str,
    term: str,
) -> tuple[float, float, float]:
    """Within-OLS coefficient and cluster-robust SE without statsmodels.

    The covariance uses the same small-sample factor as statsmodels'
    ``cov_type='cluster', use_correction=True``.  This implementation avoids
    intermittent BLAS stalls observed when many robust fits are run in one
    replication workflow.
    """
    Xdf = design_matrix(data, main_cols, categorical_controls)
    y = data["fatal"].astype(float)
    groups = data[fe_col].astype(str)
    ydm = y - y.groupby(groups).transform("mean")
    Xdm = Xdf - Xdf.groupby(groups).transform("mean")
    vary = Xdm.abs().sum(axis=0) > 1e-12
    Xdm = Xdm.loc[:, vary]
    names = list(Xdm.columns)
    X = Xdm.to_numpy(dtype=float)
    yy = ydm.to_numpy(dtype=float)
    pinvX = np.linalg.pinv(X)
    beta = pinvX @ yy
    resid = yy - X @ beta
    bread = pinvX @ pinvX.T
    clusters = pd.Categorical(data[cluster_col].astype(str)).codes
    G = int(clusters.max()) + 1
    scores = np.zeros((G, X.shape[1]), dtype=float)
    np.add.at(scores, clusters, X * resid[:, None])
    meat = scores.T @ scores
    n = X.shape[0]
    k = X.shape[1]
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    cov = bread @ meat @ bread * correction
    j = names.index(term)
    return float(beta[j]), float(math.sqrt(max(0.0, cov[j, j]))), float(2 * stats.norm.sf(abs(beta[j] / math.sqrt(max(0.0, cov[j, j])))))


def fe_granularity_one(primary_frame: pd.DataFrame, level_index: int) -> pd.DataFrame:
    col, label = FE_LEVELS[level_index]
    retained, singleton = retain_recurring(primary_frame, col)
    retained = retained.copy()
    retained["small"] = retained["small50"].astype(float)
    retained["interaction"] = retained["small"] * retained["exempt"]
    b, se, p = manual_within_cluster_term(
        retained, ["small", "exempt", "interaction"],
        ["공정율", "공사비", "공공민간구분"], fe_col=col,
        cluster_col="event_cell", term="interaction"
    )
    lo, hi = ci_from_beta(b, se)
    row = {
        "Specification": label,
        "n": int(len(retained)),
        "Coefficient_pp": b * 100,
        "CI_low_pp": lo * 100,
        "CI_high_pp": hi * 100,
        "p": p,
    }
    group_stats = retained.groupby(col, observed=True).agg(
        n=("fatal", "size"), yvar=("fatal", "nunique"), tvar=("interaction", "nunique")
    )
    row.update({
        "Groups": int(retained[col].nunique()),
        "Singletons_excluded": int(len(singleton)),
        "Singleton_fatality_percent": 100 * singleton["fatal"].mean() if len(singleton) else np.nan,
        "Groups_with_outcome_variation": int((group_stats["yvar"] > 1).sum()),
        "Groups_with_interaction_variation": int((group_stats["tvar"] > 1).sum()),
    })
    return pd.DataFrame([row])


def fe_granularity_ladder(primary_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [fe_granularity_one(primary_frame, i) for i in range(len(FE_LEVELS))],
        ignore_index=True,
    )


def conditional_logit(primary_fe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = primary_fe.copy()
    # Conditional likelihood only uses strata with both fatal and nonfatal outcomes.
    varying = d.groupby("event_cell")["fatal"].nunique()
    keep = varying[varying > 1].index
    d = d[d["event_cell"].isin(keep)].copy()
    d["small"] = d["small50"].astype(float)
    d["interaction"] = d["small"] * d["exempt"]
    X = design_matrix(d, ["small", "exempt", "interaction"], ["공정율", "공사비", "공공민간구분"])
    # Remove columns with no variation in the conditional sample.
    X = X.loc[:, X.nunique() > 1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ConditionalLogit(d["fatal"].astype(float), X.astype(float), groups=d["event_cell"].astype(str))
        result = model.fit(method="lbfgs", maxiter=500, disp=False)
        warning_text = [str(x.message) for x in w]
    rows = []
    for term, label in [("small", "Small (<50 workers)"), ("exempt", "Exempt"), ("interaction", "Small × exempt")]:
        b = float(result.params[term])
        se = float(result.bse[term])
        lo, hi = ci_from_beta(b, se)
        rows.append({
            "Term": label,
            "Odds_ratio": math.exp(b),
            "CI_low": math.exp(lo),
            "CI_high": math.exp(hi),
            "p": float(result.pvalues[term]),
        })
    meta = {
        "n": int(len(d)),
        "strata": int(d["event_cell"].nunique()),
        "covariates": int(X.shape[1]),
        "log_likelihood": float(result.llf),
        "converged": bool(result.mle_retvals.get("converged", True)) if hasattr(result, "mle_retvals") else True,
        "warnings": warning_text,
        "standard_errors": "model-based conditional-likelihood standard errors; clustering not available in statsmodels ConditionalLogit",
    }
    return pd.DataFrame(rows), meta


def cluster_sensitivity(primary_fe: pd.DataFrame) -> pd.DataFrame:
    d = primary_fe.copy()
    d["small"] = d["small50"].astype(float)
    d["interaction"] = d["small"] * d["exempt"]
    rows = []
    for label, cov_type, cluster in [
        ("HC1 heteroskedasticity-robust", "HC1", None),
        ("Clustered by event configuration (primary)", "cluster", "event_cell"),
        ("Clustered by calendar month", "cluster", "month"),
        ("Clustered by accident date", "cluster", "accident_date"),
    ]:
        fit = fit_ols(
            d,
            ["small", "exempt", "interaction"],
            ["공정율", "공사비", "공공민간구분"],
            fe_col="event_cell",
            cluster_col=cluster,
            cov_type=cov_type,
        )
        rows.append(coef_row(fit, "interaction", label))
    return pd.DataFrame(rows)


def six_band_analysis(primary_fe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = primary_fe.copy()
    # Reference band is 500+; construct dummies for first five bands and their interactions.
    main_cols = ["exempt"]
    for cat, label in zip(WORKFORCE_ORDER[:-1], WORKFORCE_LABELS[:-1]):
        col = f"wf_{label}"
        icol = f"int_{label}"
        d[col] = (d["작업자수"] == cat).astype(float)
        d[icol] = d[col] * d["exempt"]
        main_cols.extend([col, icol])
    fit = fit_ols(d, main_cols, ["공정율", "공사비", "공공민간구분"], fe_col="event_cell")
    r = fit.result
    contrast_rows = []
    # Within-band exemption contrast: beta_exempt + band-specific interaction, reference has beta_exempt.
    for label in WORKFORCE_LABELS:
        weights = {"exempt": 1.0}
        if label != "500+":
            weights[f"int_{label}"] = 1.0
        contrast_rows.append(linear_contrast(fit, weights, label))
    contrasts = pd.DataFrame(contrast_rows)

    # Standardized probabilities by band and plan status.
    pred_rows = []
    params = fit.result.params
    cov = np.asarray(fit.result.cov_params())
    for label in WORKFORCE_LABELS:
        for exempt in [0.0, 1.0]:
            Xcf = fit.X.copy()
            Xcf["exempt"] = exempt
            for lab2 in WORKFORCE_LABELS[:-1]:
                Xcf[f"wf_{lab2}"] = 1.0 if label == lab2 else 0.0
                Xcf[f"int_{lab2}"] = Xcf[f"wf_{lab2}"] * exempt
            pred = Xcf[params.index].to_numpy() @ params.to_numpy() + fit.group.map(fit.fe_means).to_numpy()
            est = float(np.mean(pred))
            xbar = Xcf[params.index].mean(axis=0).to_numpy()
            se = float(math.sqrt(max(0.0, xbar @ cov @ xbar)))
            lo, hi = ci_from_beta(est, se)
            pred_rows.append({"Workforce": label, "Plan_status": "Exempt" if exempt else "Covered", "Probability": est, "CI_low": lo, "CI_high": hi})
    predictions = pd.DataFrame(pred_rows)

    # Omnibus test of five interaction terms.
    terms = [f"int_{x}" for x in WORKFORCE_LABELS[:-1]]
    R = np.zeros((len(terms), len(params)))
    names = list(params.index)
    for i, term in enumerate(terms):
        R[i, names.index(term)] = 1.0
    test = r.wald_test(R, scalar=True)
    omnibus = {"wald_chi2": float(test.statistic), "df": int(len(terms)), "p": float(test.pvalue)}
    return contrasts, predictions, omnibus


def all_type_tables(df: pd.DataFrame, primary_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    complete = df[df["workforce_complete"] & df["plan_complete"]].copy()
    all_types = (
        complete.groupby("인적사고종류(대분류)", observed=True)
        .agg(Accidents=("fatal", "size"), Deaths=("fatal", "sum"), Zero_casualty=("casualty", lambda s: int((~s).sum())))
        .reset_index()
        .rename(columns={"인적사고종류(대분류)": "Accident_type_Korean"})
        .sort_values("Accidents", ascending=False)
    )
    all_types.insert(1, "Accident_type", all_types["Accident_type_Korean"].map(TYPE_ENGLISH))
    rows = []
    for typ, g in primary_frame.groupby("인적사고종류(대분류)", observed=True):
        small = g[g["small50"] == 1]
        large = g[g["small50"] == 0]
        ps = small["fatal"].mean() if len(small) else np.nan
        pl = large["fatal"].mean() if len(large) else np.nan
        rd = ps - pl if len(small) and len(large) else np.nan
        se = math.sqrt(ps * (1 - ps) / len(small) + pl * (1 - pl) / len(large)) if len(small) and len(large) else np.nan
        lo, hi = ci_from_beta(rd, se) if np.isfinite(se) else (np.nan, np.nan)
        rows.append({
            "Accident_type_Korean": typ,
            "Accident_type": TYPE_ENGLISH.get(typ, typ),
            "Accidents": int(len(g)),
            "Deaths": int(g["fatal"].sum()),
            "Small_n": int(len(small)),
            "Large_n": int(len(large)),
            "Small_fatality_percent": 100 * ps if np.isfinite(ps) else np.nan,
            "Large_fatality_percent": 100 * pl if np.isfinite(pl) else np.nan,
            "Risk_difference_pp": 100 * rd if np.isfinite(rd) else np.nan,
            "CI_low_pp": 100 * lo if np.isfinite(lo) else np.nan,
            "CI_high_pp": 100 * hi if np.isfinite(hi) else np.nan,
        })
    type_rd = pd.DataFrame(rows).sort_values("Accidents", ascending=False)

    # Omnibus workforce-band by broad-type heterogeneity for types with at least 50 records.
    eligible_types = type_rd.loc[type_rd["Accidents"] >= 50, "Accident_type_Korean"].tolist()
    dd = primary_frame[primary_frame["인적사고종류(대분류)"].isin(eligible_types)].copy()
    # Simple OLS with type and workforce dummies plus all interactions, no project controls; this tests heterogeneity in raw profiles.
    y = dd["fatal"].astype(float)
    type_d = pd.get_dummies(dd["인적사고종류(대분류)"], prefix="type", drop_first=True, dtype=float)
    wf_d = pd.get_dummies(dd["작업자수"], prefix="wf", drop_first=True, dtype=float)
    interactions = {}
    for tc in type_d.columns:
        for wc in wf_d.columns:
            interactions[f"{tc}__{wc}"] = type_d[tc] * wf_d[wc]
    X = pd.concat([type_d, wf_d, pd.DataFrame(interactions)], axis=1)
    X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
    rr = manual_ols_result(y.to_numpy(dtype=float), X.to_numpy(dtype=float), list(X.columns), "HC1")
    int_cols = list(interactions.keys())
    R = np.zeros((len(int_cols), len(rr.params)))
    names = list(rr.params.index)
    for i, c in enumerate(int_cols):
        R[i, names.index(c)] = 1
    w = rr.wald_test(R, scalar=True)
    omnibus = {
        "wald_chi2": float(w.statistic),
        "effective_df": int(w.df_denom),
        "nominal_constraints": int(len(int_cols)),
        "p": float(w.pvalue),
        "types": len(eligible_types),
        "note": "The robust covariance matrix had reduced rank; effective degrees of freedom are reported.",
    }
    return all_types, type_rd, omnibus


def fall_height_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    falls = df[df["workforce_complete"] & df["plan_complete"] & (df["인적사고종류(대분류)"] == "떨어짐")].copy()
    falls["Height"] = pd.Categorical(falls["인적사고종류"], categories=HEIGHT_DETAIL_ORDER, ordered=True)
    height = binomial_table(falls.dropna(subset=["Height"]), ["Height"])
    height["Height_label"] = height["Height"].map(dict(zip(HEIGHT_DETAIL_ORDER, HEIGHT_LABELS))).astype(str)

    comp = (
        falls.groupby(["작업자수", "Height"], observed=True)["fatal"]
        .agg(Accidents="size", Deaths="sum")
        .reset_index()
    )
    comp["Workforce"] = comp["작업자수"].map(dict(zip(WORKFORCE_ORDER, WORKFORCE_LABELS)))
    comp["Height_label"] = comp["Height"].map(dict(zip(HEIGHT_DETAIL_ORDER, HEIGHT_LABELS))).astype(str)
    comp["Share"] = comp["Accidents"] / comp.groupby("작업자수")["Accidents"].transform("sum")

    classified = falls[falls["인적사고종류"].isin(HEIGHT_DETAIL_ORDER[:-1])].copy()
    overall_weights = classified["인적사고종류"].value_counts(normalize=True).reindex(HEIGHT_DETAIL_ORDER[:-1]).fillna(0)
    std_rows = []
    for wf, wf_label in zip(WORKFORCE_ORDER, WORKFORCE_LABELS):
        g_all = falls[falls["작업자수"] == wf]
        g = classified[classified["작업자수"] == wf]
        raw = g_all["fatal"].mean()
        raw_se = math.sqrt(raw * (1 - raw) / len(g_all))
        rlo, rhi = ci_from_beta(raw, raw_se)
        rates = g.groupby("인적사고종류")["fatal"].agg(["mean", "count"])
        # Only report standardized rate when every classified height category is represented.
        if set(HEIGHT_DETAIL_ORDER[:-1]).issubset(rates.index):
            std = float(sum(overall_weights[h] * rates.loc[h, "mean"] for h in HEIGHT_DETAIL_ORDER[:-1]))
            var = float(sum((overall_weights[h] ** 2) * rates.loc[h, "mean"] * (1 - rates.loc[h, "mean"]) / rates.loc[h, "count"] for h in HEIGHT_DETAIL_ORDER[:-1]))
            se = math.sqrt(var)
            slo, shi = ci_from_beta(std, se)
        else:
            std, slo, shi = np.nan, np.nan, np.nan
        std_rows.append({
            "Workforce": wf_label,
            "All_falls_n": int(len(g_all)),
            "Classified_height_n": int(len(g)),
            "Raw_fatality": raw,
            "Raw_CI_low": rlo,
            "Raw_CI_high": rhi,
            "Height_standardized_fatality": std,
            "Standardized_CI_low": slo,
            "Standardized_CI_high": shi,
        })
    standardized = pd.DataFrame(std_rows)
    return height, comp, standardized


def reporting_sensitivity(primary_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def required_ratio(g1: pd.DataFrame, g0: pd.DataFrame) -> float:
        # Fatal cases assumed fully observed. Find relative nonfatal completeness r for group 1
        # such that corrected fatality equals observed fatality in group 0.
        f1 = float(g1["fatal"].sum())
        n1 = float((1 - g1["fatal"]).sum())
        p0 = float(g0["fatal"].mean())
        return p0 * n1 / (f1 * (1 - p0)) if f1 > 0 and p0 < 1 else np.nan

    comparisons = [
        ("Exempt versus covered, whole acute frame", primary_frame[primary_frame["exempt"] == 1], primary_frame[primary_frame["exempt"] == 0]),
        (
            "Exempt versus covered, under 20 workers",
            primary_frame[(primary_frame["작업자수"] == "19인 이하") & (primary_frame["exempt"] == 1)],
            primary_frame[(primary_frame["작업자수"] == "19인 이하") & (primary_frame["exempt"] == 0)],
        ),
        (
            "Under 20 versus 500+ workers",
            primary_frame[primary_frame["작업자수"] == "19인 이하"],
            primary_frame[primary_frame["작업자수"] == "500인 이상"],
        ),
    ]
    rows = []
    for label, g1, g0 in comparisons:
        r = required_ratio(g1, g0)
        rows.append({"Comparison": label, "Required_relative_nonfatal_completeness": r, "Implied_share_unrecorded_relative": 1 - r})
    table = pd.DataFrame(rows)

    # Curve for the under-20 exemption comparison.
    g1, g0 = comparisons[1][1], comparisons[1][2]
    f = float(g1["fatal"].sum())
    nf = float((1 - g1["fatal"]).sum())
    ref = float(g0["fatal"].mean())
    ratios = np.linspace(0.2, 1.0, 161)
    curve = pd.DataFrame({"Relative_completeness": ratios})
    curve["Corrected_exempt_fatality"] = f / (f + nf / ratios)
    curve["Covered_observed_fatality"] = ref

    # Calendar-period descriptive consistency check.
    periods = [
        ("July 2019–December 2021", pd.Timestamp("2019-07-01"), pd.Timestamp("2021-12-31"), 30),
        ("2022–2023", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"), 24),
        ("January 2024–June 2025", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30"), 18),
    ]
    period_rows = []
    for label, start, end, months in periods:
        day = primary_frame["date"].dt.normalize()  # whole end date is inclusive
        q = primary_frame[(day >= start) & (day <= end)]
        for wf, wlabel in [("19인 이하", "<20"), ("500인 이상", "500+")]:
            g = q[q["작업자수"] == wf]
            period_rows.append({
                "Period": label,
                "Workforce": wlabel,
                "Months": months,
                "Accidents": int(len(g)),
                "Deaths": int(g["fatal"].sum()),
                "Accidents_per_month": len(g) / months,
                "Fatality_percent": 100 * g["fatal"].mean(),
            })
    period = pd.DataFrame(period_rows)
    return table, curve, period


def sample_flow(df: pd.DataFrame, primary_frame: pd.DataFrame, primary_fe: pd.DataFrame, singleton: pd.DataFrame) -> pd.DataFrame:
    complete = df[df["workforce_complete"] & df["plan_complete"]]
    disease_heat = complete[complete["인적사고종류(대분류)"].isin(["질병", "온열질환"])]
    no_class = complete[complete["인적사고종류(대분류)"].isin(["없음", "미입력", "분류불능"])]
    zero_only = complete[(~complete["casualty"]) & complete["acute_type"]]
    return pd.DataFrame([
        {"Stage": "Construction incidents in source register", "n": len(df)},
        {"Stage": "Workforce or plan status missing", "n": int((~(df["workforce_complete"] & df["plan_complete"])).sum())},
        {"Stage": "Complete workforce and plan status", "n": len(complete)},
        {"Stage": "Disease or heat-illness records excluded", "n": len(disease_heat)},
        {"Stage": "No, missing, or unclassifiable injury type excluded", "n": len(no_class)},
        {"Stage": "Property-only incidents without a casualty excluded", "n": len(zero_only)},
        {"Stage": "Acute-traumatic analytic frame", "n": len(primary_frame)},
        {"Stage": "Singleton event configurations excluded", "n": len(singleton)},
        {"Stage": "Recurring-configuration estimation sample", "n": len(primary_fe)},
    ])


def sample_comparison(primary_fe: pd.DataFrame, singleton: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, d in [("Recurring event configurations", primary_fe), ("Singleton event configurations", singleton)]:
        rows.append({
            "Sample": label,
            "Incidents": len(d),
            "Fatal_percent": 100 * d["fatal"].mean(),
            "Falls_percent": 100 * (d["인적사고종류(대분류)"] == "떨어짐").mean(),
            "Under_50_percent": 100 * d["small50"].mean(),
            "Exempt_percent": 100 * d["exempt"].mean(),
        })
    return pd.DataFrame(rows)


def raw_four_cell(primary_frame: pd.DataFrame) -> pd.DataFrame:
    d = primary_frame.copy()
    d["Scale"] = np.where(d["small50"].eq(1), "<50 workers", "≥50 workers")
    d["Plan_status"] = np.where(d["exempt"].eq(1), "Exempt", "Covered")
    return binomial_table(d, ["Scale", "Plan_status"])


def make_figures(
    out: Path,
    adjusted: pd.DataFrame,
    six_contrasts: pd.DataFrame,
    fall_height: pd.DataFrame,
    fall_comp: pd.DataFrame,
    fall_std: pd.DataFrame,
    type_rd: pd.DataFrame,
    report_curve: pd.DataFrame,
) -> None:
    # Global publication-friendly settings.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    # Figure 1: causal/estimand schematic.
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    boxes = {
        "tech": (0.25, 4.35, 2.45, 0.88, "Statutory technical\nconditions and\nproject attributes"),
        "scale": (0.25, 2.20, 2.45, 0.88, "Recorded site scale"),
        "cap": (3.75, 4.65, 2.20, 0.88, "Contractor capability\n(unobserved)"),
        "plan": (3.75, 2.95, 2.20, 0.88, "Plan coverage or\nexemption"),
        "org": (3.75, 1.10, 2.20, 0.88, "Organizational and\nresponse capacity"),
        "event": (7.25, 4.35, 2.35, 0.88, "Accident occurrence,\ntype, and severity"),
        "report": (7.25, 2.55, 2.35, 0.88, "Entry into the\nadministrative register"),
        "fatal": (7.25, 0.75, 2.35, 0.88, "Fatal outcome among\nreported incidents"),
    }
    for _, (x, y, w, h, text) in boxes.items():
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor="white", edgecolor="black", linewidth=0.9)
        ax.add_patch(patch); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8.2)
    def connect(x1, y1, x2, y2, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=10, linewidth=0.85, linestyle="--" if dashed else "-"))
    # Project and selection structure.
    connect(2.70, 4.78, 3.75, 3.40)   # technical conditions -> coverage
    connect(2.70, 4.92, 7.25, 4.78)   # technical conditions -> event
    connect(4.85, 4.65, 4.85, 3.83)   # capability -> coverage
    connect(5.95, 4.92, 7.25, 4.78)   # capability -> event
    connect(2.70, 2.64, 3.75, 1.54)   # scale -> capacity
    connect(5.95, 1.54, 7.25, 1.19)   # capacity -> fatal outcome
    connect(5.95, 3.40, 7.25, 4.55)   # coverage -> event
    connect(5.95, 3.18, 7.25, 2.99)   # coverage -> reporting
    connect(8.43, 4.35, 8.43, 3.43)   # event -> reporting
    connect(8.43, 2.55, 8.43, 1.63)   # reporting -> fatal among reported
    connect(9.25, 4.35, 9.25, 1.63)   # event severity -> fatal outcome
    ax.text(5.0, 0.05, "Conditioning on a reported incident and its event configuration can induce selection.\nThe analysis therefore estimates a conditional association, not a total causal effect.", ha="center", va="bottom", fontsize=7.7)
    fig.tight_layout(); fig.savefig(out / "Figure_1_estimand_schematic.png", bbox_inches="tight"); plt.close(fig)

    # Figure 2: adjusted probabilities and six-band contrasts.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
    x = np.arange(len(adjusted))
    vals = adjusted["Probability"].to_numpy() * 100
    yerr = np.vstack([(adjusted["Probability"] - adjusted["CI_low"]) * 100, (adjusted["CI_high"] - adjusted["Probability"]) * 100])
    axes[0].errorbar(x, vals, yerr=yerr, fmt="o", capsize=3, linewidth=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["Large\ncovered", "Large\nexempt", "Small\ncovered", "Small\nexempt"])
    axes[0].set_ylabel("Adjusted probability of a fatal outcome (%)")
    axes[0].set_title("A. Standardized probabilities")
    axes[0].axhline(0, linewidth=.6)
    c = six_contrasts.copy()
    x2 = np.arange(len(c))
    vals2 = c["Estimate_pp"].to_numpy()
    yerr2 = np.vstack([vals2 - c["CI_low_pp"].to_numpy(), c["CI_high_pp"].to_numpy() - vals2])
    axes[1].errorbar(x2, vals2, yerr=yerr2, fmt="o", capsize=3, linewidth=1)
    axes[1].axhline(0, linewidth=.8, linestyle="--")
    axes[1].set_xticks(x2); axes[1].set_xticklabels(c["Contrast"], rotation=35, ha="right")
    axes[1].set_ylabel("Exempt minus covered (percentage points)")
    axes[1].set_title("B. Within-band exemption contrasts")
    fig.tight_layout(); fig.savefig(out / "Figure_2_adjusted_results.png", bbox_inches="tight"); plt.close(fig)

    # Figure 3: fall-height decomposition, stacked vertically for portrait pages.
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.8))
    pivot = fall_comp.pivot(index="Workforce", columns="Height_label", values="Share").reindex(WORKFORCE_LABELS)
    bottom = np.zeros(len(pivot))
    for h in HEIGHT_LABELS:
        vals = pivot.get(h, pd.Series(0, index=pivot.index)).fillna(0).to_numpy() * 100
        axes[0].bar(np.arange(len(pivot)), vals, bottom=bottom, width=.75, label=h, edgecolor="black", linewidth=.2)
        bottom += vals
    axes[0].set_xticks(range(len(pivot))); axes[0].set_xticklabels(WORKFORCE_LABELS)
    axes[0].set_ylabel("Share of fall records (%)"); axes[0].set_title("A. Recorded height composition")
    axes[0].legend(ncol=3, frameon=False, loc="upper right", fontsize=7)
    h = fall_height.set_index("Height_label").reindex(HEIGHT_LABELS)
    xx = np.arange(len(h)); vv = h["Fatality"].to_numpy() * 100
    err = np.vstack([(h["Fatality"]-h["CI_low"]) * 100, (h["CI_high"]-h["Fatality"]) * 100])
    axes[1].errorbar(xx, vv, yerr=err, fmt="o", capsize=3)
    axes[1].set_xticks(xx); axes[1].set_xticklabels(HEIGHT_LABELS)
    axes[1].set_ylabel("Fatality among reported falls (%)"); axes[1].set_title("B. Fatality by recorded height")
    f = fall_std.set_index("Workforce").reindex(WORKFORCE_LABELS)
    xx = np.arange(len(f))
    axes[2].plot(xx, f["Raw_fatality"]*100, marker="o", label="Raw")
    axes[2].plot(xx, f["Height_standardized_fatality"]*100, marker="s", linestyle="--", label="Height-standardized")
    axes[2].set_xticks(xx); axes[2].set_xticklabels(WORKFORCE_LABELS)
    axes[2].set_ylabel("Fatality among reported falls (%)"); axes[2].set_title("C. Workforce profile")
    axes[2].legend(frameon=False)
    fig.subplots_adjust(hspace=.62, top=.97, bottom=.08, left=.12, right=.98)
    fig.savefig(out / "Figure_3_fall_height.png", bbox_inches="tight"); plt.close(fig)

    # Figure 4: all acute broad-type small-large risk differences.
    q = type_rd[(type_rd["Accidents"] >= 50) & np.isfinite(type_rd["Risk_difference_pp"])].copy().sort_values("Risk_difference_pp")
    y = np.arange(len(q)); vals = q["Risk_difference_pp"].to_numpy()
    xerr = np.vstack([vals - q["CI_low_pp"].to_numpy(), q["CI_high_pp"].to_numpy() - vals])
    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    ax.errorbar(vals, y, xerr=xerr, fmt="o", capsize=2, linewidth=.9)
    ax.axvline(0, linestyle="--", linewidth=.8)
    ax.set_yticks(y); ax.set_yticklabels(q["Accident_type"])
    ax.set_xlabel("Fatality difference: <50 minus ≥50 workers (percentage points)")
    ax.set_title("Broad accident types with at least 50 reported acute-traumatic incidents")
    fig.tight_layout(); fig.savefig(out / "Figure_4_accident_type_heterogeneity.png", bbox_inches="tight"); plt.close(fig)

    # Supplementary reporting-sensitivity figure.
    fig, ax = plt.subplots(figsize=(5.7, 3.6))
    ax.plot(report_curve["Relative_completeness"], report_curve["Corrected_exempt_fatality"]*100, label="Corrected fatality: exempt, <20")
    ax.plot(report_curve["Relative_completeness"], report_curve["Covered_observed_fatality"]*100, linestyle="--", label="Observed fatality: covered, <20")
    ax.set_xlabel("Relative completeness of nonfatal reporting at exempt sites")
    ax.set_ylabel("Fatality among reported/corrected incidents (%)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "Figure_S1_reporting_sensitivity.png", bbox_inches="tight"); plt.close(fig)


def write_data_dictionary(path: Path) -> None:
    rows = [
        ("record_id", "Anonymous record identifier", "Sequential identifier assigned after random reordering of rows; carries no information about the source record.", "Not used in estimation."),
        ("사고일시", "Accident date (pseudonymized day)", "YYYY-MM-DD. Year and month are exact. Time of day was removed and the day of month was replaced by a random within-month permutation, so records sharing an accident date still share one.", "Defines calendar year, calendar month, accident-date clusters, the February 2021 restriction, and calendar periods."),
        ("공공민간구분", "Public/private client", "Categorical.", "Categorical project control."),
        ("시설물 대분류", "Broad facility category", "Categorical.", "Missing-field composition (building works)."),
        ("시설물 중분류", "Facility middle category", "Categorical.", "Additional project-composition control."),
        ("연면적", "Gross floor area (m2)", "Rounded to two significant figures; non-numeric codes indicate not entered.", "Rank correlation with the workforce field."),
        ("지상층수", "Above-ground storeys", "Count; non-numeric codes indicate not entered.", "Partial label audit (ten or more storeys); additional project-composition control."),
        ("지하층수", "Basement floors", "Count; non-numeric codes indicate not entered.", "Proxy for excavation depth in the partial label audit; additional project-composition control."),
        ("공사종류", "Construction type", "Hierarchical label (field/facility/use).", "Identifies railway work in the partial label audit."),
        ("인적사고종류(대분류)", "Broad accident type", "Categorical.", "Defines the acute traumatic analytic frame and broad-type descriptions."),
        ("인적사고종류", "Detailed accident type", "Categorical; fall categories carry broad height bands.", "Part of the event configuration; fall-height analysis."),
        ("공종(대분류)", "Broad work category", "Categorical.", "Additional project-composition control."),
        ("사고객체(대분류)", "Broad accident object", "Categorical.", "Fixed-effect granularity sensitivity."),
        ("사고객체(소분류)", "Detailed accident object", "Categorical.", "Part of the event configuration; label audit (specified machinery)."),
        ("작업프로세스", "Work process", "Categorical.", "Part of the event configuration."),
        ("사망자", "Number of deaths", "Count.", "Fatal outcome equals one when this count exceeds zero."),
        ("부상자", "Number of injuries", "Count.", "With deaths, excludes property-only incidents."),
        ("공사비", "Contract-value band (KRW)", "Ordered categorical.", "Categorical project control; alternative scale measure."),
        ("낙찰율", "Award-rate band", "Ordered categorical.", "Additional sensitivity control."),
        ("공정율", "Construction progress band", "Ordered categorical.", "Categorical project control; workforce-field progress profile."),
        ("작업자수", "Recorded workforce band", "Ordered categorical (six bands plus not entered).", "Effect modifier. The system documentation does not define the reference time of the count."),
        ("안전관리계획", "Safety-management-plan status", "Covered (first/second-class facility), covered (other statutory ground), exempt, or not entered.", "Coverage indicator; the two covered categories are combined in the primary coding."),
        ("설계안정성검토", "Design-safety-review status", "Categorical.", "Project composition; additional control."),
    ]
    pd.DataFrame(rows, columns=["Variable", "English_label", "Coding", "Use_in_analysis"]).to_csv(path, index=False, encoding="utf-8-sig")


def save_csv(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / name, index=False, encoding="utf-8-sig")


def stage_descriptive(data_path: Path, out: Path) -> None:
    df = prepare_data(data_path)
    complete_frame = df[df["workforce_complete"] & df["plan_complete"]].copy()
    primary_frame = df[df["primary_eligible"]].copy()
    primary_fe, singleton = retain_recurring(primary_frame, "event_cell")
    save_csv(sample_flow(df, primary_frame, primary_fe, singleton), out, "Table_1_sample_flow.csv")
    save_csv(sample_comparison(primary_fe, singleton), out, "Table_1b_recurring_vs_singleton.csv")
    save_csv(raw_four_cell(primary_frame), out, "Table_2_raw_four_cells.csv")
    audit, _ = label_audit(complete_frame)
    save_csv(audit, out, "Table_S2_label_audit.csv")
    all_types, type_rd, type_omnibus = all_type_tables(df, primary_frame)
    save_csv(all_types, out, "Table_S9_all_broad_types.csv")
    save_csv(type_rd, out, "Table_S10_type_risk_differences.csv")
    fall_height, fall_comp, fall_std = fall_height_analysis(df)
    save_csv(fall_height, out, "Table_S11_fall_height_fatality.csv")
    save_csv(fall_comp, out, "Table_S12_fall_height_composition.csv")
    save_csv(fall_std, out, "Table_S13_fall_height_standardization.csv")
    report_table, report_curve, period = reporting_sensitivity(primary_frame)
    save_csv(report_table, out, "Table_S14_reporting_sensitivity.csv")
    save_csv(report_curve, out, "Table_S15_reporting_curve.csv")
    save_csv(period, out, "Table_S16_period_descriptive_check.csv")
    with open(out / "type_omnibus.json", "w", encoding="utf-8") as f:
        json.dump(type_omnibus, f, ensure_ascii=False, indent=2)
    write_data_dictionary(out / "data_dictionary.csv")


def stage_main_models(data_path: Path, out: Path) -> None:
    df = prepare_data(data_path)
    primary_frame = df[df["primary_eligible"]].copy()
    primary_fe, _ = retain_recurring(primary_frame, "event_cell")
    ladder, primary_fit = make_model_ladder(primary_frame, primary_fe)
    save_csv(ladder, out, "Table_3_model_ladder.csv")
    contrasts = pd.DataFrame([
        linear_contrast(primary_fit, {"exempt": 1.0}, "Large: exempt - covered"),
        linear_contrast(primary_fit, {"exempt": 1.0, "interaction": 1.0}, "Small: exempt - covered"),
        linear_contrast(primary_fit, {"small": 1.0}, "Covered: small - large"),
        linear_contrast(primary_fit, {"small": 1.0, "exempt": 1.0, "interaction": 1.0}, "Small exempt - large covered"),
        linear_contrast(primary_fit, {"interaction": 1.0}, "Difference in exemption contrasts (interaction)"),
    ])
    save_csv(contrasts, out, "Table_2c_adjusted_contrasts.csv")
    contract_fit, _ = alternative_contract_fit(primary_frame)
    contract_result = pd.DataFrame([
        coef_row(contract_fit, "small", "Small contract value main term"),
        coef_row(contract_fit, "exempt", "Exempt main term"),
        coef_row(contract_fit, "interaction", "Small contract value x exempt"),
    ])
    save_csv(contract_result, out, "Table_S1_alternative_contract_scale.csv")


def stage_bootstrap(data_path: Path, out: Path) -> None:
    df = prepare_data(data_path)
    primary_frame = df[df["primary_eligible"]].copy()
    primary_fe, _ = retain_recurring(primary_frame, "event_cell")
    _, primary_fit = make_model_ladder(primary_frame, primary_fe)
    adjusted = bootstrap_standardized_probabilities(
        primary_fit,
        [
            ("Large covered", {"small": 0.0, "exempt": 0.0}),
            ("Large exempt", {"small": 0.0, "exempt": 1.0}),
            ("Small covered", {"small": 1.0, "exempt": 0.0}),
            ("Small exempt", {"small": 1.0, "exempt": 1.0}),
        ],
        reps=500,
        seed=20260828,
    )
    save_csv(adjusted, out, "Table_2b_adjusted_probabilities.csv")


def stage_robustness(data_path: Path, out: Path) -> None:
    df = prepare_data(data_path)
    complete_frame = df[df["workforce_complete"] & df["plan_complete"]].copy()
    primary_frame = df[df["primary_eligible"]].copy()
    primary_fe, _ = retain_recurring(primary_frame, "event_cell")
    _, primary_fit = make_model_ladder(primary_frame, primary_fe)
    _, candidate = label_audit(complete_frame)
    save_csv(same_spec_robustness(df, primary_frame, primary_fit, candidate), out, "Table_4_robustness_same_specification.csv")
    save_csv(threshold_sensitivity(primary_fe), out, "Table_S3_threshold_sensitivity.csv")
    save_csv(cluster_sensitivity(primary_fe), out, "Table_S6_cluster_sensitivity.csv")


def stage_fe_ladder(data_path: Path, out: Path) -> None:
    df = prepare_data(data_path)
    primary_frame = df[df["primary_eligible"]].copy()
    save_csv(fe_granularity_ladder(primary_frame), out, "Table_S4_fixed_effect_granularity.csv")


def stage_fe_part(data_path: Path, out: Path, level_index: int) -> None:
    df = prepare_data(data_path)
    primary_frame = df[df["primary_eligible"]].copy()
    save_csv(
        fe_granularity_one(primary_frame, level_index),
        out,
        f"Table_S4_fixed_effect_granularity_part{level_index}.csv",
    )


def stage_fe_finalize(out: Path) -> None:
    parts = [
        pd.read_csv(out / f"Table_S4_fixed_effect_granularity_part{i}.csv", encoding="utf-8-sig")
        for i in range(len(FE_LEVELS))
    ]
    save_csv(pd.concat(parts, ignore_index=True), out, "Table_S4_fixed_effect_granularity.csv")
    for i in range(len(FE_LEVELS)):
        (out / f"Table_S4_fixed_effect_granularity_part{i}.csv").unlink(missing_ok=True)


def stage_pipeline(data_path: Path, out: Path) -> None:
    """Run all OLS/descriptive stages after reading the source data once."""
    df = prepare_data(data_path)
    complete_frame = df[df["workforce_complete"] & df["plan_complete"]].copy()
    primary_frame = df[df["primary_eligible"]].copy()
    primary_fe, singleton = retain_recurring(primary_frame, "event_cell")

    # Descriptive tables and measurement checks.
    save_csv(sample_flow(df, primary_frame, primary_fe, singleton), out, "Table_1_sample_flow.csv")
    save_csv(sample_comparison(primary_fe, singleton), out, "Table_1b_recurring_vs_singleton.csv")
    save_csv(raw_four_cell(primary_frame), out, "Table_2_raw_four_cells.csv")
    audit, candidate = label_audit(complete_frame)
    save_csv(audit, out, "Table_S2_label_audit.csv")
    all_types, type_rd, type_omnibus = all_type_tables(df, primary_frame)
    save_csv(all_types, out, "Table_S9_all_broad_types.csv")
    save_csv(type_rd, out, "Table_S10_type_risk_differences.csv")
    fall_height, fall_comp, fall_std = fall_height_analysis(df)
    save_csv(fall_height, out, "Table_S11_fall_height_fatality.csv")
    save_csv(fall_comp, out, "Table_S12_fall_height_composition.csv")
    save_csv(fall_std, out, "Table_S13_fall_height_standardization.csv")
    report_table, report_curve, period = reporting_sensitivity(primary_frame)
    save_csv(report_table, out, "Table_S14_reporting_sensitivity.csv")
    save_csv(report_curve, out, "Table_S15_reporting_curve.csv")
    save_csv(period, out, "Table_S16_period_descriptive_check.csv")
    with open(out / "type_omnibus.json", "w", encoding="utf-8") as f:
        json.dump(type_omnibus, f, ensure_ascii=False, indent=2)
    write_data_dictionary(out / "data_dictionary.csv")

    # Core models, linear contrasts, and alternative site-scale measure.
    ladder, primary_fit = make_model_ladder(primary_frame, primary_fe)
    save_csv(ladder, out, "Table_3_model_ladder.csv")
    contrasts = pd.DataFrame([
        linear_contrast(primary_fit, {"exempt": 1.0}, "Large: exempt - covered"),
        linear_contrast(primary_fit, {"exempt": 1.0, "interaction": 1.0}, "Small: exempt - covered"),
        linear_contrast(primary_fit, {"small": 1.0}, "Covered: small - large"),
        linear_contrast(primary_fit, {"small": 1.0, "exempt": 1.0, "interaction": 1.0}, "Small exempt - large covered"),
        linear_contrast(primary_fit, {"interaction": 1.0}, "Difference in exemption contrasts (interaction)"),
    ])
    save_csv(contrasts, out, "Table_2c_adjusted_contrasts.csv")
    contract_fit, _ = alternative_contract_fit(primary_frame)
    save_csv(pd.DataFrame([
        coef_row(contract_fit, "small", "Small contract value main term"),
        coef_row(contract_fit, "exempt", "Exempt main term"),
        coef_row(contract_fit, "interaction", "Small contract value x exempt"),
    ]), out, "Table_S1_alternative_contract_scale.csv")

    adjusted = bootstrap_standardized_probabilities(
        primary_fit,
        [
            ("Large covered", {"small": 0.0, "exempt": 0.0}),
            ("Large exempt", {"small": 0.0, "exempt": 1.0}),
            ("Small covered", {"small": 1.0, "exempt": 0.0}),
            ("Small exempt", {"small": 1.0, "exempt": 1.0}),
        ],
        reps=500, seed=20260828,
    )
    save_csv(adjusted, out, "Table_2b_adjusted_probabilities.csv")

    # Same-specification sensitivity analyses.
    save_csv(same_spec_robustness(df, primary_frame, primary_fit, candidate), out, "Table_4_robustness_same_specification.csv")
    save_csv(threshold_sensitivity(primary_fe), out, "Table_S3_threshold_sensitivity.csv")
    save_csv(cluster_sensitivity(primary_fe), out, "Table_S6_cluster_sensitivity.csv")
    save_csv(fe_granularity_ladder(primary_frame), out, "Table_S4_fixed_effect_granularity.csv")
    six_contrasts, six_predictions, six_meta = six_band_analysis(primary_fe)
    save_csv(six_contrasts, out, "Table_S7_six_band_contrasts.csv")
    save_csv(six_predictions, out, "Table_S8_six_band_predictions.csv")
    with open(out / "six_band_omnibus.json", "w", encoding="utf-8") as f:
        json.dump(six_meta, f, ensure_ascii=False, indent=2)


def stage_conditional(data_path: Path, out: Path) -> None:
    df = prepare_data(data_path)
    primary_frame = df[df["primary_eligible"]].copy()
    primary_fe, _ = retain_recurring(primary_frame, "event_cell")
    clogit, meta = conditional_logit(primary_fe)
    save_csv(clogit, out, "Table_S5_conditional_logit.csv")
    with open(out / "conditional_logit_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def stage_six_band(data_path: Path, out: Path) -> None:
    df = prepare_data(data_path)
    primary_frame = df[df["primary_eligible"]].copy()
    primary_fe, _ = retain_recurring(primary_frame, "event_cell")
    contrasts, predictions, omnibus = six_band_analysis(primary_fe)
    save_csv(contrasts, out, "Table_S7_six_band_contrasts.csv")
    save_csv(predictions, out, "Table_S8_six_band_predictions.csv")
    with open(out / "six_band_omnibus.json", "w", encoding="utf-8") as f:
        json.dump(omnibus, f, ensure_ascii=False, indent=2)


def stage_finalize(out: Path) -> None:
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(out / name, encoding="utf-8-sig")
    adjusted = read("Table_2b_adjusted_probabilities.csv")
    six_contrasts = read("Table_S7_six_band_contrasts.csv")
    fall_height = read("Table_S11_fall_height_fatality.csv")
    fall_comp = read("Table_S12_fall_height_composition.csv")
    fall_std = read("Table_S13_fall_height_standardization.csv")
    type_rd = read("Table_S10_type_risk_differences.csv")
    report_curve = read("Table_S15_reporting_curve.csv")
    make_figures(out, adjusted, six_contrasts, fall_height, fall_comp, fall_std, type_rd, report_curve)

    ladder = read("Table_3_model_ladder.csv")
    primary = ladder.loc[ladder["Specification"].str.startswith("6 ")].iloc[0]
    contrasts = read("Table_2c_adjusted_contrasts.csv")
    contract = read("Table_S1_alternative_contract_scale.csv")
    with open(out / "conditional_logit_meta.json", encoding="utf-8") as f:
        clogit_meta = json.load(f)
    with open(out / "six_band_omnibus.json", encoding="utf-8") as f:
        six_meta = json.load(f)
    with open(out / "type_omnibus.json", encoding="utf-8") as f:
        type_meta = json.load(f)
    flow = read("Table_1_sample_flow.csv")
    source_n = int(flow.loc[flow["Stage"] == "Construction incidents in source register", "n"].iloc[0])
    frame_n = int(flow.loc[flow["Stage"] == "Acute-traumatic analytic frame", "n"].iloc[0])
    fe_n = int(flow.loc[flow["Stage"] == "Recurring-configuration estimation sample", "n"].iloc[0])
    b = float(primary["Interaction_pp"])
    lo = float(primary["Interaction_low"])
    hi = float(primary["Interaction_high"])
    assert source_n == 37196
    assert frame_n == 32700
    assert fe_n == 28821
    assert abs(b - 1.19796) < 0.0005
    assert abs(lo - 0.24278) < 0.0005
    assert abs(hi - 2.15314) < 0.0005
    summary = {
        "source_records": source_n,
        "primary_analytic_frame": frame_n,
        "primary_estimation_n": fe_n,
        "primary_event_cells": int(read("Table_S4_fixed_effect_granularity.csv").iloc[-1]["Groups"]),
        "primary_interaction_pp": b,
        "primary_interaction_ci_pp": [lo, hi],
        "primary_interaction_p": float(primary["Interaction_p"]),
        "small_simple_contrast": contrasts.iloc[1].to_dict(),
        "small_exempt_vs_large_covered": contrasts.iloc[3].to_dict(),
        "contract_scale_interaction": contract.iloc[2].to_dict(),
        "six_band_omnibus": six_meta,
        "type_omnibus": type_meta,
        "conditional_logit": clogit_meta,
        "bootstrap_reps": int(adjusted["Bootstrap_reps"].iloc[0]),
        "bootstrap_seed": int(adjusted["Bootstrap_seed"].iloc[0]),
    }
    with open(out / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def orchestrate(script: Path, data_path: Path, out: Path) -> None:
    import os
    import subprocess
    stages = ["pipeline", "descriptive", "main", "bootstrap", "robustness", "fe", "conditional", "six", "finalize"]
    env = os.environ.copy()
    # Limiting BLAS threads makes the run predictable on machines with many cores.
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    for stage in stages:
        cmd = [sys.executable, str(script), "--stage", stage, "--data", str(data_path), "--output", str(out)]
        subprocess.run(cmd, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("CSI_analytic_deidentified.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--stage",
        choices=[
            "pipeline", "descriptive", "main", "bootstrap", "robustness", "fe",
            "fe0", "fe1", "fe2", "fe3", "fe4", "fe_finalize",
            "conditional", "six", "finalize"
        ],
        default="main",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "pipeline":
        stage_pipeline(args.data, args.output)
    elif args.stage == "descriptive":
        stage_descriptive(args.data, args.output)
    elif args.stage == "main":
        stage_main_models(args.data, args.output)
    elif args.stage == "bootstrap":
        stage_bootstrap(args.data, args.output)
    elif args.stage == "robustness":
        stage_robustness(args.data, args.output)
    elif args.stage == "fe":
        stage_fe_ladder(args.data, args.output)
    elif args.stage.startswith("fe") and args.stage[2:].isdigit():
        stage_fe_part(args.data, args.output, int(args.stage[2:]))
    elif args.stage == "fe_finalize":
        stage_fe_finalize(args.output)
    elif args.stage == "conditional":
        stage_conditional(args.data, args.output)
    elif args.stage == "six":
        stage_six_band(args.data, args.output)
    elif args.stage == "finalize":
        stage_finalize(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
