"""Established computational routines for the safety-planning coverage study.

The submission-level inferential hierarchy is implemented in
``analysis_rehierarchy.py``. That workflow treats the within-band direct-support
model as the primary comparative model and the pooled event-configuration model as a
complementary detection specification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

import analysis_core as core

WORKFORCE_ORDER = core.WORKFORCE_ORDER
WORKFORCE_LABELS = core.WORKFORCE_LABELS
PRIMARY_CONTROLS = ["공정율", "공사비", "공공민간구분", "year"]


def save_csv(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / name, index=False, encoding="utf-8-sig")


def ci(beta: float, se: float) -> tuple[float, float]:
    return beta - 1.959963984540054 * se, beta + 1.959963984540054 * se


def holm_adjust(pvalues: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def primary_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = core.prepare_data(path)
    frame = df[df["primary_eligible"]].copy()
    recurring, singleton = core.retain_recurring(frame, "event_cell")
    recurring["workforce_score"] = recurring["작업자수"].map({x: i for i, x in enumerate(WORKFORCE_ORDER)}).astype(float)
    return df, frame, recurring, singleton


def binary_fit(data: pd.DataFrame, controls: list[str] | None = None, small_col: str = "small50") -> core.FitResult:
    d = data.copy()
    d["small"] = d[small_col].astype(float)
    d["interaction"] = d["small"] * d["exempt"]
    return core.fit_ols(
        d,
        ["small", "exempt", "interaction"],
        controls if controls is not None else PRIMARY_CONTROLS,
        fe_col="event_cell",
        cluster_col="event_cell",
    )


def six_band_fit(data: pd.DataFrame, controls: list[str] | None = None) -> tuple[pd.DataFrame, core.FitResult]:
    d = data.copy()
    main = ["exempt"]
    for category, label in zip(WORKFORCE_ORDER[:-1], WORKFORCE_LABELS[:-1]):
        c = f"wf_{label}"
        i = f"int_{label}"
        d[c] = (d["작업자수"] == category).astype(float)
        d[i] = d[c] * d["exempt"]
        main.extend([c, i])
    fit = core.fit_ols(
        d,
        main,
        controls if controls is not None else PRIMARY_CONTROLS,
        fe_col="event_cell",
        cluster_col="event_cell",
    )
    return d, fit


def six_band_primary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, core.FitResult]:
    d, fit = six_band_fit(data)
    counts = (
        d.groupby(["작업자수", "exempt"], observed=True)["fatal"]
        .agg(Incidents="size", Deaths="sum", Raw_fatality="mean")
        .reset_index()
    )
    count_lookup = {(r["작업자수"], int(r["exempt"])): r for _, r in counts.iterrows()}

    rows = []
    for category, label in zip(WORKFORCE_ORDER, WORKFORCE_LABELS):
        weights = {"exempt": 1.0}
        if label != "500+":
            weights[f"int_{label}"] = 1.0
        row = core.linear_contrast(fit, weights, label)
        c = count_lookup.get((category, 0), {})
        e = count_lookup.get((category, 1), {})
        row.update({
            "Covered_incidents": int(c.get("Incidents", 0)),
            "Covered_deaths": int(c.get("Deaths", 0)),
            "Covered_raw_fatality_percent": 100 * float(c.get("Raw_fatality", np.nan)),
            "Exempt_incidents": int(e.get("Incidents", 0)),
            "Exempt_deaths": int(e.get("Deaths", 0)),
            "Exempt_raw_fatality_percent": 100 * float(e.get("Raw_fatality", np.nan)),
        })
        rows.append(row)
    contrasts = pd.DataFrame(rows)
    contrasts["Holm_p"] = holm_adjust(contrasts["p"])

    terms = [f"int_{x}" for x in WORKFORCE_LABELS[:-1]]
    params = list(fit.result.params.index)
    R = np.zeros((len(terms), len(params)))
    for i, term in enumerate(terms):
        R[i, params.index(term)] = 1.0
    test = fit.result.wald_test(R, scalar=True)

    trend_data = data.copy()
    trend_data["workforce_score"] = trend_data["작업자수"].map({x: i for i, x in enumerate(WORKFORCE_ORDER)}).astype(float)
    trend_data["score_x_exempt"] = trend_data["workforce_score"] * trend_data["exempt"]
    trend_fit = core.fit_ols(
        trend_data,
        ["workforce_score", "exempt", "score_x_exempt"],
        PRIMARY_CONTROLS,
        fe_col="event_cell",
        cluster_col="event_cell",
    )
    trend = pd.DataFrame([
        core.coef_row(trend_fit, "score_x_exempt", "Ordinal band score × exemption")
    ])
    meta = {
        "wald_chi2": float(test.statistic),
        "df": 5,
        "p": float(test.pvalue),
        "reference_band": "500+",
        "trend_interpretation": "Average change in the exemption contrast for a one-category increase in the ordered workforce score; category spacing is not assumed to represent equal worker increments.",
    }
    return contrasts, trend, meta, fit



def within_band_support(data: pd.DataFrame) -> pd.DataFrame:
    """Count event configurations that directly compare covered and exempt records within each workforce band."""
    d = data.copy()
    g = (
        d.groupby(["작업자수", "event_cell"], observed=True)
        .agg(Incidents=("fatal", "size"), Plan_states=("exempt", "nunique"))
        .reset_index()
    )
    rows = []
    for category, label in zip(WORKFORCE_ORDER, WORKFORCE_LABELS):
        q = g[(g["작업자수"] == category) & (g["Plan_states"] == 2)]
        rows.append({
            "Workforce_band": label,
            "Mixed_event_configurations": int(len(q)),
            "Incidents_in_mixed_configurations": int(q["Incidents"].sum()),
        })
    return pd.DataFrame(rows)


def strict_six_band(
    data: pd.DataFrame,
    restrict_to_mixed_strata: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, core.FitResult]:
    """Support-aligned six-band model with event-configuration-by-band fixed effects.

    This specification allows the baseline fatality of a given event configuration
    to differ across workforce bands.  The exempt-versus-covered contrast is thus
    identified only from event configurations in which both plan states are
    observed within the same workforce band.  Standard errors remain clustered by
    the original event configuration.
    """
    d = data.copy()
    d["event_cell_band"] = d["event_cell"].astype(str) + " || " + d["작업자수"].astype(str)
    if restrict_to_mixed_strata:
        plan_states = d.groupby("event_cell_band", observed=True)["exempt"].transform("nunique")
        d = d[plan_states == 2].copy()

    main = ["exempt"]
    for category, label in zip(WORKFORCE_ORDER[:-1], WORKFORCE_LABELS[:-1]):
        c = f"wf_{label}"
        i = f"int_{label}"
        d[c] = (d["작업자수"] == category).astype(float)
        d[i] = d[c] * d["exempt"]
        main.extend([c, i])
    fit = core.fit_ols(
        d,
        main,
        PRIMARY_CONTROLS,
        fe_col="event_cell_band",
        cluster_col="event_cell",
    )

    counts = (
        d.groupby(["작업자수", "exempt"], observed=True)["fatal"]
        .agg(Incidents="size", Deaths="sum", Raw_fatality="mean")
        .reset_index()
    )
    count_lookup = {(r["작업자수"], int(r["exempt"])): r for _, r in counts.iterrows()}
    rows = []
    for category, label in zip(WORKFORCE_ORDER, WORKFORCE_LABELS):
        weights = {"exempt": 1.0}
        if label != "500+":
            weights[f"int_{label}"] = 1.0
        row = core.linear_contrast(fit, weights, label)
        c = count_lookup.get((category, 0), {})
        e = count_lookup.get((category, 1), {})
        row.update({
            "Covered_incidents": int(c.get("Incidents", 0)),
            "Covered_deaths": int(c.get("Deaths", 0)),
            "Exempt_incidents": int(e.get("Incidents", 0)),
            "Exempt_deaths": int(e.get("Deaths", 0)),
        })
        rows.append(row)
    contrasts = pd.DataFrame(rows)

    terms = [f"int_{x}" for x in WORKFORCE_LABELS[:-1]]
    params = list(fit.result.params.index)
    R = np.zeros((len(terms), len(params)))
    for i, term in enumerate(terms):
        R[i, params.index(term)] = 1.0
    test = fit.result.wald_test(R, scalar=True)

    trend_data = d.copy()
    score_map = {x: i for i, x in enumerate(WORKFORCE_ORDER)}
    trend_data["workforce_score"] = trend_data["작업자수"].map(score_map).astype(float)
    trend_data["score_x_exempt"] = trend_data["workforce_score"] * trend_data["exempt"]
    trend_fit = core.fit_ols(
        trend_data,
        ["workforce_score", "exempt", "score_x_exempt"],
        PRIMARY_CONTROLS,
        fe_col="event_cell_band",
        cluster_col="event_cell",
    )
    trend = pd.DataFrame([
        core.coef_row(trend_fit, "score_x_exempt", "Ordinal band score × exemption")
    ])
    meta = {
        "specification": "event configuration × workforce band fixed effects",
        "restricted_to_mixed_strata": bool(restrict_to_mixed_strata),
        "n": int(len(d)),
        "fixed_effect_strata": int(d["event_cell_band"].nunique()),
        "mixed_fixed_effect_strata": int((d.groupby("event_cell_band")["exempt"].nunique() == 2).sum()),
        "wald_chi2": float(test.statistic),
        "df": 5,
        "p": float(test.pvalue),
        "trend_pp_per_band": float(trend.iloc[0]["Coefficient_pp"]),
        "trend_ci_pp": [float(trend.iloc[0]["CI_low_pp"]), float(trend.iloc[0]["CI_high_pp"])],
        "trend_p": float(trend.iloc[0]["p"]),
    }
    return contrasts, trend, meta, fit

def model_ladder(frame: pd.DataFrame, recurring: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, d, fe, controls in [
        ("1 Full acute-traumatic frame; no fixed effect or controls", frame, None, []),
        ("2 Recurring-configuration sample; no fixed effect or controls", recurring, None, []),
        ("3 Adds event-configuration fixed effects", recurring, "event_cell", []),
        ("4 Adds progress rate", recurring, "event_cell", ["공정율"]),
        ("5 Adds contract value", recurring, "event_cell", ["공정율", "공사비"]),
        ("6 Adds public/private project status", recurring, "event_cell", ["공정율", "공사비", "공공민간구분"]),
        ("7 Adds calendar year (binary sensitivity)", recurring, "event_cell", PRIMARY_CONTROLS),
    ]:
        q = d.copy()
        q["small"] = q["small50"].astype(float)
        q["interaction"] = q["small"] * q["exempt"]
        fit = core.fit_ols(q, ["small", "exempt", "interaction"], controls, fe_col=fe, cluster_col="event_cell")
        r = {"Specification": label, "n": int(fit.result.nobs)}
        for term, prefix in [("small", "Small"), ("exempt", "Exempt"), ("interaction", "Interaction")]:
            b = float(fit.result.params[term]) * 100
            se = float(fit.result.bse[term]) * 100
            lo, hi = ci(b, se)
            r.update({f"{prefix}_pp": b, f"{prefix}_low": lo, f"{prefix}_high": hi, f"{prefix}_p": float(fit.result.pvalues[term])})
        rows.append(r)
    return pd.DataFrame(rows)


def support_diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = data.copy()
    d["combo"] = d["small50"].astype(str) + "_" + d["exempt"].astype(str)
    per_cell = d.groupby("event_cell").agg(
        Incidents=("fatal", "size"),
        Observed_combinations=("combo", "nunique"),
    )
    within_scale = d.groupby(["event_cell", "small50"])["exempt"].nunique().unstack(fill_value=0)
    rows = [
        {"Support_definition": "All recurring event configurations", "Event_configurations": int(len(per_cell)), "Incidents": int(len(d))},
        {"Support_definition": "All four scale × plan combinations observed", "Event_configurations": int((per_cell["Observed_combinations"] == 4).sum()), "Incidents": int(per_cell.loc[per_cell["Observed_combinations"] == 4, "Incidents"].sum())},
        {"Support_definition": "At least three of four combinations observed", "Event_configurations": int((per_cell["Observed_combinations"] >= 3).sum()), "Incidents": int(per_cell.loc[per_cell["Observed_combinations"] >= 3, "Incidents"].sum())},
        {"Support_definition": "Covered and exempt both observed within ≥50 band", "Event_configurations": int((within_scale.get(0, 0) == 2).sum()), "Incidents": np.nan},
        {"Support_definition": "Covered and exempt both observed within <50 band", "Event_configurations": int((within_scale.get(1, 0) == 2).sum()), "Incidents": np.nan},
    ]
    support = pd.DataFrame(rows)

    est_rows = []
    full = binary_fit(d)
    est_rows.append(core.coef_row(full, "interaction", "All recurring configurations"))
    for minimum, label in [(3, "At least three combinations"), (4, "All four combinations")]:
        keep = per_cell.index[per_cell["Observed_combinations"] >= minimum]
        q = d[d["event_cell"].isin(keep)].copy()
        fit = binary_fit(q)
        est_rows.append(core.coef_row(fit, "interaction", label))
    return support, pd.DataFrame(est_rows)


def threshold_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [("small20", "<20 workers"), ("small50", "<50 workers"), ("small100", "<100 workers")]:
        fit = binary_fit(data, small_col=col)
        rows.append(core.coef_row(fit, "interaction", label))
    out = pd.DataFrame(rows)
    out["Holm_p"] = holm_adjust(out["p"])
    return out


def leave_band_out(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_map = {x: i for i, x in enumerate(WORKFORCE_ORDER)}
    for category, label in zip(WORKFORCE_ORDER, WORKFORCE_LABELS):
        q = data[data["작업자수"] != category].copy()
        bfit = binary_fit(q)
        brow = core.coef_row(bfit, "interaction", f"Exclude {label}")
        q["workforce_score"] = q["작업자수"].map(score_map).astype(float)
        q["score_x_exempt"] = q["workforce_score"] * q["exempt"]
        tfit = core.fit_ols(q, ["workforce_score", "exempt", "score_x_exempt"], PRIMARY_CONTROLS, fe_col="event_cell", cluster_col="event_cell")
        trow = core.coef_row(tfit, "score_x_exempt", f"Exclude {label}")
        rows.append({
            "Excluded_band": label,
            "n": int(len(q)),
            "Binary_interaction_pp": brow["Coefficient_pp"],
            "Binary_low": brow["CI_low_pp"],
            "Binary_high": brow["CI_high_pp"],
            "Binary_p": brow["p"],
            "Ordinal_trend_pp_per_band": trow["Coefficient_pp"],
            "Trend_low": trow["CI_low_pp"],
            "Trend_high": trow["CI_high_pp"],
            "Trend_p": trow["p"],
        })
    return pd.DataFrame(rows)


def band_aggregation_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = data[data["작업자수"] != "500인 이상"].copy()
    row = core.coef_row(binary_fit(q), "interaction", "<50 versus 50–499 workers")
    rows.append(row)

    q = data.copy()
    q["wf4"] = q["작업자수"].replace({
        "19인 이하": "<20",
        "20~49인": "20–49",
        "50~99인": "50–299",
        "100~299인": "50–299",
        "300~499인": "300+",
        "500인 이상": "300+",
    })
    cats = ["<20", "20–49", "50–299", "300+"]
    main = ["exempt"]
    for cat in cats[:-1]:
        q[f"wf_{cat}"] = (q["wf4"] == cat).astype(float)
        q[f"int_{cat}"] = q[f"wf_{cat}"] * q["exempt"]
        main.extend([f"wf_{cat}", f"int_{cat}"])
    fit = core.fit_ols(q, main, PRIMARY_CONTROLS, fe_col="event_cell", cluster_col="event_cell")
    for cat in cats:
        weights = {"exempt": 1.0}
        if cat != "300+":
            weights[f"int_{cat}"] = 1.0
        r = core.linear_contrast(fit, weights, f"Four-band contrast: {cat}")
        rows.append({
            "Specification": r["Contrast"], "n": int(len(q)), "Coefficient_pp": r["Estimate_pp"],
            "CI_low_pp": r["CI_low_pp"], "CI_high_pp": r["CI_high_pp"], "p": r["p"],
        })
    return pd.DataFrame(rows)


def single_casualty_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    total = pd.to_numeric(data["사망자"], errors="coerce").fillna(0) + pd.to_numeric(data["부상자"], errors="coerce").fillna(0)
    q = data[total == 1].copy()
    fit = binary_fit(q)
    return pd.DataFrame([core.coef_row(fit, "interaction", "Exactly one casualty")])


def alternative_contract(data: pd.DataFrame) -> pd.DataFrame:
    q = data[data["contract_complete"]].copy()
    q["small"] = q["small_contract"].astype(float)
    q["interaction"] = q["small"] * q["exempt"]
    fit = core.fit_ols(q, ["small", "exempt", "interaction"], ["공정율", "작업자수", "공공민간구분", "year"], fe_col="event_cell", cluster_col="event_cell")
    return pd.DataFrame([
        core.coef_row(fit, "small", "Contract value < KRW 5bn"),
        core.coef_row(fit, "exempt", "Exempt"),
        core.coef_row(fit, "interaction", "Contract-value scale × exemption"),
    ])


def plan_three_category(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["small"] = d["small50"].astype(float)
    d["other_covered"] = (d["안전관리계획"] == "대상현장(1/2종외)").astype(float)
    d["exempt3"] = (d["안전관리계획"] == "비대상현장").astype(float)
    d["small_other"] = d["small"] * d["other_covered"]
    d["small_exempt3"] = d["small"] * d["exempt3"]
    fit = core.fit_ols(
        d,
        ["small", "other_covered", "exempt3", "small_other", "small_exempt3"],
        PRIMARY_CONTROLS,
        fe_col="event_cell",
        cluster_col="event_cell",
    )
    rows = [
        core.linear_contrast(fit, {"small_other": 1.0}, "Scale interaction: other statutory coverage vs first/second-class coverage"),
        core.linear_contrast(fit, {"small_exempt3": 1.0}, "Scale interaction: exempt vs first/second-class coverage"),
        core.linear_contrast(fit, {"small_exempt3": 1.0, "small_other": -1.0}, "Difference between the two scale interactions"),
        core.linear_contrast(fit, {"exempt3": 1.0, "small_exempt3": 1.0}, "Within <50: exempt vs first/second-class coverage"),
        core.linear_contrast(fit, {"exempt3": 1.0, "other_covered": -1.0, "small_exempt3": 1.0, "small_other": -1.0}, "Within <50: exempt vs other statutory coverage"),
    ]
    out = pd.DataFrame(rows)
    out.insert(1, "n", int(len(d)))
    return out


def project_balance(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["Scale"] = np.where(d["small50"] == 1, "<50", "≥50")
    d["Plan"] = np.where(d["exempt"] == 1, "Exempt", "Covered")
    d["Group"] = d["Scale"] + " " + d["Plan"]
    d["public"] = (d["공공민간구분"] == "공공").astype(float)
    d["building"] = (d["시설물 대분류"] == "건축").astype(float)
    d["design_review"] = (d["설계안정성검토"] == "대상").astype(float)
    d["large_contract"] = ((d["contract_complete"]) & (~d["공사비"].isin(core.CONTRACT_SMALL_LEVELS))).astype(float)
    d["progress_50plus"] = d["공정율"].isin(["50~59%", "60~69%", "70~79%", "80~89%", "90% 이상"]).astype(float)
    d["recent"] = d["date"].dt.year.ge(2024).astype(float)
    d["above_floors"] = pd.to_numeric(d["지상층수"].where(~d["지상층수"].eq("미입력")), errors="coerce")
    d["below_floors"] = pd.to_numeric(d["지하층수"].where(~d["지하층수"].eq("미입력")), errors="coerce")
    groups = ["<50 Covered", "<50 Exempt", "≥50 Covered", "≥50 Exempt"]
    measures = [
        ("Incidents", lambda g: len(g), "count"),
        ("Fatal incidents (%)", lambda g: 100 * g["fatal"].mean(), "percent"),
        ("Public project (%)", lambda g: 100 * g["public"].mean(), "percent"),
        ("Building facility (%)", lambda g: 100 * g["building"].mean(), "percent"),
        ("Design-safety review required (%)", lambda g: 100 * g["design_review"].mean(), "percent"),
        ("Contract value ≥ KRW 5bn (%)", lambda g: 100 * g["large_contract"].mean(), "percent"),
        ("Progress ≥50% (%)", lambda g: 100 * g["progress_50plus"].mean(), "percent"),
        ("Accident in 2024–2025 (%)", lambda g: 100 * g["recent"].mean(), "percent"),
        ("Median above-ground floors", lambda g: g["above_floors"].median(), "median"),
        ("Median basement floors", lambda g: g["below_floors"].median(), "median"),
    ]
    rows = []
    for label, func, kind in measures:
        row = {"Characteristic": label}
        for group in groups:
            row[group] = float(func(d[d["Group"] == group]))
        rows.append(row)
    return pd.DataFrame(rows)


def same_spec_robustness(df: pd.DataFrame, frame: pd.DataFrame, recurring: pd.DataFrame, candidate_all: pd.Series) -> pd.DataFrame:
    rows = [core.coef_row(binary_fit(recurring), "interaction", "Reference pooled binary sample")]

    full = df[df["workforce_complete"] & df["plan_complete"]].copy()
    full, _ = core.retain_recurring(full, "event_cell")
    rows.append(core.coef_row(binary_fit(full), "interaction", "Full administrative register"))

    no_other = frame[frame["인적사고종류(대분류)"] != "기타"].copy()
    no_other, _ = core.retain_recurring(no_other, "event_cell")
    rows.append(core.coef_row(binary_fit(no_other), "interaction", "Excludes broad 'Other'"))

    candidate = candidate_all.reindex(frame.index).fillna(False)
    cleaned = frame[~candidate].copy()
    for label, q in [
        ("Candidate label inconsistencies removed", cleaned),
        ("From February 2021", frame[frame["date"] >= pd.Timestamp("2021-02-01")].copy()),
        ("Both label cleaning and February 2021 restriction", cleaned[cleaned["date"] >= pd.Timestamp("2021-02-01")].copy()),
    ]:
        q, _ = core.retain_recurring(q, "event_cell")
        rows.append(core.coef_row(binary_fit(q), "interaction", label))

    q = recurring.copy()
    rows.append(core.coef_row(binary_fit(q, controls=PRIMARY_CONTROLS + ["낙찰율"]), "interaction", "Adds award-rate category"))
    return pd.DataFrame(rows)


def extra_controls(data: pd.DataFrame) -> pd.DataFrame:
    sets = [
        ("Fully adjusted pooled binary specification", PRIMARY_CONTROLS),
        ("Adds facility middle category", PRIMARY_CONTROLS + ["시설물 중분류"]),
        ("Adds broad work category", PRIMARY_CONTROLS + ["공종(대분류)"]),
        ("Adds design-safety-review status", PRIMARY_CONTROLS + ["설계안정성검토"]),
        ("Adds above- and below-ground floors", PRIMARY_CONTROLS + ["지상층수", "지하층수"]),
        ("Adds all observed project-composition controls", PRIMARY_CONTROLS + ["시설물 중분류", "공종(대분류)", "설계안정성검토", "지상층수", "지하층수"]),
    ]
    rows = []
    for label, controls in sets:
        rows.append(core.coef_row(binary_fit(data, controls=controls), "interaction", label))
    return pd.DataFrame(rows)


def leave_year_out(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(data["year"].unique()):
        q = data[data["year"] != year].copy()
        rows.append(core.coef_row(binary_fit(q), "interaction", f"Exclude {year}"))
    return pd.DataFrame(rows)


def conditional_logit_binary(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Conditional logit is retained as a directional sensitivity analysis.
    # It uses the same outcome-varying event strata and project controls as the
    # Calendar year is omitted here to keep the
    # conditional-likelihood problem computationally stable.
    return core.conditional_logit(data)

def bounded_logit_sensitivity(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bounded-link sensitivity estimated by deterministic logistic IRLS.

    Event components are entered as separate categorical terms because a full
    three-way event-configuration fixed effect would drop most strata in a
    nonlinear model. Cluster-robust covariance is computed by event
    configuration. This model is a link-function sensitivity, not a
    covariance-equivalent replacement for the primary within LPM.
    """
    d = data.copy()
    main = ["exempt"]
    for category, label in zip(WORKFORCE_ORDER[:-1], WORKFORCE_LABELS[:-1]):
        d[f"wf_{label}"] = (d["작업자수"] == category).astype(float)
        d[f"int_{label}"] = d[f"wf_{label}"] * d["exempt"]
        main.extend([f"wf_{label}", f"int_{label}"])
    Xdf = core.design_matrix(
        d, main, ["인적사고종류", "사고객체(소분류)", "작업프로세스"] + PRIMARY_CONTROLS
    )
    Xdf.insert(0, "const", 1.0)
    Xdf = Xdf.loc[:, Xdf.nunique() > 1]
    X = Xdf.to_numpy(dtype=float)
    y = d["fatal"].to_numpy(dtype=float)

    beta = np.zeros(X.shape[1], dtype=float)
    if "const" in Xdf.columns:
        beta[Xdf.columns.get_loc("const")] = math.log(y.mean() / (1.0 - y.mean()))
    previous_ll = -np.inf
    converged = False
    iterations = 0
    for iteration in range(1, 101):
        eta = np.clip(X @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(probability * (1.0 - probability), 1e-8)
        hessian = X.T @ (X * weight[:, None])
        score = X.T @ (y - probability)
        step = np.linalg.solve(hessian, score)
        ll0 = float(np.sum(y * np.log(np.clip(probability, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - probability, 1e-12, 1.0))))
        factor = 1.0
        while factor > 1e-6:
            candidate = beta + factor * step
            eta_candidate = np.clip(X @ candidate, -30.0, 30.0)
            p_candidate = 1.0 / (1.0 + np.exp(-eta_candidate))
            ll_candidate = float(np.sum(y * np.log(np.clip(p_candidate, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p_candidate, 1e-12, 1.0))))
            if ll_candidate >= ll0 - 1e-10:
                break
            factor *= 0.5
        beta = candidate
        iterations = iteration
        if abs(ll_candidate - previous_ll) < 1e-8:
            converged = True
            break
        previous_ll = ll_candidate

    eta = np.clip(X @ beta, -30.0, 30.0)
    probability = 1.0 / (1.0 + np.exp(-eta))
    weight = probability * (1.0 - probability)
    bread = np.linalg.inv(X.T @ (X * weight[:, None]))
    group_codes, groups = pd.factorize(d["event_cell"].astype(str), sort=True)
    scores = np.zeros((len(groups), X.shape[1]), dtype=float)
    np.add.at(scores, group_codes, X * (y - probability)[:, None])
    meat = scores.T @ scores
    n, k = X.shape
    correction = (len(groups) / (len(groups) - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread

    rows = []
    for label in WORKFORCE_LABELS:
        values, gradients = [], []
        for ex in [0.0, 1.0]:
            Xcf = Xdf.copy()
            Xcf["exempt"] = ex
            for lab in WORKFORCE_LABELS[:-1]:
                Xcf[f"wf_{lab}"] = 1.0 if label == lab else 0.0
                Xcf[f"int_{lab}"] = Xcf[f"wf_{lab}"] * ex
            M = Xcf.to_numpy(dtype=float)
            p_cf = 1.0 / (1.0 + np.exp(-np.clip(M @ beta, -30.0, 30.0)))
            values.append(float(p_cf.mean()))
            gradients.append((M * (p_cf * (1.0 - p_cf))[:, None]).mean(axis=0))
        difference = values[1] - values[0]
        gradient = gradients[1] - gradients[0]
        se = math.sqrt(max(0.0, float(gradient @ covariance @ gradient)))
        lo, hi = ci(difference, se)
        rows.append({
            "Workforce": label,
            "Covered_probability_percent": 100 * values[0],
            "Exempt_probability_percent": 100 * values[1],
            "Exempt_minus_covered_pp": 100 * difference,
            "CI_low_pp": 100 * lo,
            "CI_high_pp": 100 * hi,
            "p": float(2 * stats.norm.sf(abs(difference / se))) if se > 0 else np.nan,
        })
    meta = {
        "n": int(len(d)), "parameters": int(X.shape[1]), "converged": converged,
        "iterations": iterations, "covariance": "clustered by event configuration",
        "estimator": "deterministic logistic IRLS",
        "event_adjustment": "detailed accident type, detailed accident object, and work process entered as separate categorical terms rather than a full interaction fixed effect",
    }
    return pd.DataFrame(rows), meta

def covariance_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["small"] = d["small50"].astype(float)
    d["interaction"] = d["small"] * d["exempt"]
    rows = []
    for label, cov_type, cluster in [
        ("HC1 heteroskedasticity-robust", "HC1", None),
        ("Clustered by event configuration", "cluster", "event_cell"),
        ("Clustered by calendar month", "cluster", "month"),
        ("Clustered by accident date", "cluster", "accident_date"),
    ]:
        fit = core.fit_ols(d, ["small", "exempt", "interaction"], PRIMARY_CONTROLS, fe_col="event_cell", cluster_col=cluster, cov_type=cov_type)
        rows.append(core.coef_row(fit, "interaction", label))
    return pd.DataFrame(rows)


def prepare_weighted_within(data: pd.DataFrame) -> dict:
    """Precompute the design and event-configuration sufficient statistics.

    This makes the reporting-completeness curve fast enough to reproduce in a
    single run while retaining the same weighted within estimator and
    cluster-robust covariance used by statsmodels WLS.
    """
    d = data.copy()
    main = ["exempt"]
    for category, label in zip(WORKFORCE_ORDER[:-1], WORKFORCE_LABELS[:-1]):
        d[f"wf_{label}"] = (d["작업자수"] == category).astype(float)
        d[f"int_{label}"] = d[f"wf_{label}"] * d["exempt"]
        main.extend([f"wf_{label}", f"int_{label}"])
    Xdf = core.design_matrix(d, main, PRIMARY_CONTROLS)
    X = Xdf.to_numpy(dtype=float)
    y = d["fatal"].to_numpy(dtype=float)
    group_codes, groups = pd.factorize(d["event_cell"].astype(str), sort=True)
    group_count = len(groups)
    focal = ((d["작업자수"] == "19인 이하") & (d["exempt"] == 1) & (d["fatal"] == 0)).to_numpy()
    base_count = np.bincount(group_codes, minlength=group_count).astype(float)
    focal_count = np.bincount(group_codes, weights=focal.astype(float), minlength=group_count)
    y_sum = np.bincount(group_codes, weights=y, minlength=group_count)
    X_sum = np.zeros((group_count, X.shape[1]), dtype=float)
    X_focal_sum = np.zeros_like(X_sum)
    np.add.at(X_sum, group_codes, X)
    np.add.at(X_focal_sum, group_codes, X * focal[:, None])
    return {
        "columns": list(Xdf.columns), "X": X, "y": y, "group_codes": group_codes,
        "group_count": group_count, "focal": focal.astype(float),
        "base_count": base_count, "focal_count": focal_count,
        "y_sum": y_sum, "X_sum": X_sum, "X_focal_sum": X_focal_sum,
    }


def weighted_within_fit(prepared: dict, ratio: float) -> tuple[pd.Series, pd.DataFrame]:
    """Fit the frequency-weighted within model with cluster-robust covariance."""
    if not (0 < ratio <= 1):
        raise ValueError("ratio must be in (0, 1]")
    X = prepared["X"]
    y = prepared["y"]
    codes = prepared["group_codes"]
    focal = prepared["focal"]
    group_count = prepared["group_count"]
    alpha = 1.0 / ratio - 1.0
    weights = 1.0 + alpha * focal
    group_weights = prepared["base_count"] + alpha * prepared["focal_count"]
    y_mean = prepared["y_sum"] / group_weights
    X_mean = (prepared["X_sum"] + alpha * prepared["X_focal_sum"]) / group_weights[:, None]
    y_dm = y - y_mean[codes]
    X_dm = X - X_mean[codes]

    xtwx = X_dm.T @ (X_dm * weights[:, None])
    xtwy = X_dm.T @ (weights * y_dm)
    beta = np.linalg.solve(xtwx, xtwy)
    residual = y_dm - X_dm @ beta

    scores = np.zeros((group_count, X.shape[1]), dtype=float)
    np.add.at(scores, codes, X_dm * (weights * residual)[:, None])
    meat = scores.T @ scores
    bread = np.linalg.inv(xtwx)
    n, k = X.shape
    correction = (group_count / (group_count - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    columns = prepared["columns"]
    return pd.Series(beta, index=columns), pd.DataFrame(covariance, index=columns, columns=columns)

def contrast_from_params(params: pd.Series, cov: pd.DataFrame, weights: dict[str, float]) -> tuple[float, float, float, float]:
    v = np.array([weights.get(name, 0.0) for name in params.index])
    beta = float(v @ params.to_numpy())
    se = math.sqrt(max(0.0, float(v @ cov.to_numpy() @ v)))
    lo, hi = ci(beta, se)
    p = float(2 * stats.norm.sf(abs(beta / se))) if se > 0 else np.nan
    return 100 * beta, 100 * lo, 100 * hi, p


def reporting_weighted_reestimation(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ratios = np.linspace(0.20, 1.00, 41)
    rows = []
    prepared = prepare_weighted_within(data)
    for ratio in ratios:
        params, cov = weighted_within_fit(prepared, float(ratio))
        under20 = contrast_from_params(params, cov, {"exempt": 1.0, "int_<20": 1.0})
        largest = contrast_from_params(params, cov, {"exempt": 1.0})
        difference = contrast_from_params(params, cov, {"int_<20": 1.0})
        rows.append({
            "Relative_nonfatal_completeness_under20_exempt": ratio,
            "Under20_exempt_minus_covered_pp": under20[0],
            "Under20_low": under20[1], "Under20_high": under20[2], "Under20_p": under20[3],
            "Largest_exempt_minus_covered_pp": largest[0],
            "Largest_low": largest[1], "Largest_high": largest[2], "Largest_p": largest[3],
            "Under20_minus_largest_contrast_pp": difference[0],
            "Difference_low": difference[1], "Difference_high": difference[2], "Difference_p": difference[3],
        })
    curve = pd.DataFrame(rows)

    def crossing(column: str) -> float | None:
        values = curve[column].to_numpy()
        r = curve["Relative_nonfatal_completeness_under20_exempt"].to_numpy()
        idx = np.where(np.diff(np.sign(values)) != 0)[0]
        if not len(idx):
            return None
        i = int(idx[0])
        return float(r[i] + (0 - values[i]) * (r[i + 1] - r[i]) / (values[i + 1] - values[i]))

    meta = {
        "under20_simple_contrast_zero_crossing": crossing("Under20_exempt_minus_covered_pp"),
        "under20_vs_largest_difference_zero_crossing": crossing("Under20_minus_largest_contrast_pp"),
        "weighting_interpretation": "A nonfatal incident in the under-20 exempt group receives frequency weight 1/r; fatal incidents and all other groups retain weight 1.",
    }
    return curve, meta


def period_description(frame: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("July 2019–December 2021", pd.Timestamp("2019-07-01"), pd.Timestamp("2021-12-31"), 30),
        ("2022–2023", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"), 24),
        ("January 2024–June 2025", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30"), 18),
    ]
    rows = []
    for label, start, end, months in periods:
        day = frame["date"].dt.normalize()  # whole end date is inclusive
        q = frame[(day >= start) & (day <= end)]
        for category, workforce in [("19인 이하", "<20"), ("500인 이상", "500+")]:
            g = q[q["작업자수"] == category]
            rows.append({
                "Period": label, "Workforce": workforce, "Months": months, "Incidents": int(len(g)),
                "Deaths": int(g["fatal"].sum()), "Incidents_per_month": len(g) / months,
                "Fatality_percent": 100 * g["fatal"].mean(),
            })
    return pd.DataFrame(rows)


def variable_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        {"Item": "Fatal outcome", "Operational_definition": "One if the incident record contains at least one death", "Role": "Incident-level conditional severity"},
        {"Item": "Covered", "Operational_definition": "Either statutory plan-coverage category", "Role": "Regulatory status"},
        {"Item": "Exempt", "Operational_definition": "Recorded non-covered plan category", "Role": "Regulatory status"},
        {"Item": "Recorded workforce", "Operational_definition": "Six ordered bands: <20, 20–49, 50–99, 100–299, 300–499, 500+", "Role": "Workforce-band effect modifier"},
        {"Item": "Acute traumatic sample", "Operational_definition": "Excludes disease, heat illness, none, not entered, unclassifiable, and property-only incidents", "Role": "Acute-traumatic analytic frame"},
        {"Item": "Event configuration", "Operational_definition": "Detailed accident type × detailed accident object × work process", "Role": "Base event-configuration definition"},
        {"Item": "Recurring configuration", "Operational_definition": "Event configuration with at least two incidents", "Role": "Required for within-configuration model"},
    ])


CONTRACT_VALUE_ORDER = [
    "1,000만원 미만", "1,000만 ~ 2,000만원 미만", "2,000만 ~ 4,000만원 미만", "4,000만 ~ 1억원 미만",
    "1억 ~ 2억원 미만", "2억 ~ 3억원 미만", "3억 ~ 5억원 미만", "5억 ~ 10억원 미만", "10억 ~ 20억원 미만",
    "20억 ~ 50억원 미만", "50억 ~ 100억원 미만", "100억 ~ 150억원 미만", "150억 ~ 200억원 미만",
    "200억 ~ 300억원 미만", "300억 ~ 500억원 미만", "500억 ~ 1,000억원 미만", "1,000억원 이상",
]
PROGRESS_ORDER = ["10% 미만", "10~19%", "20~29%", "30~39%", "40~49%", "50~59%", "60~69%", "70~79%", "80~89%", "90% 이상"]
PROFILE_MIN_RECORDS = 500


def _workforce_frame(df: pd.DataFrame) -> pd.DataFrame:
    complete = df[df["workforce_complete"] & df["plan_complete"]].copy()
    complete["workforce_score"] = complete["작업자수"].map({k: i for i, k in enumerate(WORKFORCE_ORDER)})
    return complete


def workforce_progress_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ordinal workforce score (0 = <20 ... 5 = 500+) by progress decile within each contract-value band."""
    complete = _workforce_frame(df)
    complete = complete[complete["공사비"].isin(CONTRACT_VALUE_ORDER) & complete["공정율"].isin(PROGRESS_ORDER)]
    rows = []
    for band in CONTRACT_VALUE_ORDER:
        g = complete[complete["공사비"] == band]
        if len(g) < PROFILE_MIN_RECORDS:
            continue
        means = g.groupby("공정율")["workforce_score"].mean().reindex(PROGRESS_ORDER)
        row = {"Contract_value_band": band, "Records": int(len(g))}
        row.update({f"Mean_score_{p}": float(v) for p, v in means.items()})
        row["Peak_progress_decile"] = str(means.idxmax())
        row["Final_decile_minus_peak"] = float(means.iloc[-1] - means.max())
        rows.append(row)
    return pd.DataFrame(rows)


def workforce_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Evidence on the meaning of the workforce field, computed from the shared extract.

    v1.0.0 wrote these values as fixed strings; v1.1.0 computes them. The record-position
    entry describes the layout of the source export and cannot be computed from the extract.
    """
    complete = _workforce_frame(df)
    cv = complete.assign(cv_score=complete["공사비"].map({k: i for i, k in enumerate(CONTRACT_VALUE_ORDER)}))
    cv = cv.dropna(subset=["cv_score"])
    rho_cv = stats.spearmanr(cv["workforce_score"], cv["cv_score"]).statistic
    area = pd.to_numeric(complete["연면적"], errors="coerce")
    fa = complete.assign(area=area)
    fa = fa[fa["area"].notna() & (fa["area"] > 0)]
    rho_area = stats.spearmanr(fa["workforce_score"], fa["area"]).statistic
    prof = workforce_progress_profile(df)
    mid = {"40~49%", "50~59%", "60~69%"}
    k_mid = int(prof["Peak_progress_decile"].isin(mid).sum())
    k_decline = int((prof["Final_decile_minus_peak"] < 0).sum())
    k_all = int(len(prof))
    score_cols = [f"Mean_score_{q}" for q in PROGRESS_ORDER]
    k_low = int((prof[score_cols].idxmin(axis=1) == score_cols[0]).sum())
    pattern = (f"Lowest in the first progress decile in {k_low} of {k_all} bands; peak at 40-69% in {k_mid} of {k_all}; "
               f"final decile below peak in {k_decline} of {k_all}")
    return pd.DataFrame([
        {"Property": "Position in record", "Value": "Site-characteristics block", "Interpretation": "Grouped with contract value and progress, not casualty fields"},
        {"Property": "Rank correlation with contract value", "Value": f"{rho_cv:.2f}", "Interpretation": f"Consistent with project scale (Spearman; n = {len(cv):,})"},
        {"Property": "Rank correlation with gross floor area", "Value": f"{rho_area:.2f}", "Interpretation": f"Consistent with project scale (Spearman; n = {len(fa):,})"},
        {"Property": "Pattern over progress within contract-value bands", "Value": pattern, "Interpretation": "Consistent with a point-in-time rather than cumulative count"},
    ])


def missing_composition(df: pd.DataFrame) -> pd.DataFrame:
    retained = df[df["workforce_complete"] & df["plan_complete"]]
    excluded = df[~(df["workforce_complete"] & df["plan_complete"])]
    rows = []
    for measure, func in [
        ("Incidents", lambda d: len(d)),
        ("Fatal (%)", lambda d: 100 * d["fatal"].mean()),
        ("Public project (%)", lambda d: 100 * (d["공공민간구분"] == "공공").mean()),
        ("Building works (%)", lambda d: 100 * (d["시설물 대분류"] == "건축").mean()),
        ("Contract value not entered (%)", lambda d: 100 * d["공사비"].eq("미입력").mean()),
    ]:
        rows.append({"Measure": measure, "Excluded": func(excluded), "Retained": func(retained)})
    ex_type = excluded["인적사고종류(대분류)"].mode().iloc[0]
    ret_type = retained["인적사고종류(대분류)"].mode().iloc[0]
    rows.append({
        "Measure": "Most frequent accident type",
        "Excluded": core.TYPE_ENGLISH.get(ex_type, str(ex_type)),
        "Retained": core.TYPE_ENGLISH.get(ret_type, str(ret_type)),
    })
    return pd.DataFrame(rows)


def make_figures(
    out: Path,
    contrasts: pd.DataFrame,
    strict_contrasts: pd.DataFrame,
    band_support: pd.DataFrame,
    reporting_curve: pd.DataFrame,
    fall_height: pd.DataFrame,
    fall_comp: pd.DataFrame,
    fall_std: pd.DataFrame,
    type_rd: pd.DataFrame,
) -> None:
    # Import Matplotlib lazily. The table-only process used by run_all.py
    # then exits without touching the font cache, avoiding cross-process
    # cache-lock delays before the fresh plotting process starts.
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
    })

    # Figure 1: formal safety-management coverage, project context, and conditional evidence.
    fig = plt.figure(figsize=(7.45, 5.15))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.04, 1.0], hspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 12.0); ax.set_ylim(0, 3.7); ax.axis("off")
    ax.set_title("A. Formal safety-management coverage and project context", loc="left", fontweight="bold", pad=2)

    boxes = {
        "conditions": (0.15, 2.10, 2.25, 0.82, "Statutory technical\nconditions"),
        "coverage": (3.10, 2.10, 2.25, 0.82, "Plan coverage or\nexemption"),
        "context": (3.10, 0.70, 2.25, 0.82, "Recorded project-scale\ncontext"),
        "process": (6.25, 1.68, 2.35, 0.92, "Plan-related safety\nprocesses"),
        "outcome": (9.55, 1.68, 2.25, 0.92, "Incident occurrence\nand consequences"),
        "unobserved": (6.25, 0.35, 2.35, 0.72, "Organizational capacity and\nplan implementation\n(unobserved)"),
    }
    for x, y, w, h, label in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035",
                                    facecolor="white", edgecolor="black", linewidth=0.85))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.2)

    def arr(axis, start, end, rad=0.0, dashed=False):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=9,
                                      linewidth=0.8, linestyle="--" if dashed else "-",
                                      connectionstyle=f"arc3,rad={rad}"))

    arr(ax, (2.40, 2.51), (3.10, 2.51))
    arr(ax, (5.35, 2.51), (6.25, 2.22))
    arr(ax, (5.35, 1.11), (6.25, 1.94))
    arr(ax, (8.60, 2.14), (9.55, 2.14))
    arr(ax, (5.35, 1.02), (9.55, 1.88), rad=-0.08)
    arr(ax, (5.35, 1.11), (6.25, 0.72), dashed=True)
    arr(ax, (7.42, 1.07), (7.42, 1.68), dashed=True)
    arr(ax, (8.60, 0.72), (10.10, 1.68), rad=-0.08, dashed=True)

    ax.text(4.23, 1.78, "Effect modification\nby recorded scale?", ha="center", va="center",
            fontsize=7.0, fontweight="bold")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(0, 12.0); ax2.set_ylim(0, 3.6); ax2.axis("off")
    ax2.set_title("B. Conditional comparison and evidence qualification", loc="left", fontweight="bold", pad=2)

    boxes2 = [
        (0.10, 1.45, 1.75, 0.94, "Worksite\nincident"),
        (2.35, 1.45, 1.95, 0.94, "Notification and\nregister entry"),
        (4.80, 1.45, 2.05, 0.94, "Public investigation\nand coding"),
        (7.35, 1.45, 2.15, 0.94, "Fatal or nonfatal\noutcome among reports"),
        (10.00, 1.45, 1.85, 0.94, "Prospective linked-\ndata evaluation"),
    ]
    for x, y, w, h, label in boxes2:
        ax2.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035",
                                     facecolor="white", edgecolor="black", linewidth=0.85))
        ax2.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.2)

    starts=[(1.85,1.92),(4.30,1.92),(6.85,1.92),(9.50,1.92)]
    ends=[(2.35,1.92),(4.80,1.92),(7.35,1.92),(10.00,1.92)]
    labels=["Ascertainment","Classification and\nrepresentativeness","Comparability and\noverlap","Usefulness"]
    tx=[2.10,4.55,7.10,9.75]
    for s0,e0,label,x0 in zip(starts,ends,labels,tx):
        arr(ax2,s0,e0)
        ax2.text(x0,2.62,label,ha="center",va="bottom",fontsize=6.7)

    ax2.text(6.0, 0.92,
             "Coverage status, workforce band, event configuration, and outcome are observed only after register entry.",
             ha="center", va="center", fontsize=6.9)
    ax2.text(6.0, 0.20,
             "Models compare covered and exempt incidents within recurring event contexts.\n"
             "They do not estimate accident incidence, plan implementation, or a causal coverage effect.",
             ha="center", va="bottom", fontsize=6.9)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.08)
    fig.savefig(out / "Figure_1_surveillance_coverage_audit.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: coverage contrasts and direct within-band support.
    c = contrasts.copy().reset_index(drop=True)
    s = strict_contrasts.copy().reset_index(drop=True)
    support_lookup = band_support.set_index("Workforce_band")
    fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(7.45, 4.85), gridspec_kw={"width_ratios": [1.58, 1.12]})
    y = np.arange(len(c))
    offset = 0.12
    for frame, marker, label, shift in [
        (c, "o", "Event-configuration model", -offset),
        (s, "s", "Within-band support-aligned model", offset),
    ]:
        values = frame["Estimate_pp"].to_numpy()
        lower = frame["CI_low_pp"].to_numpy()
        upper = frame["CI_high_pp"].to_numpy()
        errors = np.vstack([values - lower, upper - values])
        ax.errorbar(values, y + shift, xerr=errors, fmt=marker, capsize=2.5, linewidth=0.9,
                    markersize=4.5, label=label)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels(c["Contrast"].tolist()); ax.invert_yaxis()
    ax.set_xlabel("Exempt minus covered fatal-outcome contrast\n(percentage points)")
    ax.set_ylabel("Recorded workforce band")
    ax.set_title("A. Adjusted plan-coverage contrasts", loc="left")
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)

    ax_table.axis("off")
    table_rows = []
    for _, row in c.iterrows():
        label = row["Contrast"]
        mixed = int(support_lookup.loc[label, "Mixed_event_configurations"])
        table_rows.append([
            label,
            f"{int(row['Covered_deaths'])}/{int(row['Covered_incidents'])}",
            f"{int(row['Exempt_deaths'])}/{int(row['Exempt_incidents'])}",
            f"{mixed:,}",
        ])
    tbl = ax_table.table(cellText=table_rows,
                         colLabels=["Band", "Covered\nd/n", "Exempt\nd/n", "Mixed\nconfigs"],
                         cellLoc="center", colLoc="center", loc="center",
                         colWidths=[0.22, 0.27, 0.27, 0.24])
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.9); tbl.scale(1.0, 1.44)
    for (r, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_text_props(weight="bold")
    ax_table.set_title("B. Direct covered-exempt support", loc="left", pad=8)
    ax_table.text(0.5, 0.035,
                  "d/n = fatal incidents / reported incidents\nMixed configs = event configurations with both coverage states in that band",
                  transform=ax_table.transAxes, ha="center", va="bottom", fontsize=6.5)
    fig.subplots_adjust(wspace=0.20, left=0.12, right=0.995, top=0.91, bottom=0.13)
    fig.savefig(out / "Figure_2_workforce_contrasts_and_support.png", bbox_inches="tight")
    plt.close(fig)

    # Figure S1: model-aligned reporting-completeness sensitivity.
    q = reporting_curve.sort_values("Relative_nonfatal_completeness_under20_exempt")
    x = q["Relative_nonfatal_completeness_under20_exempt"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for estimate, low, high, label in [
        ("Under20_exempt_minus_covered_pp", "Under20_low", "Under20_high", "Local under-20 contrast"),
        ("Under20_minus_largest_contrast_pp", "Difference_low", "Difference_high", "Under-20 minus 500+ contrast difference"),
    ]:
        ax.plot(x, q[estimate], label=label)
        ax.fill_between(x, q[low], q[high], alpha=.16)
    local_crossing = 0.7297666648964944
    extreme_crossing = 0.4282545893358483
    ax.axhline(0, linestyle="--", linewidth=.8)
    ax.axvline(local_crossing, linestyle="-.", linewidth=.9)
    ax.axvline(extreme_crossing, linestyle=":", linewidth=.9)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.text(local_crossing + 0.012, ymax - 0.08 * yrange,
            "Local contrast = 0\nr = 0.730", ha="left", va="top", fontsize=7.2)
    ax.text(extreme_crossing + 0.012, ymin + 0.10 * yrange,
            "Extreme-band difference = 0\nr = 0.428", ha="left", va="bottom", fontsize=7.2)
    ax.set_xlabel("Assumed relative nonfatal reporting completeness, under-20 exempt group (r)")
    ax.set_ylabel("Adjusted contrast (percentage points)")
    ax.set_title("Reporting-completeness sensitivity: two distinct tipping points")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "Figure_S1_weighted_reporting_reestimation.png", bbox_inches="tight"); plt.close(fig)

    # Figure S2: broad accident-type heterogeneity.
    q = type_rd[(type_rd["Accidents"] >= 50) & np.isfinite(type_rd["Risk_difference_pp"])].copy().sort_values("Risk_difference_pp")
    y = np.arange(len(q)); values = q["Risk_difference_pp"].to_numpy()
    xerr = np.vstack([values - q["CI_low_pp"].to_numpy(), q["CI_high_pp"].to_numpy() - values])
    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    ax.errorbar(values, y, xerr=xerr, fmt="o", capsize=2, linewidth=.9)
    ax.axvline(0, linestyle="--", linewidth=.8)
    ax.set_yticks(y); ax.set_yticklabels(q["Accident_type"])
    ax.set_xlabel("Raw fatality difference: <50 minus ≥50 workers (percentage points)")
    ax.set_title("Broad types with at least 50 acute traumatic incidents")
    fig.tight_layout(); fig.savefig(out / "Figure_S2_accident_type_heterogeneity.png", bbox_inches="tight"); plt.close(fig)

    # Figure S3: fall-height decomposition.
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.8))
    pivot = fall_comp.pivot(index="Workforce", columns="Height_label", values="Share").reindex(WORKFORCE_LABELS)
    bottom = np.zeros(len(pivot))
    for height in core.HEIGHT_LABELS:
        values = pivot.get(height, pd.Series(0, index=pivot.index)).fillna(0).to_numpy() * 100
        axes[0].bar(np.arange(len(pivot)), values, bottom=bottom, width=.75, label=height, edgecolor="black", linewidth=.2)
        bottom += values
    axes[0].set_xticks(range(len(pivot))); axes[0].set_xticklabels(WORKFORCE_LABELS)
    axes[0].set_ylabel("Share of fall records (%)"); axes[0].set_title("A. Recorded height composition")
    axes[0].set_ylim(0, 128); axes[0].set_yticks([0, 25, 50, 75, 100])  # headroom so the legend clears the bars
    axes[0].legend(ncol=6, frameon=False, loc="upper center", fontsize=7, columnspacing=1.0, handlelength=1.2)
    h = fall_height.set_index("Height_label").reindex(core.HEIGHT_LABELS)
    xx = np.arange(len(h)); vv = h["Fatality"].to_numpy() * 100
    err = np.vstack([(h["Fatality"] - h["CI_low"]) * 100, (h["CI_high"] - h["Fatality"]) * 100])
    axes[1].errorbar(xx, vv, yerr=err, fmt="o", capsize=3)
    axes[1].set_xticks(xx); axes[1].set_xticklabels(core.HEIGHT_LABELS)
    axes[1].set_ylabel("Fatality among reported falls (%)"); axes[1].set_title("B. Fatality by recorded height")
    f = fall_std.set_index("Workforce").reindex(WORKFORCE_LABELS)
    xx = np.arange(len(f))
    axes[2].plot(xx, f["Raw_fatality"] * 100, marker="o", label="Raw")
    axes[2].plot(xx, f["Height_standardized_fatality"] * 100, marker="s", linestyle="--", label="Height-standardized")
    axes[2].set_xticks(xx); axes[2].set_xticklabels(WORKFORCE_LABELS)
    axes[2].set_ylabel("Fatality among reported falls (%)"); axes[2].set_title("C. Workforce profile")
    axes[2].legend(frameon=False)
    fig.subplots_adjust(hspace=.62, top=.97, bottom=.08, left=.12, right=.98)
    fig.savefig(out / "Figure_S3_fall_height.png", bbox_inches="tight"); plt.close(fig)

def write_data_dictionary(path: Path) -> None:
    core.write_data_dictionary(path)


def run_pipeline(data_path: Path, out: Path, make_plots: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df, frame, recurring, singleton = primary_data(data_path)
    complete = df[df["workforce_complete"] & df["plan_complete"]].copy()

    # Main outputs.
    sample_flow = core.sample_flow(df, frame, recurring, singleton)
    sample_compare = core.sample_comparison(recurring, singleton)
    save_csv(sample_flow, out, "Table_1A_sample_flow.csv")
    save_csv(sample_compare, out, "Table_1B_recurring_vs_singleton.csv")

    six_contrasts, trend, six_meta, six_fit = six_band_primary(recurring)
    save_csv(six_contrasts, out, "Table_2A_primary_six_band_contrasts.csv")
    save_csv(trend, out, "Table_2B_ordinal_trend.csv")
    with open(out / "Table_2C_six_band_omnibus.json", "w", encoding="utf-8") as f:
        json.dump(six_meta, f, ensure_ascii=False, indent=2)

    band_support = within_band_support(recurring)
    strict_contrasts, strict_trend, strict_meta, strict_fit = strict_six_band(recurring, restrict_to_mixed_strata=False)
    mixed_contrasts, mixed_trend, mixed_meta, mixed_fit = strict_six_band(recurring, restrict_to_mixed_strata=True)
    save_csv(strict_contrasts, out, "Table_2D_support_aligned_six_band_contrasts.csv")
    save_csv(strict_trend, out, "Table_2E_support_aligned_ordinal_trend.csv")
    with open(out / "Table_2F_support_aligned_summary.json", "w", encoding="utf-8") as f:
        json.dump({"full_strict_model": strict_meta, "mixed_strata_only": mixed_meta}, f, ensure_ascii=False, indent=2)

    support, overlap_est = support_diagnostics(recurring)
    save_csv(support, out, "Table_3A_common_support.csv")
    save_csv(overlap_est, out, "Table_3B_overlap_restricted_binary_interaction.csv")
    binary_ladder = model_ladder(frame, recurring)
    save_csv(binary_ladder, out, "Table_3C_binary_model_ladder.csv")

    # Supplementary measurement and composition.
    save_csv(variable_definitions(), out, "Table_S01_variable_definitions.csv")
    audit, candidate = core.label_audit(complete)
    save_csv(audit, out, "Table_S02_plan_label_audit.csv")
    save_csv(workforce_evidence(df), out, "Table_S03_workforce_field_evidence.csv")
    save_csv(workforce_progress_profile(df), out, "Table_S03B_workforce_progress_profile.csv")
    save_csv(missing_composition(df), out, "Table_S04_missing_field_composition.csv")
    save_csv(project_balance(recurring), out, "Table_S05_project_composition_four_groups.csv")

    # Supplementary support, influence, and scale analyses.
    save_csv(support, out, "Table_S06A_common_support.csv")
    save_csv(overlap_est, out, "Table_S06B_overlap_restricted_estimates.csv")
    save_csv(binary_ladder, out, "Table_S06C_binary_model_ladder.csv")
    save_csv(band_support, out, "Table_S06D_within_band_support.csv")
    strict_summary = pd.DataFrame([
        {
            "Specification": "Configuration × band fixed effects, all recurring records",
            "n": strict_meta["n"],
            "Six_band_Wald_chi2": strict_meta["wald_chi2"],
            "Six_band_p": strict_meta["p"],
            "Ordered_trend_pp_per_band": strict_meta["trend_pp_per_band"],
            "Trend_low": strict_meta["trend_ci_pp"][0],
            "Trend_high": strict_meta["trend_ci_pp"][1],
            "Trend_p": strict_meta["trend_p"],
        },
        {
            "Specification": "Configuration × band fixed effects, mixed strata only",
            "n": mixed_meta["n"],
            "Six_band_Wald_chi2": mixed_meta["wald_chi2"],
            "Six_band_p": mixed_meta["p"],
            "Ordered_trend_pp_per_band": mixed_meta["trend_pp_per_band"],
            "Trend_low": mixed_meta["trend_ci_pp"][0],
            "Trend_high": mixed_meta["trend_ci_pp"][1],
            "Trend_p": mixed_meta["trend_p"],
        },
    ])
    save_csv(strict_summary, out, "Table_S06E_support_aligned_summary.csv")
    save_csv(six_contrasts, out, "Table_S07A_primary_six_band_contrasts.csv")
    save_csv(trend, out, "Table_S07B_ordinal_trend.csv")
    save_csv(strict_contrasts, out, "Table_S07C_support_aligned_band_contrasts.csv")
    save_csv(strict_trend, out, "Table_S07D_support_aligned_ordinal_trend.csv")
    save_csv(threshold_sensitivity(recurring), out, "Table_S08_threshold_sensitivity_holm.csv")
    save_csv(leave_band_out(recurring), out, "Table_S09A_leave_one_band_out.csv")
    save_csv(band_aggregation_sensitivity(recurring), out, "Table_S09B_band_aggregation.csv")
    save_csv(alternative_contract(recurring), out, "Table_S10A_alternative_contract_scale.csv")
    save_csv(plan_three_category(recurring), out, "Table_S10B_three_category_plan_status.csv")

    # Robustness and confounding diagnostics.
    save_csv(same_spec_robustness(df, frame, recurring, candidate), out, "Table_S11A_same_specification_robustness.csv")
    save_csv(extra_controls(recurring), out, "Table_S11B_extra_project_controls.csv")
    save_csv(single_casualty_sensitivity(recurring), out, "Table_S11C_single_casualty.csv")
    save_csv(leave_year_out(recurring), out, "Table_S12_leave_one_year_out.csv")
    save_csv(core.fe_granularity_ladder(frame), out, "Table_S13_fixed_effect_granularity.csv")

    bounded, bounded_meta = bounded_logit_sensitivity(recurring)
    save_csv(bounded, out, "Table_S14A_bounded_logit_marginal_contrasts.csv")
    save_csv(covariance_sensitivity(recurring), out, "Table_S14B_covariance_estimators.csv")
    with open(out / "Table_S14_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"bounded_logit": bounded_meta}, f, ensure_ascii=False, indent=2)

    reporting_curve, reporting_meta = reporting_weighted_reestimation(recurring)
    save_csv(reporting_curve, out, "Table_S15_weighted_reporting_reestimation.csv")
    with open(out / "Table_S15_reporting_crossings.json", "w", encoding="utf-8") as f:
        json.dump(reporting_meta, f, ensure_ascii=False, indent=2)
    save_csv(period_description(frame), out, "Table_S16_period_descriptive_check.csv")

    all_types, type_rd, type_meta = core.all_type_tables(df, frame)
    save_csv(all_types, out, "Table_S17A_all_broad_accident_categories.csv")
    save_csv(type_rd, out, "Table_S17B_type_risk_differences.csv")
    with open(out / "Table_S17_type_omnibus.json", "w", encoding="utf-8") as f:
        json.dump(type_meta, f, ensure_ascii=False, indent=2)
    fall_height, fall_comp, fall_std = core.fall_height_analysis(df)
    save_csv(fall_height, out, "Table_S18A_fall_height_fatality.csv")
    save_csv(fall_comp, out, "Table_S18B_fall_height_composition.csv")
    save_csv(fall_std, out, "Table_S18C_fall_height_standardization.csv")

    write_data_dictionary(out / "data_dictionary.csv")
    if make_plots:
        make_figures(out, six_contrasts, strict_contrasts, band_support, reporting_curve, fall_height, fall_comp, fall_std, type_rd)

    trend_row = trend.iloc[0]
    all4 = overlap_est.loc[overlap_est["Specification"] == "All four combinations"].iloc[0]
    summary = {
        "source_records": int(len(df)),
        "primary_acute_traumatic_frame": int(len(frame)),
        "primary_recurring_sample": int(len(recurring)),
        "primary_event_configurations": int(recurring["event_cell"].nunique()),
        "six_band_omnibus": six_meta,
        "ordinal_trend_pp_per_band": float(trend_row["Coefficient_pp"]),
        "ordinal_trend_ci_pp": [float(trend_row["CI_low_pp"]), float(trend_row["CI_high_pp"])],
        "ordinal_trend_p": float(trend_row["p"]),
        "support_aligned_six_band": strict_meta,
        "support_aligned_mixed_strata_only": mixed_meta,
        "within_band_support": band_support.to_dict(orient="records"),
        "all_four_support_binary_interaction_pp": float(all4["Coefficient_pp"]),
        "all_four_support_binary_interaction_ci_pp": [float(all4["CI_low_pp"]), float(all4["CI_high_pp"])],
        "reporting_weighted_crossings": reporting_meta,
        "bounded_logit": bounded_meta,
    }
    assert summary["source_records"] == 37196
    assert summary["primary_acute_traumatic_frame"] == 32700
    assert summary["primary_recurring_sample"] == 28821
    assert summary["primary_event_configurations"] == 3493
    assert abs(summary["six_band_omnibus"]["wald_chi2"] - 12.03208) < 0.001
    assert abs(summary["ordinal_trend_pp_per_band"] - (-0.378782)) < 0.001
    assert abs(summary["support_aligned_six_band"]["wald_chi2"] - 7.40634) < 0.001
    assert abs(summary["support_aligned_six_band"]["trend_pp_per_band"] - (-0.34541)) < 0.001
    assert summary["within_band_support"][0]["Mixed_event_configurations"] == 1066
    assert summary["within_band_support"][-1]["Mixed_event_configurations"] == 141
    with open(out / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce all manuscript and supplementary outputs.")
    parser.add_argument("--data", type=Path, default=Path("CSI_analytic_deidentified.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-figures", action="store_true", help="Write tables and summary only; figures can be generated in a fresh plotting process.")
    args = parser.parse_args()
    summary = run_pipeline(args.data, args.output, make_plots=not args.skip_figures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
