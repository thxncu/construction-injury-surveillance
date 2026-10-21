#!/usr/bin/env python3
"""Reframe the analysis around direct within-band comparison support.

The primary comparative model is the event-configuration-by-workforce-band fixed-effect
model, which identifies covered-versus-exempt contrasts only from direct
within-band support. A less saturated event-configuration model is retained as
a complementary pooled like-event detection model. Additional diagnostics
quantify endpoint dependence and sparse-outcome sensitivity in the 500-or-more
workforce band.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analysis_main as base
import analysis_core as core

WORKFORCE_ORDER = base.WORKFORCE_ORDER
WORKFORCE_LABELS = base.WORKFORCE_LABELS
PRIMARY_CONTROLS = base.PRIMARY_CONTROLS


def save_csv(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / name, index=False, encoding="utf-8-sig")


def direct_support_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Describe the strata that directly compare covered and exempt records."""
    d = data.copy()
    d["event_cell_band"] = d["event_cell"].astype(str) + " || " + d["작업자수"].astype(str)
    state_count = d.groupby("event_cell_band", observed=True)["exempt"].nunique()
    mixed_ids = state_count[state_count == 2].index
    q = d[d["event_cell_band"].isin(mixed_ids)].copy()

    rows: list[dict] = []
    for category, label in zip(WORKFORCE_ORDER, WORKFORCE_LABELS):
        b = q[q["작업자수"] == category]
        c = b[b["exempt"] == 0]
        e = b[b["exempt"] == 1]
        outcome_varying = int((b.groupby("event_cell_band", observed=True)["fatal"].nunique() == 2).sum()) if len(b) else 0
        rows.append({
            "Workforce_band": label,
            "Mixed_event_configurations": int(b["event_cell_band"].nunique()),
            "Outcome_varying_mixed_strata": outcome_varying,
            "Covered_incidents_mixed": int(len(c)),
            "Covered_deaths_mixed": int(c["fatal"].sum()),
            "Exempt_incidents_mixed": int(len(e)),
            "Exempt_deaths_mixed": int(e["fatal"].sum()),
        })
    return pd.DataFrame(rows)


def _newcombe_difference_ci(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float]:
    """Newcombe-Wilson interval for the risk difference p1 minus p0."""
    p1 = k1 / n1
    p0 = k0 / n0
    l1, u1 = core.wilson_ci(k1, n1)
    l0, u0 = core.wilson_ci(k0, n0)
    lower = (p1 - p0) - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    upper = (p1 - p0) + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return lower, upper


def sparse_500_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Raw exact diagnostics for the sparse 500-or-more band."""
    d = data[data["작업자수"] == "500인 이상"].copy()
    d["event_cell_band"] = d["event_cell"].astype(str) + " || " + d["작업자수"].astype(str)
    state_count = d.groupby("event_cell_band", observed=True)["exempt"].nunique()
    mixed_ids = state_count[state_count == 2].index
    mixed = d[d["event_cell_band"].isin(mixed_ids)].copy()

    rows: list[dict] = []
    for label, q in [("All 500+ reports", d), ("Mixed strata (within-band model)", mixed)]:
        e = q[q["exempt"] == 1]
        c = q[q["exempt"] == 0]
        ed, en = int(e["fatal"].sum()), int(len(e))
        cd, cn = int(c["fatal"].sum()), int(len(c))
        rd = ed / en - cd / cn
        lo, hi = _newcombe_difference_ci(ed, en, cd, cn)
        fisher_p = float(stats.fisher_exact([[ed, en - ed], [cd, cn - cd]], alternative="two-sided").pvalue)
        strata = np.nan
        outcome_varying = np.nan
        if q is mixed:  # strata counts apply only to the mixed-strata comparison
            strata = int(q["event_cell_band"].nunique())
            outcome_varying = int((q.groupby("event_cell_band", observed=True)["fatal"].nunique() == 2).sum())
        rows.append({
            "Comparison": label,
            "Exempt_deaths": ed,
            "Exempt_incidents": en,
            "Covered_deaths": cd,
            "Covered_incidents": cn,
            "Raw_risk_difference_pp": 100 * rd,
            "Newcombe_CI_low_pp": 100 * lo,
            "Newcombe_CI_high_pp": 100 * hi,
            "Fisher_exact_p": fisher_p,
            "Mixed_strata": strata,
            "Outcome_varying_mixed_strata": outcome_varying,
        })
    return pd.DataFrame(rows)


def _one_event_update_range(fit: core.FitResult, data: pd.DataFrame, mixed_only: bool) -> dict:
    """Update the 500+ point estimate after changing one exempt outcome.

    Point-estimate changes are computed exactly for the fixed design. Confidence
    intervals are not recomputed; the exercise is an influence diagnostic rather
    than an alternative inferential model.
    """
    names = list(fit.result.params.index)
    X = fit.X[names].astype(float)
    group = fit.group.astype(str)
    Xdm = X - X.groupby(group).transform("mean")
    inv_xtx = np.linalg.pinv(Xdm.to_numpy().T @ Xdm.to_numpy())
    contrast = np.array([1.0 if name == "exempt" else 0.0 for name in names])
    observed = float(contrast @ fit.result.params.to_numpy()) * 100

    q = data[(data["작업자수"] == "500인 이상") & (data["exempt"] == 1)].copy()
    if mixed_only:
        temp = data.copy()
        temp["event_cell_band"] = temp["event_cell"].astype(str) + " || " + temp["작업자수"].astype(str)
        state_count = temp.groupby("event_cell_band", observed=True)["exempt"].nunique()
        mixed_ids = set(state_count[state_count == 2].index)
        q_ids = q["event_cell"].astype(str) + " || " + q["작업자수"].astype(str)
        q = q[q_ids.isin(mixed_ids)]

    add_values: list[float] = []
    for idx in q.index[q["fatal"] == 0]:
        delta = float(contrast @ inv_xtx @ Xdm.loc[idx].to_numpy()) * 100
        add_values.append(observed + delta)

    remove_values: list[float] = []
    for idx in q.index[q["fatal"] == 1]:
        delta = float(contrast @ inv_xtx @ Xdm.loc[idx].to_numpy()) * 100
        remove_values.append(observed - delta)

    return {
        "Observed_pp": observed,
        "Candidate_nonfatal_records": int(len(add_values)),
        "One_nonfatal_to_fatal_min_pp": float(min(add_values)) if add_values else np.nan,
        "One_nonfatal_to_fatal_max_pp": float(max(add_values)) if add_values else np.nan,
        "Candidate_fatal_records": int(len(remove_values)),
        "One_fatal_to_nonfatal_min_pp": float(min(remove_values)) if remove_values else np.nan,
        "One_fatal_to_nonfatal_max_pp": float(max(remove_values)) if remove_values else np.nan,
    }


def one_event_perturbation(data: pd.DataFrame, pooled_fit: core.FitResult, direct_fit: core.FitResult) -> pd.DataFrame:
    return pd.DataFrame([
        {"Model": "Within-band fixed effects", **_one_event_update_range(direct_fit, direct_fit.data, mixed_only=True)},
        {"Model": "Pooled fixed effects", **_one_event_update_range(pooled_fit, data, mixed_only=False)},
    ])


def _band_fit_and_test(data: pd.DataFrame, categories: list[str], direct: bool) -> dict:
    d = data[data["작업자수"].isin(categories)].copy()
    labels = [WORKFORCE_LABELS[WORKFORCE_ORDER.index(category)] for category in categories]
    main = ["exempt"]
    interaction_terms: list[str] = []
    for category, label in zip(categories[:-1], labels[:-1]):
        main_term = f"wf_{label}"
        interaction = f"int_{label}"
        d[main_term] = (d["작업자수"] == category).astype(float)
        d[interaction] = d[main_term] * d["exempt"]
        main.extend([main_term, interaction])
        interaction_terms.append(interaction)
    if direct:
        d["event_cell_band"] = d["event_cell"].astype(str) + " || " + d["작업자수"].astype(str)
        fe_col = "event_cell_band"
    else:
        fe_col = "event_cell"
    fit = core.fit_ols(d, main, PRIMARY_CONTROLS, fe_col=fe_col, cluster_col="event_cell")
    names = list(fit.result.params.index)
    R = np.zeros((len(interaction_terms), len(names)))
    for i, term in enumerate(interaction_terms):
        R[i, names.index(term)] = 1.0
    test = fit.result.wald_test(R, scalar=True)
    return {"n": int(len(d)), "Wald_chi2": float(test.statistic), "df": int(test.df_denom), "p": float(test.pvalue)}


def _middle_equality_test(fit: core.FitResult) -> dict:
    names = list(fit.result.params.index)
    terms = ["int_20–49", "int_50–99", "int_100–299", "int_300–499"]
    R = []
    for term in terms[1:]:
        row = np.zeros(len(names))
        row[names.index(term)] = 1.0
        row[names.index(terms[0])] = -1.0
        R.append(row)
    test = fit.result.wald_test(np.asarray(R), scalar=True)
    return {"n": int(fit.result.nobs), "Wald_chi2": float(test.statistic), "df": int(test.df_denom), "p": float(test.pvalue)}


def endpoint_heterogeneity_tests(data: pd.DataFrame, pooled_fit: core.FitResult, direct_fit: core.FitResult) -> pd.DataFrame:
    rows: list[dict] = []
    for diagnostic, categories in [("Exclude <20", WORKFORCE_ORDER[1:]), ("Exclude 500+", WORKFORCE_ORDER[:-1])]:
        for model, direct in [("Within-band fixed effects", True), ("Pooled fixed effects", False)]:
            rows.append({"Diagnostic": diagnostic, "Model": model, **_band_fit_and_test(data, categories, direct)})
    for model, fit in [("Within-band fixed effects", direct_fit), ("Pooled fixed effects", pooled_fit)]:
        rows.append({"Diagnostic": "Equality across four middle bands", "Model": model, **_middle_equality_test(fit)})
    return pd.DataFrame(rows)



def _prepare_direct_weighted_within(data: pd.DataFrame) -> dict:
    """Prepare the weighted within-band direct-support design."""
    d = data.copy()
    d["event_cell_band"] = d["event_cell"].astype(str) + " || " + d["작업자수"].astype(str)
    main = ["exempt"]
    for category, label in zip(WORKFORCE_ORDER[:-1], WORKFORCE_LABELS[:-1]):
        d[f"wf_{label}"] = (d["작업자수"] == category).astype(float)
        d[f"int_{label}"] = d[f"wf_{label}"] * d["exempt"]
        main.extend([f"wf_{label}", f"int_{label}"])
    Xdf = core.design_matrix(d, main, PRIMARY_CONTROLS)
    group = d["event_cell_band"].astype(str)
    Xdm0 = Xdf - Xdf.groupby(group).transform("mean")
    varying = Xdm0.abs().sum(axis=0) > 1e-12
    Xdf = Xdf.loc[:, varying]
    X = Xdf.to_numpy(dtype=float)
    y = d["fatal"].to_numpy(dtype=float)
    group_codes, groups = pd.factorize(group, sort=True)
    cluster_codes, clusters = pd.factorize(d["event_cell"].astype(str), sort=True)
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
        "columns": list(Xdf.columns), "X": X, "y": y,
        "group_codes": group_codes, "group_count": group_count,
        "cluster_codes": cluster_codes, "cluster_count": len(clusters),
        "focal": focal.astype(float), "base_count": base_count,
        "focal_count": focal_count, "y_sum": y_sum,
        "X_sum": X_sum, "X_focal_sum": X_focal_sum,
    }


def _direct_weighted_fit(prepared: dict, ratio: float) -> tuple[pd.Series, pd.DataFrame]:
    if not (0 < ratio <= 1):
        raise ValueError("ratio must be in (0, 1]")
    X = prepared["X"]
    y = prepared["y"]
    codes = prepared["group_codes"]
    focal = prepared["focal"]
    alpha = 1.0 / ratio - 1.0
    weights = 1.0 + alpha * focal
    group_weights = prepared["base_count"] + alpha * prepared["focal_count"]
    y_mean = prepared["y_sum"] / group_weights
    X_mean = (prepared["X_sum"] + alpha * prepared["X_focal_sum"]) / group_weights[:, None]
    y_dm = y - y_mean[codes]
    X_dm = X - X_mean[codes]
    xtwx = X_dm.T @ (X_dm * weights[:, None])
    beta = np.linalg.pinv(xtwx) @ (X_dm.T @ (weights * y_dm))
    residual = y_dm - X_dm @ beta

    cluster_codes = prepared["cluster_codes"]
    cluster_count = prepared["cluster_count"]
    scores = np.zeros((cluster_count, X.shape[1]), dtype=float)
    np.add.at(scores, cluster_codes, X_dm * (weights * residual)[:, None])
    meat = scores.T @ scores
    bread = np.linalg.pinv(xtwx)
    n, k = X.shape
    correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    columns = prepared["columns"]
    return pd.Series(beta, index=columns), pd.DataFrame(covariance, index=columns, columns=columns)


def _contrast_from_weighted(params: pd.Series, cov: pd.DataFrame, weights: dict[str, float]) -> tuple[float, float, float, float]:
    vector = np.array([weights.get(name, 0.0) for name in params.index], dtype=float)
    beta = float(vector @ params.to_numpy())
    se = math.sqrt(max(0.0, float(vector @ cov.to_numpy() @ vector)))
    lo = beta - 1.959963984540054 * se
    hi = beta + 1.959963984540054 * se
    p = float(2 * stats.norm.sf(abs(beta / se))) if se > 0 else np.nan
    return 100 * beta, 100 * lo, 100 * hi, p


def direct_support_reporting_reestimation(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ratios = np.linspace(0.20, 1.00, 41)
    prepared = _prepare_direct_weighted_within(data)
    rows = []
    for ratio in ratios:
        params, cov = _direct_weighted_fit(prepared, float(ratio))
        under20 = _contrast_from_weighted(params, cov, {"exempt": 1.0, "int_<20": 1.0})
        largest = _contrast_from_weighted(params, cov, {"exempt": 1.0})
        difference = _contrast_from_weighted(params, cov, {"int_<20": 1.0})
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
        ratio_values = curve["Relative_nonfatal_completeness_under20_exempt"].to_numpy()
        idx = np.where(np.diff(np.sign(values)) != 0)[0]
        if not len(idx):
            return None
        i = int(idx[0])
        return float(ratio_values[i] + (0 - values[i]) * (ratio_values[i + 1] - ratio_values[i]) / (values[i + 1] - values[i]))

    meta = {
        "model": "Within-band fixed effects",
        "under20_simple_contrast_zero_crossing": crossing("Under20_exempt_minus_covered_pp"),
        "under20_vs_largest_difference_zero_crossing": crossing("Under20_minus_largest_contrast_pp"),
        "weighting_interpretation": "A nonfatal incident in the under-20 exempt group receives frequency weight 1/r; fatal incidents and all other records retain weight 1.",
    }
    return curve, meta


def make_reporting_figure(out: Path, direct_curve: pd.DataFrame, pooled_curve: pd.DataFrame, direct_meta: dict, pooled_meta: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.1), sharex=True)
    model_specs = [
        (direct_curve, "Within-band fixed effects", "s"),
        (pooled_curve, "Pooled fixed effects", "o"),
    ]
    for curve, label, marker in model_specs:
        q = curve.sort_values("Relative_nonfatal_completeness_under20_exempt")
        x = q["Relative_nonfatal_completeness_under20_exempt"]
        axes[0].plot(x, q["Under20_exempt_minus_covered_pp"], marker=marker, markevery=5, markersize=3.2, label=label)
        axes[0].fill_between(x, q["Under20_low"], q["Under20_high"], alpha=0.12)
        axes[1].plot(x, q["Under20_minus_largest_contrast_pp"], marker=marker, markevery=5, markersize=3.2, label=label)
        axes[1].fill_between(x, q["Difference_low"], q["Difference_high"], alpha=0.12)
    axes[0].axhline(0, linestyle="--", linewidth=0.8)
    axes[1].axhline(0, linestyle="--", linewidth=0.8)
    for value, style in [
        (direct_meta["under20_simple_contrast_zero_crossing"], "-."),
        (pooled_meta["under20_simple_contrast_zero_crossing"], ":"),
    ]:
        axes[0].axvline(value, linestyle=style, linewidth=0.9)
    for value, style in [
        (direct_meta["under20_vs_largest_difference_zero_crossing"], "-."),
        (pooled_meta["under20_vs_largest_difference_zero_crossing"], ":"),
    ]:
        axes[1].axvline(value, linestyle=style, linewidth=0.9)
    axes[0].set_title("A. Local under-20 exempt-minus-covered contrast", loc="left")
    axes[1].set_title("B. Under-20 minus 500-or-more contrast difference", loc="left")
    axes[0].set_ylabel("Adjusted contrast (percentage points)")
    axes[1].set_ylabel("Adjusted contrast (percentage points)")
    axes[1].set_xlabel("Assumed relative nonfatal reporting completeness in the under-20 exempt group (r)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].legend(frameon=False, loc="upper left")
    fig.subplots_adjust(hspace=0.36, left=0.12, right=0.98, top=0.96, bottom=0.10)
    fig.savefig(out / "Figure_S1_weighted_reporting_reestimation.png", bbox_inches="tight")
    plt.close(fig)

def make_direct_support_figure2(
    out: Path,
    direct_contrasts: pd.DataFrame,
    pooled_contrasts: pd.DataFrame,
    direct_counts: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dct = direct_contrasts.reset_index(drop=True)
    pooled = pooled_contrasts.reset_index(drop=True)
    support = direct_counts.set_index("Workforce_band")

    fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(7.45, 4.38), gridspec_kw={"width_ratios": [1.58, 1.18]})
    y = np.arange(len(dct))
    offset = 0.12
    for frame, marker, label, shift in [
        (dct, "s", "Within-band fixed effects", -offset),
        (pooled, "o", "Pooled fixed effects", offset),
    ]:
        values = frame["Estimate_pp"].to_numpy()
        lower = frame["CI_low_pp"].to_numpy()
        upper = frame["CI_high_pp"].to_numpy()
        errors = np.vstack([values - lower, upper - values])
        ax.errorbar(values, y + shift, xerr=errors, fmt=marker, capsize=2.5, linewidth=0.9,
                    markersize=4.5, label=label)
    ax.axvline(0, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(dct["Contrast"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Exempt minus covered fatal-outcome contrast\n(percentage points)")
    ax.set_ylabel("Recorded workforce band")
    ax.set_title("A. Adjusted coverage contrasts", loc="left")
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)

    ax_table.axis("off")
    rows = []
    for label in dct["Contrast"]:
        r = support.loc[label]
        rows.append([
            label,
            f"{int(r['Covered_deaths_mixed'])}/{int(r['Covered_incidents_mixed'])}",
            f"{int(r['Exempt_deaths_mixed'])}/{int(r['Exempt_incidents_mixed'])}",
            f"{int(r['Mixed_event_configurations']):,}",
            f"{int(r['Outcome_varying_mixed_strata']):,}",
        ])
    table = ax_table.table(
        cellText=rows,
        colLabels=["Band", "Covered\nd/n", "Exempt\nd/n", "Mixed\nstrata", "Outcome-\nvarying"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.18, 0.23, 0.23, 0.18, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.4)
    table.scale(1.0, 1.48)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(weight="bold")
    ax_table.set_title("B. Overlap within workforce bands", loc="left", pad=8)
    ax_table.text(
        0.5,
        0.015,
        "d/n = fatal incidents / reported incidents within mixed strata",
        transform=ax_table.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.3,
    )
    fig.subplots_adjust(wspace=0.18, left=0.12, right=0.995, top=0.91, bottom=0.13)
    fig.savefig(out / "Figure_2_workforce_contrasts_and_support.png", bbox_inches="tight")
    plt.close(fig)


def _rename_or_remove_outputs(out: Path) -> None:
    rename_map = {
        "Table_2A_primary_six_band_contrasts.csv": "Table_2C_pooled_like_event_contrasts.csv",
        "Table_2C_six_band_omnibus.json": "Table_2D_pooled_like_event_summary.json",
        "Table_2D_support_aligned_six_band_contrasts.csv": "Table_2A_within_band_direct_support_contrasts.csv",
        "Table_2F_support_aligned_summary.json": "Table_2B_within_band_direct_support_summary.json",
        "Table_S06E_support_aligned_summary.csv": "Table_S06E_within_band_direct_support_summary.csv",
        "Table_S07A_primary_six_band_contrasts.csv": "Table_S07B_pooled_like_event_contrasts.csv",
        "Table_S07C_support_aligned_band_contrasts.csv": "Table_S07A_within_band_direct_support_contrasts.csv",
    }
    for old, new in rename_map.items():
        old_path, new_path = out / old, out / new
        if old_path.exists():
            if new_path.exists():
                new_path.unlink()
            old_path.rename(new_path)
    for name in [
        "Table_2B_ordinal_trend.csv",
        "Table_2E_support_aligned_ordinal_trend.csv",
        "Table_S06D_within_band_support.csv",
        "Table_S07B_ordinal_trend.csv",
        "Table_S07D_support_aligned_ordinal_trend.csv",
    ]:
        path = out / name
        if path.exists():
            path.unlink()


def run_pipeline(data_path: Path, out: Path) -> dict:
    # Generate all established analyses and figures first.
    base_summary = base.run_pipeline(data_path, out, make_plots=True)
    df, frame, recurring, singleton = base.primary_data(data_path)
    pooled_contrasts, pooled_trend, pooled_meta, pooled_fit = base.six_band_primary(recurring)
    direct_contrasts, direct_trend, direct_meta, direct_fit = base.strict_six_band(recurring, restrict_to_mixed_strata=False)
    mixed_contrasts, mixed_trend, mixed_meta, mixed_fit = base.strict_six_band(recurring, restrict_to_mixed_strata=True)

    # Multiplicity adjustment is applied to both sets of band-specific contrasts.
    direct_contrasts = direct_contrasts.copy()
    direct_contrasts["Holm_p"] = base.holm_adjust(direct_contrasts["p"])

    counts = direct_support_counts(recurring)
    sparse = sparse_500_diagnostics(recurring)
    perturb = one_event_perturbation(recurring, pooled_fit, direct_fit)
    endpoint = endpoint_heterogeneity_tests(recurring, pooled_fit, direct_fit)
    pooled_reporting_curve, pooled_reporting_meta = base.reporting_weighted_reestimation(recurring)
    direct_reporting_curve, direct_reporting_meta = direct_support_reporting_reestimation(recurring)

    _rename_or_remove_outputs(out)
    save_csv(direct_contrasts, out, "Table_2A_within_band_direct_support_contrasts.csv")
    with (out / "Table_2B_within_band_direct_support_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"all_recurring_records": direct_meta, "mixed_strata_only": mixed_meta}, handle, ensure_ascii=False, indent=2)
    save_csv(pooled_contrasts, out, "Table_2C_pooled_like_event_contrasts.csv")
    with (out / "Table_2D_pooled_like_event_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pooled_meta, handle, ensure_ascii=False, indent=2)
    save_csv(counts, out, "Table_2E_within_band_direct_counts.csv")
    save_csv(sparse, out, "Table_2F_500plus_sparse_diagnostics.csv")
    save_csv(perturb, out, "Table_2G_500plus_one_event_perturbation.csv")

    save_csv(counts, out, "Table_S06D_within_band_direct_support.csv")
    direct_summary = pd.DataFrame([
        {
            "Specification": "Within-band fixed effects, all recurring records",
            "n": direct_meta["n"],
            "Six_band_Wald_chi2": direct_meta["wald_chi2"],
            "Six_band_p": direct_meta["p"],
        },
        {
            "Specification": "Within-band fixed effects, mixed strata only",
            "n": mixed_meta["n"],
            "Six_band_Wald_chi2": mixed_meta["wald_chi2"],
            "Six_band_p": mixed_meta["p"],
        },
    ])
    save_csv(direct_summary, out, "Table_S06E_within_band_direct_support_summary.csv")
    save_csv(direct_contrasts, out, "Table_S07A_within_band_direct_support_contrasts.csv")
    save_csv(pooled_contrasts, out, "Table_S07B_pooled_like_event_contrasts.csv")
    exploratory_trends = pd.concat([
        direct_trend.assign(Model="Within-band fixed effects"),
        mixed_trend.assign(Model="Within-band fixed effects, mixed strata only"),
        pooled_trend.assign(Model="Pooled fixed effects"),
    ], ignore_index=True)
    save_csv(exploratory_trends, out, "Table_S07C_exploratory_ordinal_trends.csv")
    save_csv(endpoint, out, "Table_S09C_endpoint_heterogeneity_tests.csv")
    save_csv(sparse, out, "Table_S09D_500plus_sparse_diagnostics.csv")
    save_csv(perturb, out, "Table_S09E_500plus_one_event_perturbation.csv")
    save_csv(direct_reporting_curve, out, "Table_S15A_direct_support_reporting_reestimation.csv")
    save_csv(pooled_reporting_curve, out, "Table_S15B_pooled_like_event_reporting_reestimation.csv")
    with (out / "Table_S15_reporting_crossings.json").open("w", encoding="utf-8") as handle:
        json.dump({"within_band_direct_support": direct_reporting_meta, "pooled_like_event": pooled_reporting_meta}, handle, ensure_ascii=False, indent=2)
    old_reporting = out / "Table_S15_weighted_reporting_reestimation.csv"
    if old_reporting.exists():
        old_reporting.unlink()

    make_direct_support_figure2(out, direct_contrasts, pooled_contrasts, counts)
    make_reporting_figure(out, direct_reporting_curve, pooled_reporting_curve, direct_reporting_meta, pooled_reporting_meta)

    summary = {
        "source_records": int(len(df)),
        "acute_traumatic_frame": int(len(frame)),
        "recurring_sample": int(len(recurring)),
        "event_configurations": int(recurring["event_cell"].nunique()),
        "inferential_anchor_within_band_direct_support": direct_meta,
        "direct_support_mixed_strata_only": mixed_meta,
        "complementary_pooled_like_event": pooled_meta,
        "exploratory_ordinal_trends": {
            "direct_support_pp_per_band": float(direct_trend.iloc[0]["Coefficient_pp"]),
            "direct_support_p": float(direct_trend.iloc[0]["p"]),
            "pooled_like_event_pp_per_band": float(pooled_trend.iloc[0]["Coefficient_pp"]),
            "pooled_like_event_p": float(pooled_trend.iloc[0]["p"]),
        },
        "within_band_direct_counts": counts.to_dict(orient="records"),
        "endpoint_heterogeneity_tests": endpoint.to_dict(orient="records"),
        "sparse_500_diagnostics": sparse.to_dict(orient="records"),
        "one_event_perturbation": perturb.to_dict(orient="records"),
        "reporting_weighted_crossings": {
            "within_band_direct_support": direct_reporting_meta,
            "pooled_like_event": pooled_reporting_meta,
        },
    }
    assert summary["source_records"] == 37196
    assert summary["acute_traumatic_frame"] == 32700
    assert summary["recurring_sample"] == 28821
    assert abs(summary["inferential_anchor_within_band_direct_support"]["wald_chi2"] - 7.40634) < 0.001
    assert abs(summary["complementary_pooled_like_event"]["wald_chi2"] - 12.03208) < 0.001
    assert summary["within_band_direct_counts"][-1]["Exempt_deaths_mixed"] == 0
    assert summary["within_band_direct_counts"][-1]["Covered_deaths_mixed"] == 4
    assert abs(summary["reporting_weighted_crossings"]["within_band_direct_support"]["under20_simple_contrast_zero_crossing"] - 0.66808) < 0.001
    assert abs(summary["reporting_weighted_crossings"]["pooled_like_event"]["under20_simple_contrast_zero_crossing"] - 0.72977) < 0.001
    with (out / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
