#!/usr/bin/env python3
"""Main-text figures and Table 1 for the Safety and Health at Work manuscript.

Reads published tables from an output folder (written by run_all.py) and writes:
  Fig1.jpg / Fig1.pdf  crude fatality by scale and coverage; difference-in-differences across adjustment steps
  Fig2.jpg / Fig2.pdf  exempt-minus-covered contrasts by workforce band (primary and pooled models)
  Fig3.jpg / Fig3.pdf  reporting-completeness sensitivity of the fewer-than-20 contrast
  Table1_heterogeneity_and_largest_band.csv
  Text_register_totals_and_worked_example.json  register totals (Section 2.1) and worked example (Section 2.4)
  TableA10_PanelF_within_band_leave_one_out_trend.csv  leave-one-band-out ordered-score trend, within-band model
Figures are 16.5 x 8.0 cm. JPG files are 600 dpi (journal minimum 500 dpi); PDF files are vector with embedded fonts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CM = 1 / 2.54
WIDTH = 16.5 * CM
HEIGHT = 8.0 * CM  # journal size: 16.5 cm wide x 8 cm high
DARK, LIGHT, ACCENT = "#1f4e79", "#9aa5b1", "#c55a11"
STYLE = {  # applied only inside this module's figure functions (never changes global settings)
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "savefig.dpi": 600,
}


def _styled(func):
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
BANDS = ["<20", "20–49", "50–99", "100–299", "300–499", "500+"]


def _read(out: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out / name, encoding="utf-8-sig")


def _save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.jpg", dpi=600, pil_kwargs={"quality": 95})
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


@_styled
def figure1(out: Path, names: dict) -> None:
    comp = _read(out, names["composition"]).set_index("Characteristic")
    ladder = _read(out, names["ladder"])
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    a = fig.add_axes([0.075, 0.175, 0.25, 0.71]); b = fig.add_axes([0.62, 0.175, 0.355, 0.71])
    groups = [("<50 workers", "<50"), ("≥50 workers", "≥50")]
    x = np.arange(len(groups)); w = 0.36
    for j, (status, colour) in enumerate([("Covered", LIGHT), ("Exempt", DARK)]):
        vals, lo, hi = [], [], []
        for _, key in groups:
            n = float(comp.loc["Incidents", f"{key} {status}"])
            pct = float(comp.loc["Fatal incidents (%)", f"{key} {status}"])
            k = int(round(pct * n / 100)); l, h = _wilson(k, int(n))
            vals.append(pct); lo.append(pct - 100 * l); hi.append(100 * h - pct)
        pos = x + (j - 0.5) * w
        a.bar(pos, vals, width=w, color=colour, label=status, edgecolor="black", linewidth=0.4)
        a.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.6, capsize=2)
        for px, v, e in zip(pos, vals, hi):
            a.text(px, v + e + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(x, [g[0] for g in groups]); a.set_ylabel("Fatal incidents (%)")
    a.set_ylim(0, 10.5); a.legend(frameon=False, loc="upper right")
    a.set_title("A. Crude fatality", loc="left"); a.spines[["top", "right"]].set_visible(False)

    labels = ["Crude, all acute traumatic", "Crude, recurring configurations", "+ Event configuration",
              "+ Construction progress", "+ Contract value", "+ Client type", "+ Calendar year"]
    est = ladder["Interaction_pp"].to_numpy(); low = ladder["Interaction_low"].to_numpy(); high = ladder["Interaction_high"].to_numpy()
    y = np.arange(len(labels))[::-1]
    b.hlines(y, low, high, color=DARK, linewidth=1)
    b.plot(est, y, "s", color=DARK, markersize=4)
    for yy, e, h in zip(y, est, high):
        b.text(h + 0.25, yy, f"{e:.2f}", va="center", fontsize=7)
    b.axvline(0, color="black", linewidth=0.6, linestyle="--")
    b.set_yticks(y, labels); b.set_xlim(-0.5, 9.8)
    b.set_xlabel("Difference-in-differences (pp)")
    b.set_title("B. Adjustment steps", loc="left"); b.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "Fig1")


@_styled
def figure2(out: Path, names: dict) -> None:
    direct = _read(out, names["direct"]); pooled = _read(out, names["pooled"]); counts = _read(out, names["counts"])
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0.11, 0.175, 0.55, 0.72]); tx = fig.add_axes([0.68, 0.175, 0.31, 0.72]); tx.axis("off")
    y = np.arange(len(BANDS))[::-1].astype(float)
    for df, dy, marker, colour, label in [(direct, 0.15, "s", DARK, "Within-band model (primary)"), (pooled, -0.15, "o", ACCENT, "Pooled model")]:
        d = df.set_index("Contrast").loc[BANDS]
        ax.hlines(y + dy, d["CI_low_pp"], d["CI_high_pp"], color=colour, linewidth=1)
        ax.plot(d["Estimate_pp"], y + dy, marker, color=colour, markersize=4, label=label, linestyle="none")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_yticks(y, BANDS); ax.set_ylabel("Recorded workforce band")
    ax.set_xlabel("Exempt − covered difference in fatality (pp)")
    ax.set_ylim(-0.6, len(BANDS) - 0.4); ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    c = counts.set_index("Workforce_band").loc[BANDS]
    cols = [("Exempt\nd/n", lambda r: f"{int(r.Exempt_deaths_mixed)}/{int(r.Exempt_incidents_mixed):,}"),
            ("Covered\nd/n", lambda r: f"{int(r.Covered_deaths_mixed)}/{int(r.Covered_incidents_mixed):,}"),
            ("Mixed\nstrata", lambda r: f"{int(r.Mixed_event_configurations):,}")]
    tx.set_ylim(ax.get_ylim()); tx.set_xlim(0, 3)
    for j, (head, fmt) in enumerate(cols):
        tx.text(j + 0.5, len(BANDS) - 0.25, head, ha="center", va="bottom", fontsize=7, fontweight="bold")
        for yy, (_, r) in zip(y, c.iterrows()):
            tx.text(j + 0.5, yy, fmt(r), ha="center", va="center", fontsize=7)
    _save(fig, out, "Fig2")


@_styled
def figure3(out: Path, names: dict) -> None:
    direct = _read(out, names["rep_direct"]); pooled = _read(out, names["rep_pooled"])
    meta = json.loads((out / names["tipping"]).read_text(encoding="utf-8"))
    tips = {"d_local": meta["within_band_direct_support"]["under20_simple_contrast_zero_crossing"],
            "d_cross": meta["within_band_direct_support"]["under20_vs_largest_difference_zero_crossing"],
            "p_local": meta["pooled_like_event"]["under20_simple_contrast_zero_crossing"],
            "p_cross": meta["pooled_like_event"]["under20_vs_largest_difference_zero_crossing"]}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharex=True)
    r = "Relative_nonfatal_completeness_under20_exempt"
    specs = [(axes[0], "Under20_exempt_minus_covered_pp", "Under20_low", "Under20_high", "A. Fewer than 20 workers", "d_local", "p_local"),
             (axes[1], "Under20_minus_largest_contrast_pp", "Difference_low", "Difference_high", "B. Fewer than 20 minus 500 or more", "d_cross", "p_cross")]
    for ax, est, lo, hi, title, td, tp in specs:
        for df, colour, style, marker, label in [(direct, DARK, "-", "s", "Within-band model (primary)"), (pooled, ACCENT, "--", "o", "Pooled model")]:
            q = df.sort_values(r)
            ax.fill_between(q[r], q[lo], q[hi], color=colour, alpha=0.13, linewidth=0)
            ax.plot(q[r], q[est], linestyle=style, color=colour, marker=marker, markevery=5, markersize=3, linewidth=1, label=label)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(tips[td], color=DARK, linewidth=0.8, linestyle=":")
        ax.axvline(tips[tp], color=ACCENT, linewidth=0.8, linestyle=":")
        ax.set_title(title, loc="left"); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Assumed relative reporting completeness (r)")
    axes[0].set_ylabel("Exempt − covered (pp)"); axes[1].set_ylabel("Difference (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.81, wspace=0.28)
    _save(fig, out, "Fig3")


def table1(out: Path, names: dict) -> None:
    g = _read(out, names["global"]); pooled = json.loads((out / names["pooled_global"]).read_text(encoding="utf-8"))
    ep = _read(out, names["endpoint"]).set_index(["Diagnostic", "Model"]); ex = _read(out, names["exact"]).set_index("Comparison")
    oe = _read(out, names["one_event"]).set_index("Model")
    W, P = "Within-band fixed effects", "Pooled fixed effects"
    chi = lambda s, p: f"{s:.2f} (p = {p:.3f})"
    rows = [
        ("A", "Six bands, χ² (5 df)", chi(g.iloc[0]["Six_band_Wald_chi2"], g.iloc[0]["Six_band_p"]), chi(pooled["wald_chi2"], pooled["p"])),
        ("A", "Six bands, mixed strata only, χ² (5 df)", chi(g.iloc[1]["Six_band_Wald_chi2"], g.iloc[1]["Six_band_p"]), "NA"),
        ("A", "Excluding fewer than 20 workers, χ² (4 df)", chi(ep.loc[("Exclude <20", W), "Wald_chi2"], ep.loc[("Exclude <20", W), "p"]), chi(ep.loc[("Exclude <20", P), "Wald_chi2"], ep.loc[("Exclude <20", P), "p"])),
        ("A", "Excluding 500 or more workers, χ² (4 df)", chi(ep.loc[("Exclude 500+", W), "Wald_chi2"], ep.loc[("Exclude 500+", W), "p"]), chi(ep.loc[("Exclude 500+", P), "Wald_chi2"], ep.loc[("Exclude 500+", P), "p"])),
        ("A", "Four middle bands, χ² (3 df)", chi(ep.loc[("Equality across four middle bands", W), "Wald_chi2"], ep.loc[("Equality across four middle bands", W), "p"]), chi(ep.loc[("Equality across four middle bands", P), "Wald_chi2"], ep.loc[("Equality across four middle bands", P), "p"])),
    ]
    m, a = ex.loc["Mixed strata (within-band model)"], ex.loc["All 500+ reports"]
    rd = lambda r: f"{r.Raw_risk_difference_pp:.2f} ({r.Newcombe_CI_low_pp:.2f} to {r.Newcombe_CI_high_pp:.2f})".replace("-", "−")
    rng = lambda r: f"{r.One_nonfatal_to_fatal_min_pp:.2f} to {r.One_nonfatal_to_fatal_max_pp:+.2f}".replace("-", "−")
    rows += [
        ("B", "Exempt fatal/reported incidents", f"{int(m.Exempt_deaths)}/{int(m.Exempt_incidents):,}", f"{int(a.Exempt_deaths)}/{int(a.Exempt_incidents):,}"),
        ("B", "Covered fatal/reported incidents", f"{int(m.Covered_deaths)}/{int(m.Covered_incidents):,}", f"{int(a.Covered_deaths)}/{int(a.Covered_incidents):,}"),
        ("B", "Unadjusted risk difference, pp (95% CI)", rd(m), rd(a)),
        ("B", "Fisher exact p", f"{m.Fisher_exact_p:.3f}", f"{a.Fisher_exact_p:.3f}"),
        ("B", "Adjusted contrast after recoding one exempt nonfatal incident as fatal, pp", rng(oe.loc[W]), rng(oe.loc[P])),
    ]
    pd.DataFrame(rows, columns=["Panel", "Test_or_diagnostic", "Within_band_model_or_mixed_strata", "Pooled_model_or_all_reports"]).to_csv(
        out / "Table1_heterogeneity_and_largest_band.csv", index=False, encoding="utf-8-sig")


def extras(data_path: Path, out: Path) -> None:
    """Outputs computed from the extract: register totals and the worked example cited in the text,
    and the leave-one-band-out ordered-score trend in the within-band model (Table A.10, Panel F)."""
    import analysis_main as engine  # local import: the figure functions above need only published tables

    df, _, recurring, _ = engine.primary_data(data_path)
    _text_numbers(engine, df, recurring, out)
    _within_band_trend_leave_one_out(engine, recurring, out)


def _text_numbers(engine, df: pd.DataFrame, recurring: pd.DataFrame, out: Path) -> None:
    labels = dict(zip(engine.WORKFORCE_ORDER, BANDS))
    config = {"인적사고종류": "떨어짐(2미터 이상 ~ 3미터 미만)", "사고객체(소분류)": "비계", "작업프로세스": "설치작업"}
    q = recurring
    for column, value in config.items():
        q = q[q[column] == value]
    bands = []
    for code in engine.WORKFORCE_ORDER:
        g = q[q["작업자수"] == code]
        cov, exm = g[g["exempt"] == 0], g[g["exempt"] == 1]
        if len(g):
            bands.append({"Workforce_band": labels[code],
                          "Covered_incidents": int(len(cov)), "Covered_fatal_incidents": int(cov["fatal"].sum()),
                          "Exempt_incidents": int(len(exm)), "Exempt_fatal_incidents": int(exm["fatal"].sum()),
                          "Mixed_stratum": bool(len(cov) and len(exm))})
    result = {
        "register": {"Records": int(len(df)), "Persons_died": int(pd.to_numeric(df["사망자"]).sum()),
                     "Persons_injured": int(pd.to_numeric(df["부상자"]).sum()),
                     "First_accident_month": df["date"].min().strftime("%Y-%m"),
                     "Last_accident_month": df["date"].max().strftime("%Y-%m")},
        "worked_example": {"Detailed_accident_type": "Fall from 2 m to less than 3 m (떨어짐(2미터 이상 ~ 3미터 미만))",
                           "Detailed_accident_object": "Scaffolding (비계)", "Work_process": "Installation (설치작업)",
                           "Recurring_sample_incidents": int(len(q)), "By_workforce_band": bands},
    }
    (out / "Text_register_totals_and_worked_example.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _within_band_trend_leave_one_out(engine, recurring: pd.DataFrame, out: Path) -> None:
    """Same specification as the within-band ordered-score trend (analysis_main.strict_six_band), refitted
    without each workforce band in turn; band scores keep their full-sample values (0 = <20 ... 5 = 500+)."""
    core = engine.core
    score_map = {x: i for i, x in enumerate(engine.WORKFORCE_ORDER)}
    rows = []
    for code, label in zip(engine.WORKFORCE_ORDER, BANDS):
        d = recurring[recurring["작업자수"] != code].copy()
        d["event_cell_band"] = d["event_cell"].astype(str) + " || " + d["작업자수"].astype(str)
        d["workforce_score"] = d["작업자수"].map(score_map).astype(float)
        d["score_x_exempt"] = d["workforce_score"] * d["exempt"]
        fit = core.fit_ols(d, ["workforce_score", "exempt", "score_x_exempt"], engine.PRIMARY_CONTROLS,
                           fe_col="event_cell_band", cluster_col="event_cell")
        r = core.coef_row(fit, "score_x_exempt", f"Exclude {label}")
        rows.append({"Excluded_band": label, "n": int(len(d)), "Trend_pp_per_band": r["Coefficient_pp"],
                     "Trend_low": r["CI_low_pp"], "Trend_high": r["CI_high_pp"], "Trend_p": r["p"],
                     "Model": "Within-band fixed effects"})
    pd.DataFrame(rows).to_csv(out / "TableA10_PanelF_within_band_leave_one_out_trend.csv", index=False, encoding="utf-8-sig")


def make_all(out: Path, names: dict) -> None:
    figure1(out, names); figure2(out, names); figure3(out, names); table1(out, names)
