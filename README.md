# Replication package (version 1.2.0)

**Safety-management-plan coverage, site scale and fatal outcomes in reported construction incidents: evidence from South Korea**

*Review copy: author names are withheld for double-anonymized peer review and will be restored on publication.*

Manuscript prepared for *Safety and Health at Work*. This package (Appendix B of the manuscript) reproduces every
number, table, and figure in the manuscript and in Appendix A from a de-identified extract of the Korean
Construction Safety Management Integrated Information (CSI) register. The outcome is whether a reported acute
traumatic construction incident was fatal. The analyses do not estimate accident incidence, total fatality risk,
plan implementation, or a causal effect of statutory coverage.

## Quick start

Requires Python 3.11 or later (tested with Python 3.13.16 on Linux). A full run takes under a minute.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
python run_all.py                    # writes outputs_rebuilt/
python verify_outputs.py             # compares outputs_rebuilt/ with the archived outputs/
```

`verify_outputs.py` prints `RESULT: PASS` when every rebuilt file matches the archived reference. Numbers are
compared with a relative tolerance of 1e-6 because the final digits of floating-point results vary across
operating systems and linear-algebra libraries; raster figures (PNG, JPG) are compared pixel by pixel; vector
PDFs are compared byte for byte and reported as a note if they differ, because their JPG twins are already
checked. `MANIFEST_SHA256.txt` checks that the package files arrived intact; it is not used to compare a rebuild.

## Contents

| File | Purpose |
|---|---|
| `CSI_analytic_deidentified.csv` | De-identified analytic extract, UTF-8 (37,196 records, 23 columns) |
| `VALUE_LABELS.csv` | English labels for every coded value in the extract |
| `run_all.py` | One-command workflow: runs the engine, publishes outputs under manuscript labels, draws Figs. 1-3 and Table 1 |
| `shaw_outputs.py` | Draws Figs. 1-3 (16.5 x 8 cm; 600-dpi JPG and vector PDF), assembles Table 1, and writes the register totals and worked-example counts cited in the text and Table A.10 Panel F |
| `verify_outputs.py` | Tolerance-based comparison of a rebuild with `outputs/` |
| `make_figures.py` | Redraws all figures from published tables without re-estimating models |
| `analysis_core.py`, `analysis_main.py`, `analysis_rehierarchy.py` | Analysis engine (data preparation, estimators, tables, Appendix A figures) |
| `OUTPUT_MAP.csv` | Source, engine file name, published file name, and manuscript location of every output |
| `outputs/` | Archived reference outputs (never overwritten by the scripts) |
| `requirements-lock.txt` | Exact package versions used to create `outputs/` |
| `requirements.txt` | Compatible version ranges |
| `DATA_PRIVACY_NOTE.md`, `SOURCE_PROVENANCE.md` | Disclosure control and data source |
| `LICENSE`, `LICENSE-DATA.md`, `CITATION.cff` | Code license (MIT), data and output license (CC BY 4.0), citation |
| `CHANGELOG.md` | Version history |
| `SOURCE_DATA_SHA256.txt`, `MANIFEST_SHA256.txt` | Integrity checksums |

## Where each manuscript item comes from

| Manuscript item | Output files |
|---|---|
| Fig. 1 | `Fig1.jpg` (submission file), `Fig1.pdf`; drawn from `TableA7_project_composition.csv` (panel A) and `TableA8_PanelC_model_ladder.csv` (panel B) |
| Fig. 2 | `Fig2.jpg`, `Fig2.pdf`; drawn from `TableA9_PanelA_within_band_contrasts.csv`, `TableA9_PanelB_pooled_contrasts.csv`, `TableA9_PanelA_overlap_counts.csv` |
| Fig. 3 | `Fig3.jpg`, `Fig3.pdf`; drawn from `TableA11_PanelA_reporting_within_band.csv`, `TableA11_pooled_reporting_curve.csv`, `TableA11_PanelB_tipping_points.json` |
| Table 1 | `Table1_heterogeneity_and_largest_band.csv`; assembled from `TableA8_PanelE_global_tests.csv`, `TableA9_PanelB_global_test.json`, `TableA10_PanelC/D/E_*` |
| Tables A.2-A.20 (Appendix A) | `TableA2_*` to `TableA20_*` (Table A.1 is conceptual and has no numerical output) |
| Figs. A.1-A.3 (Appendix A) | `FigA1_conceptual_framework.png`, `FigA2_accident_type_differences.png`, `FigA3_fall_height.png` |
| Sections 2.1 and 2.4 (register totals; worked example) | `Text_register_totals_and_worked_example.json` |

Panel D of Table A.8 repeats the overlap counts in Panel A of Table A.9 and therefore uses
`TableA9_PanelA_overlap_counts.csv`.

In the output tables, columns named `Deaths` or `*_deaths*` count fatal incidents (incidents with at least one
death), not persons; Appendix A labels them "Fatal incidents" or uses d/n. Only the extract column `사망자`
counts persons who died, as do `Persons_died` and `Persons_injured` in `Text_register_totals_and_worked_example.json`.

## Analytic design

The analytic sample contains 28,821 reported acute traumatic incidents in 3,493 recurring event configurations
(detailed accident type x detailed accident object x work process).

- **Crude-to-adjusted decomposition (Fig. 1, Table A.8 Panel C).** The exemption-by-small-site interaction
  (fewer than 50 workers) is estimated first without fixed effects or controls, then with event-configuration
  fixed effects, and then with construction progress, contract value, client type, and calendar year added in turn.
- **Within-band fixed-effects model (primary; Fig. 2).** A linear probability model that absorbs an
  event-configuration-by-workforce-band fixed effect. Each band's exempt-minus-covered contrast is identified only
  from mixed strata, that is, strata containing both covered and exempt incidents. Standard errors are clustered by
  event configuration.
- **Pooled fixed-effects model (complementary; Fig. 2).** Workforce-band indicators and one set of
  event-configuration fixed effects shared across bands.
- **Reporting-completeness sensitivity (Fig. 3).** Nonfatal incidents in the exempt fewer-than-20 group receive
  frequency weight 1/r for r from 0.20 to 1.00; the tipping points are the values of r at which the fewer-than-20
  contrast, and its difference from the 500-or-more contrast, are zero.

The analysis engine keeps the internal names used during development: "direct support" and "within-band
direct-support" refer to the within-band fixed-effects model; "like-event" and "pooled like-event" refer to the
pooled fixed-effects model.

## Interpretation boundaries

- The analysis conditions on entry into the administrative register and on recurring event configurations.
- The primary model's global test does not reject equal contrasts across the six workforce bands. An exploratory
  ordered-score trend is negative (Table A.9 Panel C) but loses conventional significance when either endpoint band
  is removed (Table A.10 Panel A, pooled model; Panel F, within-band model).
- The pooled model's heterogeneity depends on both endpoint bands; the four middle-band contrasts are statistically indistinguishable.
- In the 500-or-more band, mixed strata contain 0 deaths among 246 exempt reports and 4 deaths among 755 covered reports, in only three outcome-varying strata.
- Exact 2-by-2 comparisons and one-event perturbations are sparse-data diagnostics, not replacements for the adjusted models.
- The workforce field may reflect accident-day staffing, planned workforce, typical or maximum staffing, or a reporting approximation.
- Reporting tipping points are r = 0.668 (local under-20 contrast) and r = 0.477 (under-20 minus 500-or-more) in the primary model, and r = 0.730 and r = 0.428 in the pooled model. They are sensitivity thresholds, not estimates of reporting completeness.
- Contractor capability, exposure denominators, plan implementation, emergency-response intervals, and a stable project identifier are not available.

## License and citation

Code: MIT (`LICENSE`). Extract, value labels, and outputs: CC BY 4.0 (`LICENSE-DATA.md`). Citation metadata: `CITATION.cff`.
