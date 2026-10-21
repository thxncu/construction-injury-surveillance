# Changelog

## 1.2.0 (2026-10-06)

Release accompanying the *Safety and Health at Work* manuscript. The analytic extract, the estimators, and every
reported number are unchanged from version 1.1.0.

- Outputs are published under the manuscript labels: Figs. 1-3 and Table 1 of the main text, and Tables A.2-A.20
  and Figs. A.1-A.3 of Appendix A (`OUTPUT_MAP.csv`, which lists the source, engine file name, published file name,
  and manuscript location of every output).
- Added `shaw_outputs.py`, which draws Figs. 1-3 (16.5 x 8 cm; 600-dpi JPG submission files and vector PDFs),
  assembles Table 1 from the published tables, and writes the register totals and worked-example counts cited in
  Sections 2.1 and 2.4 (`Text_register_totals_and_worked_example.json`).
- Added the leave-one-band-out ordered-score trend for the within-band model (Table A.10 Panel F,
  `TableA10_PanelF_within_band_leave_one_out_trend.csv`); Panel A reports the same analysis for the pooled model.
- Fig. A.3: the legend of panel A now sits above the bars instead of covering them.
- `verify_outputs.py` compares numbers within a relative tolerance of 1e-6 and raster figures pixel by pixel, and
  reports byte-level differences in vector PDFs as notes.
- `make_figures.py` redraws the Appendix A figures and Figs. 1-3 from published tables.

## 1.1.0 (2026-10-06)

Development release. Every number in the version 1.0.0 outputs is reproduced within a relative tolerance of 1e-6,
except the two corrections listed below. Model names in output tables and figure labels were changed to
"within-band fixed effects" and "pooled fixed effects".

### Data and disclosure control
- Extract converted from CP949 to UTF-8 (with byte-order mark).
- Rows randomly reordered and `record_id` reassigned; time of day removed; day of month pseudonymized
  within each calendar month; gross floor area rounded to two significant figures (see `DATA_PRIVACY_NOTE.md`).
- Added `VALUE_LABELS.csv` with English labels for all 402 coded values.

### Corrections
- Calendar periods (now Table A.17) include the whole final day of each period. Version 1.0.0 compared
  timestamps with midnight of the end date, which omitted incidents later on 31 December 2021, 31 December 2023,
  and 30 June 2025. Under-20 incidents change from 4,179 to 4,182 (July 2019-December 2021) and from 3,039 to
  3,041 (January 2024-June 2025); fatal incidents are unchanged.
- The workforce-field evidence (now Table A.6) is computed from the data; version 1.0.0 wrote fixed strings. The
  rank correlations reproduce (0.74 with contract value, 0.70 with gross floor area). The progress pattern is
  described from the computed profile (`TableA6_progress_profile.csv`): among the 11 contract-value bands with at
  least 500 records, the mean workforce score is lowest in the first progress decile in 8, peaks at 40-69%
  progress in 7, and is below its peak in the final decile in 9.

### Outputs and workflow
- `run_all.py` writes to `outputs_rebuilt/` by default and never overwrites the archived `outputs/`.
- Added `verify_outputs.py`. Byte-identical duplicate outputs are no longer published.
- `requirements-lock.txt` pins the full environment, including indirect dependencies.

### Documentation
- `data_dictionary.csv` covers all 23 columns (version 1.0.0 covered 17).
- Added `LICENSE` (MIT, code), `LICENSE-DATA.md` (CC BY 4.0, extract and outputs), and `CITATION.cff`.
- Fixed a label-dependent condition in `sparse_500_diagnostics` so that strata counts no longer depend on the
  display label.

## 1.0.0 (2026-10-06)

Initial package.
