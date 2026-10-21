# Data privacy note (v1.2.0)

`CSI_analytic_deidentified.csv` is derived from an export of the Construction Safety Management
Integrated Information (CSI) system. It contains structured project and incident fields only. It
contains no names, contact details, resident-registration numbers, addresses, free-text narratives,
or worker demographic fields. A pattern scan found no strings matching email-address, Korean
telephone-number, or Korean resident-registration-number formats.

## Disclosure-control steps (introduced in v1.1.0; the extract is unchanged in v1.2.0)

| Step | What was done | Effect on reported results |
|---|---|---|
| Row order | Rows were randomly reordered and `record_id` was reassigned (the source order was chronological). | None (no estimator depends on row order beyond floating-point rounding). |
| Time of day | Removed. | None (time of day is not used). |
| Day of month | Replaced by a random permutation of the days of the same calendar month. Records that shared an accident date still share one; year and month are exact. The permutation seed was not recorded. | None (year, month, accident-date clusters, the February 2021 restriction, and calendar periods are preserved). |
| Gross floor area | Rounded to two significant figures. | None at reported precision (Spearman correlation with the workforce field 0.7031 before and after). |

## Residual risk

Uniqueness was assessed on a combination of fields an outside party could plausibly know:
calendar month, facility middle category, construction type, above- and below-ground storeys,
gross floor area, and contract-value band.

| Version | Linkable date precision | All records unique | Fatal records unique |
|---|---|---|---|
| v1.0.0 | Minute | 99.2% | 1,554 of 1,556 (99.9%) |
| v1.1.0 | Month | 61.9% (23,013 of 37,196) | 1,511 of 1,556 (97.1%) |

Fatal construction incidents are rare within a month, so a few project attributes still single out
most of them. Coarsening the storey counts further would change reported estimates (Table A.13, Panel B
in Appendix A). The fields describe projects and incidents, not individual workers, and many fatal
incidents are also publicly reported. Users must not attempt to identify any project, site,
contractor, or person, and must follow the source provider's access and reuse terms.
