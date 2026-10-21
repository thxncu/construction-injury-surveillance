#!/usr/bin/env python3
"""Redraw all figures from published table outputs without re-estimating the models.

Redraws Figs. 1-3 of the manuscript (shaw_outputs.py) and Figs. A.1-A.3 of Appendix A
in an output folder (default: outputs_rebuilt). run_all.py already does this; use this
script only to redraw figures.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd  # noqa: E402

import analysis_main as base  # noqa: E402
import shaw_outputs  # noqa: E402
from run_all import engine_to_published, figure_inputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs_rebuilt")
    args = parser.parse_args()
    out = args.output.resolve()
    if out == (HERE / "outputs").resolve():
        parser.error("outputs/ holds the archived reference outputs and is never overwritten.")
    pub = engine_to_published()

    def table(engine_name: str) -> pd.DataFrame:
        return pd.read_csv(out / pub[engine_name], encoding="utf-8-sig")

    counts = table("Table_2E_within_band_direct_counts.csv")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        base.make_figures(
            stage, table("Table_2C_pooled_like_event_contrasts.csv"), table("Table_2A_within_band_direct_support_contrasts.csv"),
            counts[["Workforce_band", "Mixed_event_configurations"]].copy(),
            table("Table_S15A_direct_support_reporting_reestimation.csv"),
            table("Table_S18A_fall_height_fatality.csv"), table("Table_S18B_fall_height_composition.csv"),
            table("Table_S18C_fall_height_standardization.csv"), table("Table_S17B_type_risk_differences.csv"),
        )
        for engine_name, published_name in pub.items():
            if engine_name.endswith(".png") and (stage / engine_name).exists():
                shutil.copy2(stage / engine_name, out / published_name)
    shaw_outputs.make_all(out, figure_inputs())
    print(f"Figures redrawn in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
