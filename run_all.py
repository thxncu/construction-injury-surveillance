#!/usr/bin/env python3
"""Run the complete replication workflow in one deterministic process (v1.2.0).

1. The analysis engine writes its outputs to a temporary staging folder.
2. Engine outputs are published under the manuscript labels in OUTPUT_MAP.csv
   (Tables A.1-A.20 and Figs. A.1-A.3 of Appendix A).
3. shaw_outputs.py draws Figs. 1-3 and assembles Table 1 of the Safety and Health at Work manuscript
   from the published tables, and writes the register totals and worked-example counts cited in the text
   and the within-band leave-one-band-out trend (Table A.10, Panel F).

The archived reference outputs in outputs/ are never overwritten. Compare a rebuild with them using
verify_outputs.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def read_output_map() -> list[dict[str, str]]:
    with (HERE / "OUTPUT_MAP.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def engine_to_published() -> dict[str, str]:
    return {r["engine_file"]: r["published_file"] for r in read_output_map() if r["source"] == "engine" and r["published_file"]}


def figure_inputs() -> dict[str, str]:
    """Published files read by shaw_outputs.py."""
    p = engine_to_published()
    return {
        "composition": p["Table_S05_project_composition_four_groups.csv"],
        "ladder": p["Table_S06C_binary_model_ladder.csv"],
        "direct": p["Table_2A_within_band_direct_support_contrasts.csv"],
        "pooled": p["Table_2C_pooled_like_event_contrasts.csv"],
        "counts": p["Table_2E_within_band_direct_counts.csv"],
        "rep_direct": p["Table_S15A_direct_support_reporting_reestimation.csv"],
        "rep_pooled": p["Table_S15B_pooled_like_event_reporting_reestimation.csv"],
        "tipping": p["Table_S15_reporting_crossings.json"],
        "global": p["Table_S06E_within_band_direct_support_summary.csv"],
        "pooled_global": p["Table_2D_pooled_like_event_summary.json"],
        "endpoint": p["Table_S09C_endpoint_heterogeneity_tests.csv"],
        "exact": p["Table_S09D_500plus_sparse_diagnostics.csv"],
        "one_event": p["Table_S09E_500plus_one_event_perturbation.csv"],
    }


def publish(stage: Path, out: Path) -> None:
    rows = [r for r in read_output_map() if r["source"] == "engine"]
    produced = {p.name for p in stage.iterdir()}
    expected = {r["engine_file"] for r in rows}
    unmapped, missing = produced - expected, expected - produced
    if unmapped or missing:
        raise RuntimeError(f"Engine outputs do not match OUTPUT_MAP.csv. Unmapped: {sorted(unmapped)}; missing: {sorted(missing)}")
    for r in rows:
        if r["published_file"]:
            shutil.copy2(stage / r["engine_file"], out / r["published_file"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "CSI_analytic_deidentified.csv")
    parser.add_argument("--output", type=Path, default=HERE / "outputs_rebuilt")
    args = parser.parse_args()

    out = args.output.resolve()
    if out == (HERE / "outputs").resolve():
        parser.error("outputs/ holds the archived reference outputs and is never overwritten; choose another --output.")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    import analysis_rehierarchy as analysis
    import shaw_outputs  # uses a local plotting style; does not alter engine figures

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        summary = analysis.run_pipeline(args.data.resolve(), stage)
        publish(stage, out)

    shaw_outputs.make_all(out, figure_inputs())
    shaw_outputs.extras(args.data.resolve(), out)
    expected = {r["published_file"] for r in read_output_map() if r["source"] == "shaw_outputs.py"}
    missing = sorted(f for f in expected if not (out / f).exists())
    if missing:
        raise RuntimeError(f"shaw_outputs.py did not produce: {missing}")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nOutputs written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
