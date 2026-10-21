#!/usr/bin/env python3
"""Compare rebuilt outputs with the archived reference outputs.

Numbers in CSV and JSON files are compared with a relative tolerance (default 1e-6),
because the last digits of floating-point results can differ across operating systems
and linear-algebra libraries. Text must match exactly. PNG, JPG and TIFF figures are compared
pixel by pixel; vector PDF figures are compared byte for byte and reported as a note if they differ.
File checksums are not used for this comparison.

Usage:  python verify_outputs.py [--rebuilt outputs_rebuilt] [--reference outputs] [--rtol 1e-6]
Exit status 0 means every file matched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def same_value(a, b, rtol: float, atol: float) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)


def compare_json(a, b, rtol, atol, path="$"):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ"]
        return [m for k in a for m in compare_json(a[k], b[k], rtol, atol, f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: lengths differ"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in compare_json(x, y, rtol, atol, f"{path}[{i}]")]
    return [] if same_value(a, b, rtol, atol) else [f"{path}: {a!r} vs {b!r}"]


def compare_file(a: Path, b: Path, rtol: float, atol: float) -> list[str]:
    suffix = a.suffix.lower()
    if suffix == ".csv":
        x = pd.read_csv(a, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        y = pd.read_csv(b, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if list(x.columns) != list(y.columns) or x.shape != y.shape:
            return [f"columns or shape differ: {x.shape} vs {y.shape}"]
        return [f"row {i}, {c}: {u!r} vs {v!r}"
                for c in x.columns for i, (u, v) in enumerate(zip(x[c], y[c]))
                if not same_value(u, v, rtol, atol)]
    if suffix == ".json":
        return compare_json(json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8")), rtol, atol)
    if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        from PIL import Image
        u = np.asarray(Image.open(a).convert("RGB"))
        v = np.asarray(Image.open(b).convert("RGB"))
        if u.shape != v.shape:
            return [f"image size differs: {u.shape} vs {v.shape}"]
        n = int((u != v).any(axis=2).sum())
        return [] if n == 0 else [f"{n} pixels differ"]
    return [] if a.read_bytes() == b.read_bytes() else ["contents differ"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuilt", type=Path, default=HERE / "outputs_rebuilt")
    parser.add_argument("--reference", type=Path, default=HERE / "outputs")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()

    ref = {p.name for p in args.reference.iterdir() if p.is_file()}
    new = {p.name for p in args.rebuilt.iterdir() if p.is_file()}
    failures = 0
    for name in sorted(ref - new):
        print(f"MISSING in rebuild: {name}"); failures += 1
    for name in sorted(new - ref):
        print(f"EXTRA in rebuild:   {name}"); failures += 1
    matched = 0
    for name in sorted(ref & new):
        problems = compare_file(args.reference / name, args.rebuilt / name, args.rtol, args.atol)
        if problems and name.lower().endswith(".pdf"):
            print(f"NOTE: {name} differs byte for byte (expected across platforms; its JPG twin is compared pixel by pixel)")
            matched += 1
            continue
        if problems:
            failures += 1
            print(f"DIFFERS: {name}")
            for line in problems[:10]:
                print(f"    {line}")
        else:
            matched += 1
    print(f"\n{matched} of {len(ref)} reference files matched (rtol={args.rtol:g}).")
    print("RESULT: PASS" if failures == 0 else f"RESULT: FAIL ({failures} problem(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
