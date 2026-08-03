#!/usr/bin/env python3
"""Driver for timing image::cropImage and image::maskImage.

Generates square source BMPs (256/512/768/1024) as top-left crops of demo.bmp,
1-bit masks via mask.py, runs the `bench` binary over the crop grid and per
resolution for mask, and writes results to CSV + prints a summary table.

Crop grid (top-left crop to each smaller grid resolution):
  256:   no crop
  512  -> 256
  768  -> 256, 512
  1024 -> 256, 512, 768
Mask: one measurement per resolution (source R x R + mask R x R).
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
BENCH = ROOT / "build" / "bin" / "bench"
DEMO_BMP = HERE.parent / "demo.bmp"
MASK_PY = HERE.parent / "mask.py"
WORKDIR = ROOT / "build" / "bench"
RESULTS_CSV = HERE / "bench_results.csv"

RESOLUTIONS = [256, 512, 768, 1024]
GRID = [256, 512, 768, 1024]
TRIALS = 10

RESULT_RE = re.compile(
    r"RESULT\s+op=(?P<op>\S+)\s+src=(?P<src>\d+)\s+\S+?=(?P<target>\d+)\s+trials=(?P<trials>\d+)\s+"
    r"avg_ms=(?P<avg>[\d.]+)\s+min_ms=(?P<min>[\d.]+)\s+max_ms=(?P<max>[\d.]+)"
)


def generate_sources() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    with Image.open(DEMO_BMP) as img:
        for r in RESOLUTIONS:
            out = WORKDIR / f"src_{r}.bmp"
            if out.exists():
                continue
            cropped = img.crop((0, 0, r, r))
            cropped.save(out, "BMP")
            print(f"  wrote {out.name} ({r}x{r})")


def generate_masks() -> None:
    for r in RESOLUTIONS:
        out = WORKDIR / f"mask_{r}.bmp"
        if out.exists():
            continue
        subprocess.run(
            ["python3", str(MASK_PY), str(WORKDIR / f"src_{r}.bmp"),
             "--out-width", str(r), "--out-height", str(r), "-o", str(out)],
            check=True, stdout=subprocess.DEVNULL,
        )
        print(f"  wrote {out.name} ({r}x{r})")


def run_bench(args: list[str]) -> dict:
    cmd = [str(BENCH)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = None
    for ln in proc.stdout.splitlines():
        if ln.startswith("RESULT"):
            line = ln
            break
    if line is None:
        raise RuntimeError(f"no RESULT line from: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    m = RESULT_RE.match(line.strip())
    if not m:
        raise RuntimeError(f"unparsable RESULT line: {line!r}")
    return {
        "op": m["op"],
        "src_res": int(m["src"]),
        "target_res": int(m["target"]),
        "trials": int(m["trials"]),
        "avg_ms": float(m["avg"]),
        "min_ms": float(m["min"]),
        "max_ms": float(m["max"]),
    }


def crop_cases() -> list[list[str]]:
    cases = []
    for r in RESOLUTIONS:
        for t in GRID:
            if t < r:
                cases.append(["--op=crop", f"--input={WORKDIR / f'src_{r}.bmp'}",
                              f"--crop-width={t}", f"--crop-height={t}",
                              f"--trials={TRIALS}"])
    return cases


def mask_cases() -> list[list[str]]:
    cases = []
    for r in RESOLUTIONS:
        cases.append(["--op=mask", f"--input={WORKDIR / f'src_{r}.bmp'}",
                      f"--mask={WORKDIR / f'mask_{r}.bmp'}", f"--trials={TRIALS}"])
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regen", action="store_true", help="Regenerate source/mask images even if present")
    args = ap.parse_args()

    if not BENCH.exists():
        print(f"error: {BENCH} not found; build the `bench` target first", file=sys.stderr)
        return 1
    if not DEMO_BMP.exists():
        print(f"error: {DEMO_BMP} not found", file=sys.stderr)
        return 1

    if args.regen:
        for f in WORKDIR.glob("*.bmp"):
            f.unlink()

    print("Generating source images...")
    generate_sources()
    print("Generating masks...")
    generate_masks()

    rows = []
    print("\nRunning crop cases...")
    for case in crop_cases():
        print("  " + " ".join(case))
        rows.append(run_bench(case))

    print("Running mask cases...")
    for case in mask_cases():
        print("  " + " ".join(case))
        rows.append(run_bench(case))

    rows.sort(key=lambda d: (d["op"], d["src_res"], d["target_res"]))
    with RESULTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["op", "src_res", "target_res", "avg_ms", "min_ms", "max_ms", "trials"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {RESULTS_CSV}")

    print("\n| op    | src_res | target_res | avg_ms | min_ms | max_ms | trials |")
    print("|-------|---------|------------|--------|--------|--------|--------|")
    for d in rows:
        print(f"| {d['op']} | {d['src_res']} | {d['target_res']} | {d['avg_ms']:.4f} | {d['min_ms']:.4f} | {d['max_ms']:.4f} | {d['trials']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
