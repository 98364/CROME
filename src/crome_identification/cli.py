"""CLI entry: crome-exp <name> [--mode smoke|dev|main]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crome-exp", description="Run CROME synthetic experiments")
    parser.add_argument(
        "experiment",
        choices=[
            "exp01",
            "exp02",
            "exp03",
            "exp04",
            "exp05",
            "exp05b",
            "exp06",
            "exp07",
            "cs01",
            "cs02",
            "all-smoke",
        ],
        help="Experiment id (see README)",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "dev", "main"],
        default="smoke",
        help="Monte Carlo size preset (default: smoke)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Results directory (default: results/raw)",
    )
    args = parser.parse_args(argv)

    # Late imports so --help is fast
    from experiments import (
        exp01_causal_weighting,
        exp02_support_gap,
        exp03_async_phase,
        exp04_partial_id,
        exp05_overlap_functionals,
        exp05b_mark_information,
        exp06_regularization_nullspace,
        exp07_perturbations,
        cs01_end_to_end_certification,
        cs02_split_calibrated,
    )

    outdir = args.outdir or (Path.cwd() / "results" / "raw")
    outdir.mkdir(parents=True, exist_ok=True)

    runners = {
        "exp01": exp01_causal_weighting.run,
        "exp02": exp02_support_gap.run,
        "exp03": exp03_async_phase.run,
        "exp04": exp04_partial_id.run,
        "exp05": exp05_overlap_functionals.run,
        "exp05b": exp05b_mark_information.run,
        "exp06": exp06_regularization_nullspace.run,
        "exp07": exp07_perturbations.run,
        "cs01": cs01_end_to_end_certification.run,
        "cs02": cs02_split_calibrated.run,
    }

    if args.experiment == "all-smoke":
        summary = {}
        for name, fn in runners.items():
            summary[name] = fn(mode="smoke", outdir=outdir)
        path = outdir / "all_smoke_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {path}")
        return 0

    result = runners[args.experiment](mode=args.mode, outdir=outdir)
    console_summary = dict(result)
    console_summary.pop("replication_records", None)
    print(json.dumps(console_summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
