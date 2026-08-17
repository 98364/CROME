"""Download, verify, profile, and preprocess UCI Online Retail II."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import urllib.request
import zipfile

from crome_identification.benchmarks.online_retail import (
    load_workbook_rows,
    preprocess_rows,
    save_processed,
    sha256_file,
)


URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DOI = "https://doi.org/10.24432/C5CG6D"
LICENSE = "CC BY 4.0"


def prepare(root: Path, *, min_customer_events: int = 5) -> dict:
    root = Path(root)
    raw_dir = root / "data" / "raw" / "online_retail_ii"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "online_retail_ii.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        candidates = [name for name in bundle.namelist() if name.lower().endswith(".xlsx")]
        if len(candidates) != 1:
            raise ValueError(f"expected one XLSX in UCI archive, found {candidates}")
        member = candidates[0]
        workbook = raw_dir / Path(member).name
        if not workbook.exists():
            bundle.extract(member, raw_dir)
            extracted = raw_dir / member
            if extracted != workbook:
                extracted.replace(workbook)
    rows, sheets = load_workbook_rows(workbook)
    events, profile = preprocess_rows(
        rows, quantity_threshold=20.0, min_customer_events=min_customer_events
    )
    profile.update(
        {
            "source_url": URL,
            "doi": DOI,
            "license": LICENSE,
            "retrieved": date.today().isoformat(),
            "archive_sha256": sha256_file(archive),
            "workbook_sha256": sha256_file(workbook),
            "workbook_file": workbook.name,
            "sheet_names": sheets,
            "selection_rule": "all customers with nonmissing ID and at least min_customer_events",
            "outcome_usage": "none; original outcomes/prices are not benchmark responses",
        }
    )
    npz_path = processed_dir / "online_retail_ii_events.npz"
    profile_path = processed_dir / "online_retail_ii_profile.json"
    save_processed(events, profile, npz_path=npz_path, profile_path=profile_path)
    profile["processed_npz_sha256"] = sha256_file(npz_path)
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--min-customer-events", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, min_customer_events=args.min_customer_events), indent=2))


if __name__ == "__main__":
    main()
