"""Attach a complete QC-score file to a frozen Outcome recording manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest, scores = read_rows(args.manifest), read_rows(args.scores)
    keyed = {}
    for row in scores:
        key = row["filename"]
        if key in keyed:
            raise ValueError(f"Duplicate QC score filename: {key}")
        value = row.get("quality_score_q", "")
        if value == "":
            raise ValueError(f"Missing QC score: {key}")
        keyed[key] = value
    output = []
    for row in manifest:
        key = Path(row.get("server_source_path") or row["source_path"]).name
        if key not in keyed:
            raise ValueError(f"No QC score for manifest recording: {key}")
        output.append({**row, "quality_score_q": keyed[key]})
    if len(keyed) != len(output):
        extra = sorted(set(keyed) - {Path(row.get("server_source_path") or row["source_path"]).name for row in manifest})
        raise ValueError(f"QC-score/manifest cardinality mismatch; extra scores: {extra[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*manifest[0].keys(), "quality_score_q"])
        writer.writeheader(); writer.writerows(output)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
