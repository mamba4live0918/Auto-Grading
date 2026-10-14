"""Flatten the local per-recording JSON into the two CSV tables the 9/29 scripts read, with the same columns and
row order as the 9/26 files (data/private/acoustic_redo_20260926/features_private.csv and
data/private/reference_grouping_20260926/professional_features.csv), then report how far the values moved.
Run: .venv311/Scripts/python.exe scripts/local_rerun_20261015/make_tables_local.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

import localpaths as LP

OLD_STUDENTS = LP.CLOUD_FEATURES
OLD_PROS = LP.PRIV / "reference_grouping_20260926" / "professional_features.csv"


def fmt(v) -> str:
    return "" if v is None else repr(float(v)) if isinstance(v, float) else str(v)


def students() -> None:
    with open(OLD_STUDENTS, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        old = list(reader)
    rows = []
    for r in old:
        d = json.load(open(LP.OUT / f"{r['sample_id']}.json"))
        row = {}
        for col in header:
            if col == "sample_id":
                row[col] = r["sample_id"]
            elif "__" in col:
                part, key = col.split("__", 1)
                row[col] = fmt((d.get(part) or {}).get(key))
            else:
                row[col] = fmt(d.get(col))
        rows.append(row)
    with open(LP.RUN / "features_private.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    report("students", old, rows, header[1:])   # header[1] is sha256: the audio must be the same file


def professionals() -> None:
    with open(OLD_PROS, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        old = list(reader)
    split = {c["speaker_hash"]: c["split"] for c in json.load(open(LP.REF_MANIFEST, encoding="utf-8"))["clips"]}
    rows = []
    for r in old:
        sid = r["speaker_hash"]
        d = json.load(open(LP.REF_OUT / f"{sid}.json"))
        row = {"speaker_hash": sid, "split": split[sid], "duration_used_s": fmt(d["duration_s"])}
        for col in header[3:]:
            row[col] = fmt(d["whole"][col])
        rows.append(row)
    with open(LP.RUN / "professional_features.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    report("professionals", old, rows, header[2:])


def report(label: str, old: list[dict], new: list[dict], cols: list[str]) -> None:
    worst, n_diff = 0.0, 0
    for a, b in zip(old, new):
        for c in cols:
            if c == "sha256":
                assert a[c] == b[c], (a.get("sample_id"), c)
                continue
            if a[c] == "" or b[c] == "":
                assert a[c] == b[c], (label, c, a[c], b[c])
                continue
            dv = abs(float(a[c]) - float(b[c]))
            if dv > 1e-9:
                n_diff += 1
                worst = max(worst, dv / max(abs(float(a[c])), 1e-9))
    print(f"{label}: {len(new)} rows; {n_diff} cells differ from 9/26, largest relative difference {worst:.2g}")


if __name__ == "__main__":
    students()
    professionals()
