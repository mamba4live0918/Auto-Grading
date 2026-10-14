"""Station 7, validation step 1 (plan: docs/research/2026-09-29-verbatim-plan.md, section 1).

Does the verbatim transcript find the fillers and repetitions the listener marked in the 24 P clips?
Pass line (frozen in the plan): uh recall >= 0.40. The earlier Whisper large-v3 transcript is scored the same way
as a baseline. Also runs the plan's secondary check 5 (filler totals, word-count ratio to the earlier transcript).

Run: .venv311\\Scripts\\python.exe scripts\\acoustic_redo\\verbatim_check.py
Numbers without recording ids go to output/verbatim_20260929/check1.json; per-clip rows stay in data/private/.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import verbatim as VB

ROOT = Path(__file__).resolve().parents[2]
PRIV = ROOT / "data" / "private"
NEW = PRIV / "verbatim_20260929" / "verbatim_transcripts.jsonl"
OLD = ROOT / "artifacts/research/a4a7e64ba59438f3cc8dbfb1174e4b6a680db93c4bc6ae4e97925c8e814ca4fe/transcripts.jsonl"
KEY = PRIV / "pause_check_20260928" / "pause_check_key_DO_NOT_OPEN.json"
LABELS = PRIV / "pause_check_20260928" / "卡顿标注_2026-09-28.csv"
OUT = ROOT / "output" / "verbatim_20260929"
TOL = 0.3
PASS_UH_RECALL = 0.40
RATIO_LOW, RATIO_HIGH = 0.7, 1.5
WINDOW_S = 30.0
FROZEN = {"clips": 24, "uh": 16, "repeat": 12}  # what the plan and the listening session fixed


def load_new_records(path):
    """{sample_id: record} for the succeeded records of the verbatim run (the last one per recording wins)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "succeeded":
                out[r["sample_id"]] = r
    return out


def words_of_record(record):
    return [(w["start"], w["end"], w["word"]) for w in record["words"]]


def load_old(path):
    """{sample_id: [(start, end, word)]} from the earlier Whisper large-v3 transcripts (words without times are skipped)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") == "succeeded" and r.get("transcription"):
            out[r["sample_id"]] = [(float(w["start"]), float(w["end"]), w["word"])
                                   for s in r["transcription"]["segments"] for w in s.get("words", [])
                                   if w["start"] is not None and w["end"] is not None]
    return out


def load_human(path):
    """{clip_id: {"uh": [(a, b)], "repeat": [(a, b)]}} in seconds from the clip start."""
    human = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["type"] in ("uh", "repeat"):
            human.setdefault(r["clip_id"], {}).setdefault(r["type"], []).append((float(r["start_s"]), float(r["end_s"])))
    return human


def validate_inputs(key, human):
    """Refuse to score when the clip key or the listener's labels differ from what the plan froze."""
    if len(key) != FROZEN["clips"]:
        raise ValueError(f"expected {FROZEN['clips']} clips, the key has {len(key)}")
    for k in key:
        if abs((k["end_s"] - k["start_s"]) - WINDOW_S) > 0.01:
            raise ValueError(f"clip {k['clip_id']}: window is not {WINDOW_S:.0f} s long")
    unknown = set(human) - {k["clip_id"] for k in key}
    if unknown:
        raise ValueError(f"label clip ids {sorted(unknown)} are not in the key")
    for kind in ("uh", "repeat"):
        n = sum(len(v.get(kind, [])) for v in human.values())
        if n != FROZEN[kind]:
            raise ValueError(f"{n} human {kind} marks, the plan froze {FROZEN[kind]}")


def check_provenance(records):
    """All records must come from one model commit and one set of settings (no mixing after a resumed run)."""
    combos = {(r.get("revision"), json.dumps(r.get("settings"), sort_keys=True)) for r in records}
    if len(combos) > 1:
        raise ValueError(f"mixed model revisions or settings among the records ({len(combos)} combinations)")


def summarize(rows, kind):
    matched = sum(r[kind]["matched"] for r in rows)
    detected = sum(r[kind]["detected"] for r in rows)
    marked = sum(r[kind]["human"] for r in rows)
    recall = matched / marked if marked else None
    precision = matched / detected if detected else None
    counts = ([r[kind]["detected"] for r in rows], [r[kind]["human"] for r in rows])
    rho = float(spearmanr(*counts)[0]) if len(set(counts[0])) > 1 and len(set(counts[1])) > 1 else None
    return {"human": marked, "detected": detected, "matched": matched,
            "recall": None if recall is None else round(recall, 3),
            "precision": None if precision is None else round(precision, 3),
            "clip_spearman_detected_vs_human": None if rho is None else round(rho, 3)}


def uh_passed(summary):
    """The frozen pass line, applied to the exact recall (not the rounded one)."""
    return summary["human"] > 0 and summary["matched"] / summary["human"] >= PASS_UH_RECALL


def score_all(words_by_id, key, human):
    return [{"clip_id": k["clip_id"], **VB.score_clip(words_by_id[k["sample_id"]], (k["start_s"], k["end_s"]),
                                                       human.get(k["clip_id"], {}), TOL)} for k in key]


def word_count_ratios(new, old):
    return {sid: len(new[sid]) / len(old[sid]) for sid in new if sid in old and old[sid]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    key = json.load(open(KEY, encoding="utf-8"))
    human = load_human(LABELS)
    validate_inputs(key, human)
    records, old = load_new_records(NEW), load_old(OLD)
    check_provenance(list(records.values()))
    new = {sid: words_of_record(r) for sid, r in records.items()}
    missing = sorted({k["sample_id"] for k in key} - set(new))
    if missing:
        raise SystemExit(f"{len(missing)} of the 24 clip recordings have no verbatim transcript yet")

    rows_new, rows_old = score_all(new, key, human), score_all(old, key, human)
    result = {"tolerance_s": TOL, "n_clips": len(key), "pass_line_uh_recall": PASS_UH_RECALL,
              "verbatim": {k: summarize(rows_new, k) for k in ("uh", "repeat")},
              "earlier_whisper_baseline": {k: summarize(rows_old, k) for k in ("uh", "repeat")}}
    result["uh_passed"] = uh_passed(result["verbatim"]["uh"])

    common = sorted(set(new) & set(old))
    ratios = word_count_ratios({s: new[s] for s in common}, old)
    if not ratios:
        raise SystemExit("no recordings in common between the two transcripts")
    vals = np.array(list(ratios.values()))
    flagged = sorted(sid for sid, r in ratios.items() if r < RATIO_LOW or r > RATIO_HIGH)
    result["secondary_check_5"] = {
        "recordings_compared": len(common),
        "fillers": {"earlier_whisper": sum(len(VB.filler_spans(old[s])) for s in common),
                    "verbatim": sum(len(VB.filler_spans(new[s])) for s in common)},
        "repetitions": {"earlier_whisper": sum(len(VB.repetition_spans(old[s])) for s in common),
                        "verbatim": sum(len(VB.repetition_spans(new[s])) for s in common)},
        "word_ratio_verbatim_over_whisper": {"median": round(float(np.median(vals)), 3), "min": round(float(vals.min()), 3),
                                             "max": round(float(vals.max()), 3)},
        "recordings_outside_ratio_band": len(flagged), "ratio_band": [RATIO_LOW, RATIO_HIGH]}
    result["source_sha256_mismatch_recordings"] = sum(1 for r in records.values() if not r.get("source_sha256_match"))
    result["recordings_transcribed"] = len(records)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(OUT / "check1.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({"clips": rows_new, "clips_earlier_whisper": rows_old, "flagged_recordings": flagged},
              open(PRIV / "verbatim_20260929" / "check1_rows.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    recall = result["verbatim"]["uh"]["recall"]
    print("UH PASS LINE:", "PASSED" if result["uh_passed"] else "NOT PASSED", f"(recall {recall} vs >= {PASS_UH_RECALL})")


if __name__ == "__main__":
    main()
