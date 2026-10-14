"""Local paths for the 2026-10-15 local re-run of the analyses first run on 2026-09-26..29.

Every input is a file on this computer; every output goes to new folders, so the earlier results
(data/private/*_2026092x, output/*_2026092x) are not touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIV = ROOT / "data" / "private"

# parselmouth (Praat 6.1.38) lives in the .venv-asr environment (same Python 3.11)
_ASR_SITE = ROOT / ".venv-asr" / "Lib" / "site-packages"
if str(_ASR_SITE) not in sys.path:
    sys.path.append(str(_ASR_SITE))

# ---- outputs (new) ----
RUN = PRIV / "local_rerun_20261015"            # private, per-recording
OUT = RUN / "out"                               # the cloud scripts' "out/" folder
PRAAT_DIR = OUT / "praat"
EGEMAPS_DIR = OUT / "egemaps"
REF_OUT = RUN / "ref" / "out"                   # professional speakers, first 180 s
REF_OUT_FULL = RUN / "ref" / "out_full"         # professional speakers, full length
PUBLIC = ROOT / "output" / "local_rerun_20261015"
REF180_WAV = Path("D:/ThesisLocalRerun_20261015/ref180_wav")   # professional audio cut to 180 s (public corpus)

# ---- inputs (existing, local) ----
_FROZEN = ROOT / ".worktrees" / "europarl-500" / "artifacts" / "research" / \
    "444eb952b40469ad9e3dfe8983ea7461335f89f23ca0ed99de465d20d55c10ca"
MAN = _FROZEN / "manifest.csv"                  # frozen manifest with the teacher total scores
TR_CHECK = _FROZEN / "transcripts.jsonl"        # transcript run used by the 9/26 ASR syllable check
TR_A = ROOT / "artifacts" / "research" / "a4a7e64ba59438f3cc8dbfb1174e4b6a680db93c4bc6ae4e97925c8e814ca4fe" / "transcripts.jsonl"
STUDENT_WAV_DIR = ROOT / "artifacts" / "research" / "staging" / "notebook-01-local-v1" / "work"
LISTEN_KEY = PRIV / "listening_20260926" / "listening_key_DO_NOT_OPEN.csv"
LISTEN_LABELS = PRIV / "listening_20260926" / "听辨结果_2026-09-28.csv"
PC_KEY_CLOUD = PRIV / "pause_check_20260928" / "pause_check_key_DO_NOT_OPEN.json"   # clip windows (+ old program pauses)
PC_KEY = RUN / "pause_check_key_local.json"     # same windows, program pauses re-measured locally
PC_LABELS = PRIV / "pause_check_20260928" / "卡顿标注_2026-09-28.csv"
REF_WAV_DIR = Path("D:/ThesisReferenceData/europarl_asr_v1.0/selected_audio.v2")
REF_MANIFEST = REF_WAV_DIR / "selection.manifest.json"
CLOUD_FEATURES = PRIV / "acoustic_redo_20260926" / "features_private.csv"   # only for the before/after comparison


def student_wav(sid: str) -> Path:
    """The 16 kHz mono PCM16 analysis.wav made locally on 2026-08-25..27 (byte-identical to the 9/26 input)."""
    return STUDENT_WAV_DIR / sid / "audio" / "analysis.wav"


for _d in (OUT, PRAAT_DIR, EGEMAPS_DIR, REF_OUT, REF_OUT_FULL, PUBLIC):
    _d.mkdir(parents=True, exist_ok=True)
