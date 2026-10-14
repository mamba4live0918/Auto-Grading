# Auto-Grading

Tools for assessing *how* a recorded spoken response is delivered, built for an honours thesis on interview-style student recordings. The code turns a recording into a small set of interpretable acoustic indicators, combines them into a reference index for ranking clips, and evaluates the result against teacher total scores and listener judgements. It is meant to help human markers decide what to listen to first, not to replace them.

## What is here

| Path | Contents |
|---|---|
| `src/presentation_assessment/` | The package: audio and video features, transcription helpers, late-fusion scoring, evaluation, and the `research/` pipeline (frozen datasets, provenance, splits, clustering, modelling) |
| `scripts/local_rerun_20261015/` | Analysis scripts behind most of the numbers reported in the thesis: pause detection, fillers and repetitions, the professional-speaker reference, clustering, statistics and the checks on the reference index. The README there lists the inputs and the run order |
| `scripts/` (top level) | Notebook builders and runners, and PowerShell helpers for setting up the environment and processing a video |
| `apps/audio_review/`, `apps/segment_review_20260930/` | Small local web apps used for blind listening and segment review |
| `configs/` | Feature registry, experiment protocol template and fusion configurations |
| `tests/` | Unit and integration tests (synthetic data only) |

## Data

No student recordings, transcripts, scores, listener labels or video links are included. The analysis scripts read private inputs from local folders (`data/private/`, `artifacts/research/`) that are not part of this repository. `data/sample_manifest.csv` is a three-row synthetic example.

## Setup

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

Optional extras: `audio`, `speech` (faster-whisper), `vision` (OpenCV, MediaPipe), `research` (pandas, scikit-learn, notebooks) and `real` (the full audio, speech and vision stack).

## Quick demo

```powershell
python -m presentation_assessment run `
  --manifest data/sample_manifest.csv `
  --config configs/full_multimodal.yaml `
  --output outputs/demo_predictions.csv

python -m presentation_assessment evaluate --predictions outputs/demo_predictions.csv

python -m presentation_assessment ablate `
  --manifest data/sample_manifest.csv `
  --config-dir configs `
  --output-dir outputs/ablations
```

To process a real video, run `.\scripts\setup_real_env.ps1` once, then `.\scripts\process_video.ps1 -Video <path> -Language en`. Outputs go to `outputs/video_runs/<video_name>/`.

`models/haarcascade_frontalface_default.xml` is OpenCV's Haar cascade face detector. It is used as a fallback when the installed OpenCV wheel does not include cascade files.
