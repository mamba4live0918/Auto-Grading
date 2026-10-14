# Local Model Assets

- `haarcascade_frontalface_default.xml`: OpenCV Haar cascade face detector used
  as a fallback when the installed OpenCV wheel does not ship cascade data.

This is only a lightweight baseline detector. For thesis-quality visual
engagement experiments, prefer OpenFace or MediaPipe/Tasks-based face landmarks
when the dataset is ready.
