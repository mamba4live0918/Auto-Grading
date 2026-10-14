param(
  [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

py -$PythonVersion -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -U pip
.\.venv311\Scripts\python.exe -m pip install -e ".[dev,real]"

Write-Host "Real video environment ready: .\.venv311"
Write-Host "Use: .\.venv311\Scripts\python.exe -m presentation_assessment process-video --video <path>"
