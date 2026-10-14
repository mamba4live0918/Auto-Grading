param(
  [Parameter(Mandatory = $true)]
  [string]$Video,

  [string]$OutputDir = "outputs/video_runs",
  [string]$Config = "configs/full_multimodal.yaml",
  [string]$ModelSize = "large-v3",
  [string]$Language = "",
  [string]$Device = "auto",
  [string]$ComputeType = "auto"
)

$ErrorActionPreference = "Stop"
$python = ".\.venv311\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Missing .venv311. Run .\scripts\setup_real_env.ps1 first."
}

$argsList = @(
  "-m", "presentation_assessment", "process-video",
  "--video", $Video,
  "--output-dir", $OutputDir,
  "--config", $Config,
  "--model-size", $ModelSize,
  "--device", $Device,
  "--compute-type", $ComputeType
)

if ($Language.Trim().Length -gt 0) {
  $argsList += @("--language", $Language)
}

& $python @argsList
