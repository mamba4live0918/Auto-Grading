$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskPythonPath = Join-Path $taskProjectRoot '.venv311\Scripts\python.exe'
$taskServerPath = Join-Path $PSScriptRoot 'server.py'
$taskReviewUrl = 'http://127.0.0.1:8874/'
$taskAlreadyListening = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8874 -State Listen -ErrorAction SilentlyContinue
if (-not $taskAlreadyListening) {
    $taskServerArgs = @('-u', ('"{0}"' -f $taskServerPath), '--port', '8874')
    Start-Process -FilePath $taskPythonPath -ArgumentList $taskServerArgs -WorkingDirectory $taskProjectRoot -WindowStyle Hidden
}
$taskReady = $false
for ($taskAttempt = 0; $taskAttempt -lt 30; $taskAttempt++) {
    try {
        $taskServiceState = Invoke-RestMethod -Uri ($taskReviewUrl + 'api/state') -TimeoutSec 1
        if ($taskServiceState.study -ne 'segment-pregrade-20260930') { throw 'Port 8874 belongs to another service.' }
        $taskReady = $true
        break
    } catch {
        Start-Sleep -Milliseconds 200
    }
}
if (-not $taskReady) { throw 'The local review service could not be started. Check port 8874 and the saved pregrade data.' }
Start-Process $taskReviewUrl
