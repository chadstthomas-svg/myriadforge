# Recover a wedged Myriad device: restart it at the USB bus level (equivalent to
# unplug/replug without touching the cable), then re-probe OpenVINO.
#
# The restart requires admin - this script self-elevates; approve the UAC prompt.
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools\ncs1\reset_device.ps1

$ErrorActionPreference = "Stop"
$dev = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "*VID_03E7*" }
if (-not $dev) { Write-Host "[reset] no Movidius device enumerated - is the stick plugged in?"; exit 1 }
$id = $dev.InstanceId
Write-Host "[reset] restarting $id (self-elevating - approve UAC)..."

$proc = Start-Process -FilePath "pnputil.exe" `
    -ArgumentList "/restart-device `"$id`"" -Verb RunAs -Wait -PassThru -WindowStyle Hidden
if ($proc.ExitCode -ne 0) {
    Write-Warning "[reset] pnputil restart exited $($proc.ExitCode). Alternatives:"
    Write-Warning "  - run an admin terminal yourself: pnputil /restart-device `"$id`""
    Write-Warning "  - or physically unplug and replug the stick"
}

Start-Sleep -Seconds 2
$venvPy = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    & $venvPy -c "from openvino.runtime import Core; import json; print('[reset] OpenVINO devices:', json.dumps(sorted(Core().available_devices)))"
}
