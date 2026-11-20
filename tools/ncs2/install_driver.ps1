# NCS2 driver contact (roadmap "blocked on hardware" item 1): install the
# WinUSB function-driver INF for VID_03E7&PID_2150/2485 and re-probe OpenVINO.
#
# The INF install requires admin (pnputil); this script self-elevates, so
# approve the UAC prompt when it appears. Unsigned INF is fine here: the
# USBDevice device class is signature-exempt and WinUSB is an in-box driver.
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools\ncs2\install_driver.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$inf = Join-Path $scriptDir "driver\ncs2_winusb.inf"

if (-not (Test-Path $inf)) { throw "INF not found: $inf" }

# --- 1) install the INF (elevated) ---
$pnputilArgs = "/add-driver `"$inf`" /install"
$proc = Start-Process -FilePath "pnputil.exe" -ArgumentList $pnputilArgs `
    -Verb RunAs -Wait -PassThru -WindowStyle Hidden
if ($proc.ExitCode -ne 0) {
    Write-Warning "pnputil exited $($proc.ExitCode); if this was a UAC decline or an"
    Write-Warning "access error, install manually from an admin prompt:"
    Write-Warning "  pnputil /add-driver `"$inf`" /install"
}

# --- 2) re-enumerate: does Windows now bind WinUSB? ---
$dev = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "*03E7*" }
if ($dev) {
    $dev | ForEach-Object { Write-Host "[driver] $($_.Status) | $($_.Class) | $($_.FriendlyName) | $($_.InstanceId)" }
} else {
    Write-Warning "[driver] no 03E7 device enumerated - is the stick plugged in?"
}

# --- 3) OpenVINO probe (venv python) ---
$venvPy = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    & $venvPy -c "from openvino.runtime import Core; import json; devs = sorted(Core().available_devices); print('[driver] OpenVINO devices: ' + json.dumps(devs))"
}

# Re-run the probe once more after a short settle delay if MYRIAD is still
# absent (device re-enumeration after driver bind can lag a beat).
