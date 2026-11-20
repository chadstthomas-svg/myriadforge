# Recover the booted-state Myriad (PID_F63B): write the missing DeviceInterfaceGUIDs
# REG_MULTI_SZ (winusb.sys reads this one, and the WCID auto-bind left it empty),
# then restart the device. Self-elevates via a temp batch file; approve UAC.

$ErrorActionPreference = "Stop"
$dev = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "*03E7*" } | Select-Object -First 1
if (-not $dev) { Write-Host "[recover] no Movidius device present"; exit 1 }
$id = $dev.InstanceId
Write-Host "[recover] device: $id ($($dev.FriendlyName))"

$bat = Join-Path $env:TEMP "ncs1_recover.bat"
@"
reg add "HKLM\SYSTEM\CurrentControlSet\Enum\$id\Device Parameters" /v DeviceInterfaceGUIDs /t REG_MULTI_SZ /d "{19E08104-0543-40A5-B107-87EF463DCEF1}" /f
pnputil /restart-device "$id"
"@ | Set-Content -Path $bat -Encoding ASCII

$proc = Start-Process -FilePath "cmd.exe" -Verb RunAs -Wait -PassThru -WindowStyle Hidden `
    -ArgumentList "/c", "`"$bat`""
Write-Host "[recover] elevated step exit: $($proc.ExitCode)"

Start-Sleep -Seconds 3
$venvPy = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".venv\Scripts\python.exe"
& $venvPy -c "import sys; sys.path.insert(0, r'D:\NTVStudy\tools\ncs1'); import ncs1_paths; ncs1_paths.setup(); from openvino.inference_engine import IECore; print('[recover] devices:', sorted(IECore().available_devices))"
