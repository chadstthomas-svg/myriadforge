# NCS1 (Myriad 2) toolchain pin - isolated OpenVINO 2020.3.2 venv, Python 3.7.
#
# The original Neural Compute Stick (Myriad 2 MA2450, USB PID 0x2150) is only
# supported through OpenVINO 2020.3.x: the MYRIAD boot table disabled Myriad 2
# from 2020.4 ("deprecated since 2020.4"). 2020.3 uses the older IECore API and
# ships cp37 Python bindings only, hence Python 3.7 (not the ncs2 lane's 3.10).
#
# This script is idempotent and captures every bring-up gotcha discovered on
# hardware arrival (2026-11-19):
#   * Python 3.7.9 from python.org (uv has no 3.7 standalone build).
#   * OpenVINO 2020.3.2 dev archive (runtime + Model Optimizer + ngraph + the
#     usb-ma2450.mvcmd Myriad 2 firmware). The PyPI wheel does not exist for
#     2020.3 and would lack the MYRIAD plugin anyway.
#   * Intel C++ runtime DLLs (libmmd.dll, svml_dispmd.dll) from the
#     intel-cmplr-lib-rt pip wheel - inference_engine_legacy.dll / myriadPlugin.dll
#     import them and Intel does NOT bundle them in the OpenVINO archive.
#   * numpy pinned to 1.19.5 (cp37 ie_api ABI); MO ONNX deps; opencv/pillow for apps.
#
# Native Windows only - the MYRIAD driver story is Windows-native (WinUSB).
# The WinUSB function driver for VID_03E7&PID_2150 is installed separately via
# tools/ncs2/install_driver.ps1 (the INF covers PID_2150 and PID_2485).
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools\ncs1\setup_env.ps1

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # tools/ncs1
$archDir   = Join-Path $scriptDir ".archive"
$py37Dir   = Join-Path $scriptDir "python37"
$py37      = Join-Path $py37Dir "python.exe"
$venv      = Join-Path $scriptDir ".venv"
$venvPy    = Join-Path $venv "Scripts\python.exe"
$ovDir     = Join-Path $scriptDir "ov2020"
$redist    = Join-Path $scriptDir "redist"
New-Item -ItemType Directory -Force -Path $archDir | Out-Null

function Download($url, $dest) {
    if (Test-Path $dest) { Write-Host "[setup] cached: $dest"; return }
    Write-Host "[setup] downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
}

# --- 1) Python 3.7.9 (per-user silent install, no admin) ---
if (-not (Test-Path $py37)) {
    $installer = Join-Path $archDir "python-3.7.9-amd64.exe"
    Download "https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe" $installer
    Write-Host "[setup] installing Python 3.7.9 -> $py37Dir"
    $p = Start-Process -FilePath $installer -Wait -PassThru -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_test=0",
        "Include_launcher=0", "Include_doc=0", "TargetDir=$py37Dir")
    if ($p.ExitCode -ne 0) { throw "Python 3.7.9 install failed (exit $($p.ExitCode))" }
} else { Write-Host "[setup] Python 3.7 present: $py37" }

# --- 2) venv ---
if (-not (Test-Path $venvPy)) {
    & $py37 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else { Write-Host "[setup] venv present: $venv" }

# --- 3) OpenVINO 2020.3.2 dev archive -> ov2020 ---
$ieMarker = Join-Path $ovDir "deployment_tools\inference_engine\bin\intel64\Release\myriadPlugin.dll"
if (-not (Test-Path $ieMarker)) {
    $zip = Join-Path $archDir "w_openvino_toolkit_dev_p_2020.3.355.zip"
    Download "https://storage.openvinotoolkit.org/repositories/openvino/packages/2020.3.2/w_openvino_toolkit_dev_p_2020.3.355.zip" $zip
    & $py37 (Join-Path $scriptDir "_extract_ov.py") $zip $ovDir
    if ($LASTEXITCODE -ne 0) { throw "OpenVINO archive extraction failed" }
} else { Write-Host "[setup] OpenVINO 2020.3.2 runtime present" }

# --- 4) Intel C++ runtime DLLs (libmmd / svml_dispmd) ---
if (-not (Test-Path (Join-Path $redist "libmmd.dll"))) {
    $dlDir = Join-Path $archDir "redist_dl"
    & $venvPy -m pip download intel-cmplr-lib-rt --no-deps -d $dlDir --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "intel-cmplr-lib-rt download failed" }
    $wheel = Get-ChildItem $dlDir -Filter "intel_cmplr_lib_rt-*.whl" | Select-Object -First 1
    & $venvPy (Join-Path $scriptDir "_extract_redist.py") $wheel.FullName $redist
    if ($LASTEXITCODE -ne 0) { throw "Intel runtime extraction failed" }
} else { Write-Host "[setup] Intel C++ runtime DLLs present" }

# --- 5) pinned venv packages ---
# numpy 1.19.5 matches the cp37 ie_api ABI; onnx 1.7 / IR6 for MO 2020.3;
# opencv 4.6 / pillow 9.5 are the last py3.7-friendly lines.
$pins = @(
    "numpy==1.19.5", "onnx==1.7.0", "protobuf==3.19.6", "networkx==2.5",
    "defusedxml==0.7.1", "onnxruntime==1.10.0", "typing-extensions", "six",
    "opencv-python==4.6.0.66", "pillow==9.5.0"
)
Write-Host "[setup] installing pinned packages"
& $venvPy -m pip install --disable-pip-version-check @pins
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# --- 6) env sanity probe (MYRIAD absence logged, not fatal) ---
Write-Host "[setup] probing OpenVINO 2020.3.2 runtime..."
& $venvPy (Join-Path $scriptDir "probe_myriad.py")
Write-Host "[setup] OK - NCS1 venv ready at $venv"
