# WS0 toolchain pin - isolated OpenVINO 2022.3.1 LTS venv for the NCS2 lane.
#
# Creates tools/ncs2/.venv (Python 3.10) and installs the pinned OpenVINO
# stack. OpenVINO 2023.0 removed the MYRIAD plugin, so 2022.3.1 is the last
# LTS with Neural Compute Stick support; the myriadforge main env never imports
# OpenVINO - everything in this venv is reached via tools/ncs2/bridge_runner.py
# and the apps/ pilots. Native Windows only - the MYRIAD driver story is
# Windows-native; do not route this lane through WSL2.
#
# Usage (from repo root, any shell):
#   powershell -ExecutionPolicy Bypass -File tools\ncs2\setup_env.ps1
#
# Idempotent: skips venv creation when .venv exists; pip install is safe to
# re-run. Absence of MYRIAD in the final probe is expected and logged, not
# fatal (the dongle driver may still need installing - see README lane table).

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # tools/ncs2
$repoRoot  = Split-Path -Parent (Split-Path -Parent $scriptDir)
$venv      = Join-Path $scriptDir ".venv"
$venvPy    = Join-Path $venv "Scripts\python.exe"

# --- 1) interpreter: py -3.10 if present, else a uv-managed standalone 3.10 ---
$py310 = $null
try {
    $out = & py -3.10 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) { $py310 = "$out".Trim() }
} catch { }

if ($py310) {
    Write-Host "[setup] interpreter: py -3.10 -> $py310"
} else {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "No 'py -3.10' and no 'uv' found. Install Python 3.10 or uv, then re-run."
    }
    Write-Host "[setup] no py -3.10; using uv-managed standalone CPython 3.10"
    & uv python install 3.10
    if ($LASTEXITCODE -ne 0) { throw "uv python install 3.10 failed" }
}

# --- 2) venv (skip when it already exists) ---
if (Test-Path $venvPy) {
    Write-Host "[setup] venv already present: $venv"
} elseif ($py310) {
    & $py310 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    & uv venv --seed --python 3.10 $venv
    if ($LASTEXITCODE -ne 0) { throw "uv venv creation failed" }
}

# --- 3) pinned packages ---
# numpy 1.x is load-bearing and openvino 2022.3.1 caps it at <1.25, so 1.24.4
# is the floor for the whole venv (onnxruntime/opencv still resolve fine).
$pins = @(
    "numpy==1.24.4",
    "openvino==2022.3.1",
    "openvino-dev[onnx]==2022.3.1",   # Model Optimizer (mo CLI / openvino.tools.mo)
    "onnxruntime==1.17.3",
    "opencv-python==4.10.0.84",
    "pillow==10.4.0"
)
Write-Host "[setup] installing pinned packages: $($pins -join ' ')"
& $venvPy -m pip install --disable-pip-version-check @pins
if ($LASTEXITCODE -ne 0) {
    Write-Warning "PyPI install failed. Fallback: download the 2022.3.1 LTS wheels from"
    Write-Warning "https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3.1/ and"
    Write-Warning "pip-install them into $venv manually."
    throw "pip install failed"
}

# --- 4) MYRIAD plugin fallback ---
# The PyPI openvino==2022.3.1 wheel does NOT ship the MYRIAD plugin (its
# plugins.xml references it, but the DLL is absent) and neither wheel nor
# pip carries the NCS2 USB driver. Pull the plugin + usb-ma2x8x firmware
# from the official 2022.3.1 LTS archive on storage.openvinotoolkit.org.
$libs = Join-Path $venv "Lib\site-packages\openvino\libs"
if (-not (Test-Path (Join-Path $libs "openvino_intel_myriad_plugin.dll"))) {
    Write-Host "[setup] PyPI wheel lacks MYRIAD plugin - fetching from the 2022.3.1 LTS archive"
    $archDir = Join-Path $scriptDir ".archive"
    New-Item -ItemType Directory -Force -Path $archDir | Out-Null
    $zipPath = Join-Path $archDir "w_openvino_toolkit_windows_2022.3.1.9227.cf2c7da5689_x86_64.zip"
    $zipUrl  = "https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3.1/windows/w_openvino_toolkit_windows_2022.3.1.9227.cf2c7da5689_x86_64.zip"
    if (-not (Test-Path $zipPath)) {
        Write-Host "[setup] downloading ~118 MB archive..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        if ($LASTEXITCODE -ne 0) { throw "archive download failed" }
    }
    $root = "w_openvino_toolkit_windows_2022.3.1.9227.cf2c7da5689_x86_64/"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        foreach ($pair in @(
            @("${root}runtime/bin/intel64/Release/openvino_intel_myriad_plugin.dll", "$libs\openvino_intel_myriad_plugin.dll"),
            @("${root}runtime/bin/intel64/Release/usb-ma2x8x.mvcmd", "$libs\usb-ma2x8x.mvcmd"),
            @("${root}runtime/bin/intel64/Release/usb-ma2x8x.mvcmd", "$libs\mvnc\usb-ma2x8x.mvcmd")
        )) {
            $entry = $zip.Entries | Where-Object { $_.FullName -eq $pair[0] }
            $dest = $pair[1]
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
            Write-Host "[setup] extracted -> $dest"
        }
    } finally { $zip.Dispose() }
}

# --- 5) env sanity probe (roadmap WS0 acceptance) ---
Write-Host "[setup] probing OpenVINO runtime..."
& $venvPy -c "from openvino.runtime import Core; import json; devs = sorted(Core().available_devices); print('available devices: ' + json.dumps(devs)); print('MYRIAD present: ' + str(any(d.startswith('MYRIAD') for d in devs)))"
if ($LASTEXITCODE -ne 0) { throw "OpenVINO runtime probe failed" }

Write-Host "[setup] OK - venv ready at $venv"
