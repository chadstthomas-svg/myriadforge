# First physical NCS2 audio replay

Windows detected the NCS1 and NCS2 together as USB `03E7:2150` and `03E7:2485`. OpenVINO 2022.3.1 identified the selected `MYRIAD` target as Intel Movidius Myriad X. Both sticks remained connected during these NCS2 replays. [Intel runtime compatibility](https://www.intel.com/content/www/us/en/support/articles/000091265/software.html)

| Workload | Physical NCS2 result |
| --- | --- |
| MagicEightBall support-policy Gather | 128/128 Boolean assignments in one device request. Median 5.51 ms per warmed batch over five passes, including host preparation. |
| Cached BreakShot raw industry clips | All 33 wake decisions matched CPU. Ground-truth labels matched on 32/33 clips. Event traces matched on 32/33 clips. One retail product-recall interval ended one 512-sample frame later. |
| Cached BreakShot raw DESED room clips | Detected 424/561 annotated speech events and emitted 90 false wakes over 464 recordings. The pinned CPU replay detected 425/561 and emitted 92 false wakes. |

Both BreakShot learned stages ran on Myriad X. The original IR scan failed during graph sorting on OpenVINO 2022.3.1. Direct ONNX compilation ran both stages successfully. In the room replay, stage wall-time p95 was 2.91 ms for the encoder and 4.55 ms for the scan, measured separately and including the host/device call. The audio frontend ran on the host, as did the isolation and transcription experiments. [Measured results and hashes](../labs/racklab/measurements/physical-ncs2-audio-v1.json) · [Room-audio source](https://zenodo.org/records/14808180)

## Replay

Use your OpenVINO 2022.3.1 interpreter and the research checkout containing the cached graphs and clip manifests. Use the normal modern host Python with the existing annotated-audio scorer for DESED aggregation.

```powershell
$ncs2 = 'D:/NTVStudy/tools/ncs2/.venv/Scripts/python.exe'
$research = 'D:/models/ncs-shave-research'
$fixtures = "$research/racklab-build/workflow-lab/industry-wave-02/cpu"
$desed = "$research/breakshot-holdout/desed-room-v1"

& $ncs2 labs/racklab/ncs2_breakshot.py --detector-checkout D:/MyriadForge --research $research --fixtures $fixtures --out ./ncs2-industry.json
& $ncs2 labs/racklab/ncs2_breakshot.py --detector-checkout D:/MyriadForge --research $research --desed $desed --out ./ncs2-room-raw.json
python labs/racklab/ncs2_breakshot.py --research $research --desed $desed --score-input ./ncs2-room-raw.json --out ./ncs2-room-scored.json
```

The verifier defaults to ONNX and requires one cold NCS2. `--scan-device cpu` explicitly runs the scan on the host for diagnosing graph or device differences. Keep host isolation and transcription measurements separate from device timings. A longer soak and independent operator audio are next release checks.
