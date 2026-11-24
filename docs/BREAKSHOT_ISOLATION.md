# BreakShot isolation experiment

The direct control is the cached BreakShot detector on the original industry clips. Compare the same detector after compact spectral subtraction, Inworld realtime denoising at 20 ms look-ahead, and Inworld offline denoising. Inworld is a reference. [Inworld model card](https://huggingface.co/inworld/voice-isolator/tree/9b0285b3fe8403aa8e2f1dbb2165a80a4294c970)

## Fixed method

- Use the pinned industry-wave-02 manifest, with clean speech, ambient-only, and a 0 dB mixture for each of 11 authored workflows. Verify fixture and detector digests. Require the direct control to reproduce every published event.
- Compare the same sample count and timeline. Keep amplitude scale when converting to PCM16. Resample the Inworld inputs from 16 kHz to 48 kHz and back. These are recorded clips. Offline resampling and file replay do not measure microphone latency.
- Start with the three retail POS and recorder conditions, then replay all 33 when the lane is feasible on the host. Score wake correctness by condition, the original retail miss, and SI-SDR against clean speech on each mixture. Record host wall time, real-time factor, and first returned chunk where applicable.
- Keep a lane experimental unless it rescues the retail mix with no regression on clean speech, quiet clips, or the other mixed clips. Streaming promotion also needs independent site recordings and measured added latency. A perceptual improvement without a detection improvement supports a separate post-wake cleanup experiment.
- The compact candidate is a 16 kHz, 512-sample overlap-add filter with a 256-sample hop and a running noise estimate. Its parameters were fixed before the replay. Startup can absorb speech while the noise estimate is settling.
- Use the existing 464-clip annotated DESED room regression as a separate veto for the compact filter. Require no reduction in detected speech events and no increase in false wakes. DESED has previously been used for BreakShot development. Reserve new operator recordings for field validation. [Dataset and annotations](https://zenodo.org/records/14808180)

## Follow-up probe

The first compact filter improved detected speech and increased false wakes on DESED. Test one fixed confirmation rule. A filtered frame can raise the raw posterior only when the raw score is at or above 0.60, the cached detector's release threshold below its 0.75 onset. Retain raw posterior scores and the unchanged event state machine. Treat this as exploratory on both corpora, since the rule followed the first measurements. Apply the same event-count veto before proposing a detector change.

## Replay

Use the research checkout containing the cached detector and the pinned CPU fixture directory. Install NumPy and ONNX Runtime for the control and compact filter. The Inworld comparisons additionally use `inworld-voice-isolation==1.0.0`, PyTorch, and SciPy. The script verifies both checkpoint hashes and stores the resulting WAV files only when requested.

```powershell
python labs/racklab/isolation_bench.py --detector-checkout D:/MyriadForge --research D:/models/ncs-shave-research --fixtures D:/models/ncs-shave-research/racklab-build/workflow-lab/industry-wave-02/cpu --lane raw --out ./runs/isolation-raw
```

Run once for each lane into a fresh output directory. Add `--workflow store_equipment` for the retail three-clip probe. Add `--save-audio` for listening comparisons.

To try the compact filter alone, use `python labs/racklab/isolation.py input.wav isolated.wav` with a 16 kHz mono PCM16 WAV. This path needs only NumPy.

## Results

The direct BreakShot replay reproduced every recorded event from the 33-clip industry pilot. Both compact variants recovered the retail POS and recorder wake. Both failed the room-audio false-wake gate.

| BreakShot input | Industry wake labels | Room speech events detected | Room false wakes |
| --- | ---: | ---: | ---: |
| Raw | 32/33 | 425/561 | 92 |
| Compact spectral filter | 33/33 | 472/561 | 152 |
| Compact filter with raw-score confirmation | 33/33 | 437/561 | 113 |

The compact filter has zero learned weights. On the three retail clips its median processing cost was about 0.002 seconds per second of audio. The median excludes BreakShot itself. Confirmation uses a second detector pass and requires a separate deployment latency measurement.

Inworld realtime on an RTX 4050 Laptop GPU matched all 33 industry wake labels at a median isolation real-time factor of **1.70×**. Its median mixed-clip SI-SDR was **7.82 dB**, compared with **0.46 dB** after the compact filter and **0.003 dB** on the raw clips. The full GPU replay used FP32 with the same 20 ms look-ahead and checkpoint. The room-audio stress test covered the three compact paths. [Measured results and digests](../labs/racklab/measurements/breakshot-isolation-v1.json)

| Retail reference probe on Core i7-12650H CPU | Wake labels | Mixed clip SI-SDR | Median isolation RTF |
| --- | ---: | ---: | ---: |
| Raw | 2/3 | −0.14 dB | — |
| Compact spectral filter | 3/3 | 1.88 dB | 0.002× |
| Inworld realtime, FP32 | 3/3 | 7.37 dB | 5.40× |
| Inworld offline, FP32 | 3/3 | 10.14 dB | 5.35× |

RTF is wall time divided by audio duration. The Inworld tests use pinned weights and four PyTorch CPU threads. SI-SDR compares the result with the clean synthetic speech track. It measures signal separation and does not establish perceived speech quality or identifier accuracy.

## Next experiment

Keep the raw BreakShot gate. Compare backup isolation after wake on vehicle and part identifiers, work-order readouts, and retail device names. Preserve the raw recording for workplace alarms and review. Use locked operator recordings with fan noise, transients, and competing speakers. Before allowing an isolated signal to trigger a wake, require a paired improvement in event recall without more false wakes on a held-out site. Measure actual push latency and timing alignment through the whole pipeline. Give an NCS1 compatibility lane its own new device test if a learned mask head is introduced.
