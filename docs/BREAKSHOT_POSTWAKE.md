# BreakShot post-wake transcription probe

## Fixed evaluation

The original audio stays the wake authority and remains available for alarms and review. Compare raw, compact, and Inworld-isolated copies through the same frozen Whisper Tiny English checkpoint. The hypothesis is that isolation can preserve more critical fields without losing a field that the raw transcript already got right. Require a net gain and no raw-correct field losses before proposing an always-on cleanup default.

- Replay the 11 original 0 dB mixes with reviewed expected phrases, plus nine new CueBall prompts containing specific identifiers. The extra identifiers include a recall campaign, bulletin, part, work order, lift, SKU, return model, and POS device. Record the clean synthetic control for each clip to expose limitations of the synthetic voice or ASR checkpoint.
- Use BreakShot's event on the raw audio and common 400 ms pre-roll for every lane. Score inputs with no raw wake in a separate forced-capture group. The cleaner sees the original prefix before the common ASR crop, so it can estimate background noise using the same available context. This is a recorded replay, with no live ASR latency claim.
- Hold the clip, raw wake timeline, ASR crop, tokenizer and decoding settings fixed across lanes. Record exact field matches and losses against raw, plus word error rate. Treat case, punctuation and approved numeric speech variants as equivalent. Keep digit order exact.
- The new prompts use synthesized CueBall speech with previously licensed ambient recordings, mixed at 0 dB. The identifiers are authored probes. Keep field promotion dependent on independent operator recordings.

[Pinned Whisper Tiny English weights](https://huggingface.co/openai/whisper-tiny.en/tree/87c7102498dcde7456f24cfd30239ca606ed9063) · [Prior isolation experiment](BREAKSHOT_ISOLATION.md)

## Capture-length diagnostic

After inspecting the 400 ms run, repeat the same comparison with a fixed 2,500 ms pre-roll. This retains the start of these clips on the observed wake timeline. If the clean control improves, capture truncation explains part of the field loss. Remaining misses on the full clean capture belong to the synthesis and transcription chain. Keep the wake decision itself fixed. This follow-up is exploratory because its buffer was chosen after inspecting the first replay.

## Reproduce

Use the cached BreakShot graphs and the local CueBall comparison kit with its Whisper manifest. Set `PYTHONPATH` to include the pinned `inworld-voice-isolation==1.0.0` installation or install the package in the active environment.

```powershell
$root = 'D:/models/ncs-shave-research'
$fixtures = "$root/racklab-build/workflow-lab/industry-wave-02/cpu"
$voice = "$root/cueball-hf-release/continuous/onnx"
$espeak = "$root/voice-to-voice/kokoro/.venv/Lib/site-packages/espeakng_loader"
python labs/racklab/postwake_asr.py --detector-checkout D:/MyriadForge --research $root --fixtures $fixtures --whisper-kit "$root/cueball/comparison" --cueball $voice --espeak $espeak --inworld cuda --pre-roll-ms 400 --out ./runs/postwake-400ms
```

Repeat with `--pre-roll-ms 2500` and a fresh output path for the capture diagnostic. The pre-roll is measured before the estimated speech onset. A live buffer also needs space for the delay between the onset and the emitted wake. The script uses actual samples from the replay and retains raw wake decisions. [Physical NCS2 detector check](NCS2_AUDIO_REPLAY.md)

## Results

The two runs used identical clean and mixed PCM samples and reproduced the same raw events for all 20 clips. Nineteen clips woke on raw BreakShot; the other clip was the original retail POS and recorder miss. The table counts 29 expected fields on the clips that actually woke.

| Matched capture | Raw fields | Compact fields | Inworld realtime fields | Clean control |
| --- | ---: | ---: | ---: | ---: |
| 400 ms pre-roll | 2/29 | 2/29 | 1/29 | 8/29 |
| 2,500 ms pre-roll | 3/29 | 2/29 | 0/29 | 10/29 |

At 400 ms, Inworld recovered one field and lost both fields already correct in raw audio. Its output recovered the register-support phrase on forced capture of the original retail miss. That clip remained outside the raw-wake score. The compact filter tied raw field counts at 400 ms and lost one raw-correct field at 2,500 ms.

Longer pre-roll preserved the start of all 20 clips, including a register ID previously cut off. Even then, the clean control recovered only ten fields on the 19 clips with a wake. On those clips the compact lane's word error rose to 119.8%, driven by extra words. The field gate rejects both cleanup variants for transcription with this voice and ASR pair. Keep raw audio as the transcription input and preserve isolated audio as an optional listening copy. The next comparison needs operator recordings and a stronger frozen ASR baseline before revisiting the cleanup decision. [Scorecard, transcripts, and artifact digests](../labs/racklab/measurements/breakshot-postwake-v1.json)
