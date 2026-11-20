# Custom models for Myriad 2

This lab includes trained ONNX weights, deployable OpenVINO FP16 XML/BIN pairs and held-out evaluation records for the original Intel NCS1. The host prepares inputs and interprets scores. All learned layers in the four FP16 bundles ran through `IECore.load_network(..., device_name="MYRIAD")` on the MA2450. The separate Needle bundle is a native Cactus model with its own Windows host evaluation.

[MagicEightBall](boolean/README.md) returns a yes/no answer, a category and a tool proposal from up to 131,072 Needle 3 tokens. Its smallest tested NCS1 build is a 512-byte device blob. The batch runner learns exact execution paths by vertical and document structure.

| Bundle | Task and evaluation scope | Physical NCS1 result | Warm p95 |
|---|---|---:|---:|
| [voice](bundles/voice/device-evaluation.json) | Eight isolated English commands on 901 recordings from held-out speakers | 817/901 (90.7%) | 3.89 ms |
| [intent](bundles/intent/device-evaluation.json) | Route a novel text utterance to a learned SLURP intent. 91 labels appear in training. One held-out example has an unseen label and counts as a miss. | 2,376/2,967 (80.1%) | 3.08 ms |
| [forms](bundles/forms/device-evaluation.json) | Localize a button, text field, checkbox and radio button from pixels. Four point-in-box checks on each of 90 new randomized Windows Forms images. | 321/360 (89.2%) | 45.49 ms/image |
| [grounding](bundles/grounding/device-evaluation.json) | Pick a browser control from recorded Mind2Web steps, using task text, earlier actions and DOM candidates. Ten held-out sites. | 23/310 (7.4%) | 2.82 ms/shortlist |

Latency covers a synchronous batch-one call after five warm-ups, in one device session per model. Audio capture, host feature extraction, DOM processing, model load and tool execution are outside these timings. Cold network load took 1.64–1.89 s. The physical device and FP32 reference agreed on every selected class or candidate for voice, intent and grounding. The pixel model had exact grid-point agreement on 98.3% of queries. No energy or direct per-core utilization measurement was made.

The voice model always chooses one of eight commands and expects a pre-segmented clip. The intent model produces a route only. The forms model proposes four coordinates and assumes each role is visible. Its test images come from the same Windows renderer and layout generator as training. General desktop screenshots, absent controls, repeated roles, free dictation, typed voice arguments and multi-step task completion remain open.

Browser target selection is experimental. The deterministic first candidate matched on 6/310 steps, the learned ranker matched on 23, and the correct target reached its shortlist on only 68. These scores include steps with missing candidates. They measure one-step proposals with recorded prior actions. The 32-candidate retrieval limit is the first bottleneck to remove. The next comparison should score all eligible controls in blocks on the stick, include collapsed/hidden targets in failure accounting, and evaluate a fresh site split.

## Run the trained models

From the MyriadForge repository root, use the Python 3.7 interpreter and `tools/ncs1` directory from a working 2020.3.2 installation. Both can come from another checkout. These commands emit decisions or coordinates. They do not click, type, submit forms or execute tools.

```powershell
$lane = (Resolve-Path tools/ncs1).Path
$py = Join-Path $lane '.venv/Scripts/python.exe'
$lab = 'labs/shave_models'
& $py "$lab/device.py" "$lab/bundles/intent" --toolchain $lane --text "what will the weather be tomorrow"
& $py "$lab/device.py" "$lab/bundles/voice" --toolchain $lane --wav command.wav
& $py "$lab/device.py" "$lab/bundles/forms" --toolchain $lane --image form.png
& $py "$lab/device.py" "$lab/bundles/grounding" --toolchain $lane --screen controls.json
```

Voice input is mono 16-bit PCM at 16 kHz, up to one second. Forms accepts an image and maps normalized prediction coordinates back to its pixel dimensions. The browser model accepts UTF-8 JSON with a task, optional history and controls in accessibility/DOM order, each with a unique ID. Example input:

```json
{"task":"Open the reading list","controls":[{"id":"a","name":"Download","role":"button","clickable":true},{"id":"b","name":"Reading list","role":"link","clickable":true}]}
```

The existing benchmark CLI also resolves `myriadforge/voice`, `myriadforge/intent`, `myriadforge/forms` and `myriadforge/grounding` to their evaluated FP16 IR file pairs. The CLI's generated inputs measure inference speed. Use the labeled replay below to reproduce task quality.

## Reproduce training and labeled hardware replay

Use modern Python for training with PyTorch, NumPy, ONNX, SoundFile, Pillow and lxml. The recorded run used Python 3.13, Torch 2.7.1 and ONNX 1.23.2. Use a separate working directory for training data and fixtures. `fetch_data.py` verifies SHA-256 for every input. Official train/validation/test splits are retained for SLURP after text deduplication. Speech Commands uses a seeded-by-speaker hash split. Mind2Web uses three upstream training shards, divided locally by website. This is a developmental site-held-out evaluation.

```powershell
$data = 'D:/models/ncs-shave-research'
foreach ($family in 'voice','intent','grounding') {
    python "$lab/fetch_data.py" $family --cache $data
}
python "$lab/train.py" voice --cache $data
python "$lab/train.py" intent --cache $data
python "$lab/train_grounding.py" --cache $data
powershell -NoProfile -File "$lab/render_forms.ps1" -Output "$data/forms"
python "$lab/train_forms.py" --cache $data
foreach ($family in 'voice','intent','forms','grounding') {
    & $py "$lab/device.py" "$lab/bundles/$family" --toolchain $lane --test "$data/$family-test.npz"
}
```

Use a fresh output directory via `--output` to preserve the provided weights during retraining. The worker compiles a new ONNX hash when necessary. Model manifests contain the train/validation/test counts and selection epoch. Device reports include both IR hashes, the evaluation fixture hash, accuracy and time measurements.

**Data attribution.** The voice CNN uses Pete Warden's Speech Commands corpus via TensorFlow's eight-word excerpt, licensed CC BY 4.0. The intent model uses only the CC BY 4.0 text annotations from Bastianelli et al.'s SLURP. The browser ranker uses CC BY 4.0 recorded Mind2Web data by Deng et al. The desktop training images are original offscreen Windows Forms renders with randomized positions, text and appearance. [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) · [SLURP data and licensing](https://github.com/pswietojanski/slurp) · [Mind2Web dataset](https://huggingface.co/datasets/osunlp/Mind2Web)

## Native Cactus UI model

[Needle UI](bundles/needle-ui) contains a custom rank-4 LoRA adapter and a merged 20-layer `.cact` model. The source model is Cactus Compute's Needle 3. This derivative proposes two local interface operations, finding a named control and scrolling a view. It was trained locally on 32 original examples including abstentions with three held out internally. The ten separate tests hold out control names and phrasings. The native base matched 3/10, and the fine-tune matched 4/10. The fine-tune matched both scrolls and both abstentions but missed all six unseen control lookups. Cactus's native runtime drops the untrained confidence head on export. These results support additional training and evaluation before using it as an automation policy. [Base weights and license](https://huggingface.co/Cactus-Compute/needle3/tree/d6c7d6a2e9655505e0f8e23339356dd1705900eb) · [Per-prompt base and tuned predictions](bundles/needle-ui/native-evaluation.json)

To reproduce, set the following paths to absolute paths in PowerShell, activate a separate Python 3.12 environment with `cactus-compute[train]==4.2.6`, and run from the data directory to keep the fetched base checkpoint outside the repository.

```powershell
$lab = (Resolve-Path labs/shave_models).Path
$needle = Join-Path $data 'needle'
python "$lab/needle_examples.py" $needle
Push-Location $needle
try {
    cactus finetune ui-train.jsonl --epochs 1 --batch-size 2 --max-len 256 --lora-rank 4 --lora-alpha 8 --seed 91 --checkpoint-dir . --out "$lab/bundles/needle-ui/ui-lora.safetensors"
    cactus build --lora "$lab/bundles/needle-ui/ui-lora.safetensors" --out "$lab/bundles/needle-ui/needle-ui.cact"
} finally { Pop-Location }
python "$lab/evaluate_needle.py"
```

The Needle archive was verified with Cactus on the Windows host. An NCS1 implementation of its learned text-to-schema behavior remains a separate engineering milestone. The general Cactus capability range, including open ASR, OCR, dense retrieval, grounded generation, synthesis and multimodal agents, remains under development.
