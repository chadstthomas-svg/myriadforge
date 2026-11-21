# MagicEightBall frontline starter packs

Three policy banks propose routes to a human review queue. Each bank uses seven explicit phrase predicates. It first classifies evidence, then checks the workflow's approval and vetoes, then proposes the specific review. When current evidence lacks the approval phrase or includes a veto phrase, the category can remain visible while the action becomes `null`. All three banks share a 32-byte compiled table in the 192-byte ONNX graph. The policy manifest supplies each bank's phrases and result labels.

| Workflow | Evidence examples | Approval | Vetoes | Example proposal |
|---|---|---|---|---|
| [Dealership](dealership.json) | Service requests and inspection referrals | `intake approved` | `do not queue`, `customer declined` | `review_service_intake` |
| [Manufacturing](manufacturing.json) | Vibration alerts, high filter differential and low lubrication | `maintenance review approved` | `maintenance hold`, `work order blocked` | `review_mechanical_alert` |
| [Store security](store-security.json) | Door events and camera outages | `security review approved` | `drill active`, `alert cleared` | `review_access_alert` |

The dealership pack covers service and inspection intake. The manufacturing pack recommends maintenance review or a proposed work order for a person to assess. The store security pack routes objective equipment alerts to a person for review. Keep live evidence fields limited to current workflow facts emitted by the source system. Use the host application to validate current approval and resolve the resulting review requests.

## Run a case on NCS1

From the directory containing `runtime.py`, use your working Windows OpenVINO 2020.3.2 NCS1 toolchain. Keep the Needle tokenizer beside the runner.

```powershell
$toolchain = 'D:/NTVStudy/tools/ncs1'
$python = Join-Path $toolchain '.venv/Scripts/python.exe'
$bundle = 'industries/frontline/bundle'
& $python runtime.py $bundle --toolchain $toolchain --blob "$bundle/ncs1-native.blob" --bank dealership --text 'Service requested. Intake approved.'
```

Expect `may_route=true`, `category=service`, and `action=review_service_intake`. Try `--bank manufacturing --text 'Vibration alert. Maintenance review approved.'` for `review_mechanical_alert` or `--bank store_security --text 'Door forced. Security review approved.'` for `review_access_alert`.

## Route a structured record

The [adaptive config](adaptive.json) supplies text and record modes for all three banks. The record modes count the whole serialized record toward the token limit and read decision evidence from the top-level `evidence` string. Put archived approvals, cleared holds, and historical notes in their own fields.

```powershell
'{"vertical":"store-security-record","documents":[{"id":"door-042","text":"{\"evidence\":\"Door held open. Security review approved.\",\"history\":\"Drill active yesterday.\"}"}]}' |
  & $python adaptive.py industries/frontline/adaptive.json --toolchain $toolchain --batch-timeout 30
```

The live record proposes `review_access_alert`. Moving `Drill active` into `evidence` clears the action.

## Replay the labeled simulations

The [27 cases](scenarios.json) include routine and combined routes, missing approvals, active holds, wording that should not trigger, archived evidence, and a final veto at exactly 131,072 tokens for each bank. Both CPU and physical NCS1 matched all 27 labels and all 384 raw Boolean assignments across the three banks. It returned zero proposals in the 15 cases labeled for no action. The ungated control proposed 12 actions in those cases after the scorer removed only the approval and veto requirements. These measurements describe the included simulated records. [CPU report](../../measurements/industry-simulation-frontline-cpu.json) · [Physical NCS1 report](../../measurements/industry-simulation-frontline-ncs1.json)

Rebuild the ONNX and policy manifest with `python build.py industries/frontline/policies.json <output> --variant banks --slots 128`. Use OpenVINO 2022.3.1 for the CPU replay and supply the pinned corpus used for the release measurements.

```powershell
$corpus = 'decision-corpus.txt'
Invoke-WebRequest 'https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt' -OutFile $corpus
python simulate_industries.py industries/frontline/bundle --scenarios industries/frontline/scenarios.json --corpus $corpus --verify-assignments --output frontline-cpu.json
```

On NCS1, use the legacy runtime and add `--backend ncs1 --toolchain <path> --blob industries/frontline/bundle/ncs1-native.blob`.
