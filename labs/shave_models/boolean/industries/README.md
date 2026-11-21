# MagicEightBall industry starter packs

One resident 512-byte native model supports two policies. Its policy bank is selected per request. The result is a proposal your application can review before doing any work.

| Pack | Use case | Example result |
|---|---|---|
| Logistics and fulfillment | Triage missing shipments and address exceptions. A proposal requires explicit dispatch approval and clears when a hold is present. | `parcel`, `trace_parcel` |
| Software on-call | Triage network and capacity incidents. Capacity-only proposals need both capacity signals, plus paging approval. Maintenance and cancellation veto paging. | `capacity`, `page_capacity` |

## Run a case on NCS1

From the directory containing `runtime.py`, set the toolchain to your working Windows OpenVINO 2020.3.2 NCS1 installation. The Needle tokenizer belongs beside the runner.

```powershell
$toolchain = 'D:/NTVStudy/tools/ncs1'
$python = Join-Path $toolchain '.venv/Scripts/python.exe'
$bundle = 'industries/bundle'
& $python runtime.py $bundle --toolchain $toolchain --blob "$bundle/ncs1-native.blob" --bank logistics --text 'Parcel missing. Dispatch approved.'
```

Expect `may_escalate=true`, `category=parcel`, and `action=trace_parcel`. For software on-call, use `--bank oncall --text 'Queue backed up. Workers saturated. Paging approved.'` and expect `may_page=true`, `category=capacity`, and `action=page_capacity`.

For JSON records, the [adaptive config](adaptive.json) provides `logistics-record` and `oncall-record`. Those modes count the whole document toward the token limit and extract decision evidence from the `evidence` field. Use `adaptive.py industries/adaptive.json --toolchain <path> --batch-timeout 30` and send a JSON line containing `vertical` and `documents`. Each document contains `id` and `text`, where `text` is the serialized source record. Keep prior approvals and old holds outside `evidence`.

## Simulation results

The [labeled suite](scenarios.json) contains 18 hand-authored cases. It includes routine authorization, incomplete evidence, contradictory history fields, explicit vetoes and two documents that place a veto at the 131,072-token boundary. The adapted policies matched **18 of 18** cases on both CPU and physical NCS1. Across the ten cases labeled for no action, the policies returned no proposals. An ungated baseline retained the class and corroboration logic but ignored approval and veto. It proposed actions in seven of those ten cases. These scores describe the labeled simulations. Live use needs labels from the intended workflow and validation with its actual records. [CPU report](../measurements/industry-simulation-cpu.json) · [NCS1 report](../measurements/industry-simulation-ncs1.json)

Rebuild the bank from the policy sources with `python build.py industries/policies.json <output> --variant banks --slots 128`. The bank has the same ONNX graph bytes as the previously measured two-bank build. Its manifest supplies the logistics and on-call phrases and result labels. To reproduce the simulation, run `simulate_industries.py industries/bundle --corpus <pinned-corpus> --output <report.json>` with OpenVINO 2022.3.1 for CPU. To run on NCS1, use the legacy runtime and add `--backend ncs1 --toolchain <path> --blob industries/bundle/ncs1-native.blob`.
