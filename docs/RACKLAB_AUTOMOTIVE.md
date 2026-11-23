# Rack Lab automotive CPU preview

The authored service-intake pack looks up a recall or a technical service bulletin and returns a source-cited proposal for staff review. The bundle contains three catalog records and runs locally on CPU. It demonstrates review intake. Staff still confirm VIN-level recall status and bulletin build information at the linked source.

## Install and try it

From a clean checkout, with Python 3.11 or newer:

```sh
python -m pip install '.[racklab,racklab-tuning]'
myriadforge racklab demo
```

The demo opens and approves a recall reference. It reopens the journal, retrieves the exact prior reply, and applies a hold. A separate stream proposes review of a technical service bulletin.

[Recorded CPU demo](./workflows/automotive-preview.json) · [Authored scorecard](./workflows/automotive-preview-score.json) · [Physical NCS1 Boolean-head smoke results](./workflows/automotive-ncs1-smoke.json)

The ten-case scorecard includes an unsupported Honda recall paraphrase. Use the authored JSONL to reproduce that miss. [Download labeled examples](https://github.com/chadstthomas-svg/myriadforge/blob/main/labs/racklab/automotive-preview.jsonl)

```sh
myriadforge racklab run --journal automotive.sqlite < events.jsonl
```

Pass `--pack /path/to/automotive.json` to use a versioned custom catalog. Keep the three pinned artifacts beside the pack manifest. Start a new journal when changing the manifest, source catalog, or policy. Input is one JSON object per line from a trusted local caller.

```jsonl
{"stream_id":"case-a","seq":1,"event":"append","text":"2016 Ford F-150 door latch freezes. Find recall 17V652."}
{"stream_id":"case-a","seq":2,"event":"approve"}
{"stream_id":"case-a","seq":3,"event":"hold"}
```

Supply `vehicle` as a `year`, `make`, and `model` object on an append when these fields come from a confirmed intake form. Conflicting vehicle evidence stops the proposal. Approval binds to the current vehicle, route, and reference. A hold persists until a reset. A reset starts a new document within the same stream. The journal retains inputs and results locally. Exact retries return the stored response for that sequence, including after a restart.

## Collect and score a field holdout

Use reviewed operator notes and record policy version, source provenance, and review agreement in the restricted intake workspace. Use the same scenario group for related site, campaign, vehicle, or equipment evidence. Preserve near duplicates in one split. Reserve at least 30 fit notes, 20 selection notes, and an independent holdout with 60 eligible cases, 60 blocked or abstention cases, both document types, and at least two held-out groups. Include look-alike document IDs, unconfirmed vehicles, quoted approvals, conflicting approvals, and late holds.

Expected reference IDs are evaluation labels. The runner independently retrieves references from the pinned catalog. Use the shape below for each holdout row. The `reviewed_operator` provenance label requires human review. Authored rows use `authored`.

```jsonl
{"id":"heldout-01","scenario_group":"heldout-family-a","provenance":"authored","events":[{"event":"append","text":"2016 Ford F-150 recall 17V652."},{"event":"approve"}],"expected":{"route":"recall_lookup","subject":{"year":2016,"make":"Ford","model":"F-150"},"references":["17V652"],"proposal":"review_service_reference"}}
```

```sh
myriadforge racklab evaluate --cases frozen.jsonl --fit fit.jsonl --selection selection.jsonl --out report.json
```

Fit and selection rows require `id`, `scenario_group`, and `note` or `events`. The command returns 1 for a report with a failed data gate and 2 for an invalid run. The report counts exact eligible proposals, blocked proposals, and all-target top-three reference coverage. It checks split overlaps by case ID, scenario group, and normalized text. A field release also requires the fixed policy and recovery controls and a clean-download run.

For a shadow routing comparison, fit a `racklab-tune` adapter whose `automotive` market declares `recall_lookup`, `bulletin_lookup`, and `other`, then add `--shadow-adapter adapter.json` to evaluation. Shadow results add route-accuracy counts. The workflow keeps its pinned rules and approval contract. Keep reviewed corrections in the restricted workspace, fit after 30 new reviewed cases, and use a fresh holdout for any change to a released adapter.

## Evidence and compatibility

The earlier 66-case industry pilot reproduced 40 exact handoffs, including all 33 blocked cases. The new automotive demo is a separate, narrower authored pack. Its evaluation needs an independent operator holdout before field promotion. [Original raw pilot](https://chadstthomas-svg.github.io/myriadforge/workflows/industry-results.json)

The standalone automotive policy head passed 32 of 32 Boolean assignments in a physical NCS1 smoke test. Retrieval, routing, and journaling were outside that test. To repeat it, compile the packaged policy with the existing compiler, then invoke the verifier in the pinned legacy environment with SentencePiece 0.1.99.

```powershell
python labs/shave_models/boolean/build.py src/myriadforge/racklab/packs/policy.json ./work/auto-head --variant gather --slots 32
& $legacyPython tools/qualify_racklab_ncs1.py ./work/auto-head --toolchain $toolchain --out ./work/auto-head/physical.json
```

The original Needle 3 SentencePiece model is restored losslessly and verified by digest. Its source uses Apache-2.0. The package includes the license with the tokenizer. [Tokenizer provenance](https://huggingface.co/tablinum/MagicEightBall/blob/2b48c99709675f23b2e15354a35219a9030fcbc1/NOTICE.txt)
