# Rack Lab fine-tuning

Fit small routing and source-ranking heads for business intake. The CLI measures keyword routing, lexical search, fitted models, and an eight-bit router weight comparison. It exports a plain JSON adapter for CPU inference.

## Try the included pilot

From a fresh checkout with Python 3.11 or later:

~~~sh
python -m pip install -e ".[racklab-tuning]"
racklab-tune demo --pack labs/racklab/industry-packs.json --out runs/racklab-demo
racklab-tune predict --model runs/racklab-demo/adapter.json --market retail --text "Anker A1647 power bank returned; retrieve both recall notices"
~~~

The demo uses 44 fit notes, 11 selection notes, and 33 separately evaluated notes from the eleven-workflow pack. Each note appears once here. The larger workflow replay also pairs notes with approval, hold, or missing-field states. The demo writes the split files, adapter, and comparison report under the requested output folder. Use a fresh folder for each run.

The included CPU demo kept keyword routing after an 11/11 selection tie and selected the learned reranker. On its 33 evaluation notes, the candidate router matched 21 routes and the keyword rules matched seven. The reranker returned all targets in the top three for 30 notes, compared with 27 for the lexical baseline. These counts describe this adapter recipe and its 33-note evaluation.

## Fit your own pack

Use the included pack as a template. Give each workflow a stable route ID, market, disambiguating keywords, and approved reference IDs. Each reference has a title, source URL, and concise index text. The `fit` command reads the catalog and workflow definitions from the pack. Your labeled training cases come from separate files.

Supply JSON arrays or one JSON object per line in a `.jsonl` file. Each record needs these fields.

~~~json
{"id":"return-025","scenario_group":"site-c-recall-family-25","market":"retail","note":"Returned A1647 power bank lacks a serial number","route":"product_recall","targets":["anker_011","anker_466"]}
~~~

Use `route: null` for a labeled routing abstention and an empty `targets` list for a case without a relevant reference. Teach each declared route with reviewed notes. Include close alternatives, incomplete notes, and irrelevant requests in selection and evaluation. Null route labels contribute to those scorecards. Reuse the same group ID for all notes derived from the same case, equipment family, site record, or notice cluster. Split those groups before tuning. Review near duplicates and source leakage separately. Train only on material approved for the adapter's distribution. Its vocabulary stores terms from the fit notes.

~~~sh
racklab-tune fit --pack pack.json --train fit.jsonl --select selection.jsonl --out adapter.json
racklab-tune evaluate --model adapter.json --cases frozen.jsonl
~~~

The shim rejects overlapping case IDs, scenario groups, and normalized note text. It selects a learned component only when it strictly improves its selection metric. `evaluate` reports both baselines and learned profiles, including the number of cases with all required references in the top three. Use a fresh frozen set after changing the training data or selection rule.

`predict` returns a suggested route and ranked source links for policy review. Configure approvals, holds, missing-field requirements, and vetoes on the versioned MagicEightBall policy side. Use `--route int8` to inspect the quantized weight representation. It dequantizes for CPU scoring, so the result can support a weight-size comparison. Physical NCS1 compilation and device execution are separate checks.
