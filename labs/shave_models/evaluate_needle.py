"""Measure base and tuned Needle with the same unseen prompts and no tool execution."""
import hashlib
import json
from pathlib import Path

import cactus

from needle_examples import examples, TOOLS


def main():
    bundle = Path(__file__).parent / "bundles/needle-ui"
    weights = bundle / "needle-ui.cact"
    report = {"backend": "native Cactus Windows host", "trained_sha256":
              hashlib.sha256(weights.read_bytes()).hexdigest(), "results": {}}
    for name, model in [("base", None), ("tuned", str(weights))]:
        rows = []
        with cactus.Cactus(tools=TOOLS, weights=model) as agent:
            for case in examples(True):
                agent.reset()
                response = agent.complete(case["query"])
                calls = [{"name": c["name"], "arguments": c["arguments"]}
                         for c in response.get("function_calls", [])]
                rows.append(dict(query=case["query"], expected=case["answers"], actual=calls,
                                 correct=calls == case["answers"]))
        report["results"][name] = dict(correct=sum(r["correct"] for r in rows), cases=rows)
    (bundle / "native-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v["correct"] for k, v in report["results"].items()})


if __name__ == "__main__":
    main()
