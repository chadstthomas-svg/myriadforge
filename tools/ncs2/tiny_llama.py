"""Run a trained llama2.c checkpoint as a fixed-window Myriad X graph."""

from argparse import ArgumentParser
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

import numpy as np
from openvino.runtime import Core, Model, get_version, opset8 as op

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from decoder_checkpoint import Checkpoint
from myriadforge.myriad_x import allocation_properties, model_fingerprint, require_myriad_x


def constant(value):
    return op.constant(np.asarray(value, dtype=np.float32))


def rms_norm(x, weight):
    variance = op.reduce_mean(op.multiply(x, x), [-1], True)
    return op.multiply(op.divide(x, op.sqrt(op.add(variance, constant(1e-5)))), constant(weight))


def dense(x, weight):
    return op.matmul(x, constant(weight), False, True)


def make_model(weights, context, *, project=True):
    d, heads, width = weights.dim, weights.heads, weights.head_dim
    embedded = op.parameter([context, d], np.float32, name="embedded")
    select = op.parameter([1, context], np.float32, name="last_row")
    embedded.output(0).get_tensor().set_names({"embedded"})
    select.output(0).get_tensor().set_names({"last_row"})

    freq = 10_000.0 ** (-np.arange(0, width, 2, dtype=np.float32) / width)
    angle = np.repeat(np.arange(context)[:, None] * freq, 2, axis=-1)
    cos, sin = constant(np.cos(angle)), constant(np.sin(angle))
    rotation = np.zeros((width, width), np.float32)
    pairs = np.arange(0, width, 2)
    rotation[pairs + 1, pairs] = -1
    rotation[pairs, pairs + 1] = 1
    rotation = constant(rotation)
    mask = constant(np.triu(np.full((context, context), -10_000.0), k=1))
    scale = constant(width ** -.5)
    shared = np.arange(heads, dtype=np.int32) // (heads // weights.kv_heads)
    grouped = weights.kv_heads != heads

    def split(x, count):
        return op.transpose(op.reshape(x, [context, count, width], False), [1, 0, 2])

    def rotate(x):
        return op.add(op.multiply(x, cos), op.multiply(op.matmul(x, rotation, False, False), sin))

    x = embedded
    for layer in range(weights.layers):
        normalized = rms_norm(x, weights.att_norm[layer])
        q = rotate(split(dense(normalized, weights.wq[layer]), heads))
        k = rotate(split(dense(normalized, weights.wk[layer]), weights.kv_heads))
        v = split(dense(normalized, weights.wv[layer]), weights.kv_heads)
        if grouped:
            k = op.gather(k, shared, 0)
            v = op.gather(v, shared, 0)
        scores = op.multiply(op.matmul(q, k, False, True), scale)
        probability = op.softmax(op.add(scores, mask), -1)
        attention = op.matmul(probability, v, False, False)
        joined = op.reshape(op.transpose(attention, [1, 0, 2]), [context, d], False)
        x = op.add(x, dense(joined, weights.wo[layer]))
        normalized = rms_norm(x, weights.ffn_norm[layer])
        up = dense(normalized, weights.w1[layer])
        gated = op.multiply(op.multiply(up, op.sigmoid(up)), dense(normalized, weights.w3[layer]))
        x = op.add(x, dense(gated, weights.w2[layer]))
    last = op.matmul(select, rms_norm(x, weights.final_norm), False, False)
    output = dense(last, weights.output) if project else last
    return Model([output], [embedded, select], "Static TinyLlama")


def generate(weights, context, request, prompt, limit, eos, reference=None):
    history = list(prompt)
    embedded = np.zeros((context, weights.dim), np.float32)
    select = np.zeros((1, context), np.float32)
    samples, generated, traces = [], [], []
    for step in range(limit):
        embedded[:len(history)] = weights.embed[history]
        select.fill(0)
        select[0, len(history) - 1] = 1
        feed = {"embedded": embedded, "last_row": select}
        start = perf_counter_ns()
        output = request.infer(feed)
        samples.append((perf_counter_ns() - start) / 1e6)
        logits = np.asarray(next(iter(output.values()))).reshape(-1)
        if logits.shape != (weights.vocab,) or not np.isfinite(logits).all():
            raise ArithmeticError("Non-finite logits or wrong vocabulary size")
        guess = int(logits.argmax())
        generated.append(guess)
        if reference is None:
            traces.append(logits.copy())
            history.append(guess)
            if guess == eos:
                break
        else:
            traces.append(float(np.max(np.abs(logits - reference[step]))))
            history.append(int(reference[step].argmax()))
    return generated, samples, traces


class DeviceProjection:
    """Run the CPU transformer and send its final hidden row to the VPU."""

    def __init__(self, cpu, device):
        self.cpu = cpu
        self.input = device.input(0)
        self.device = device.create_infer_request()
        self.samples = []

    def infer(self, feed):
        hidden = next(iter(self.cpu.infer(feed).values()))
        start = perf_counter_ns()
        result = self.device.infer({self.input: hidden})
        self.samples.append((perf_counter_ns() - start) / 1e6)
        return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--partition", choices=("full", "head"), default="head",
                        help="Compile the full graph or run the vocabulary projection on MYRIAD")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import sentencepiece as spm

    weights = Checkpoint(args.checkpoint)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if tokenizer.vocab_size() != weights.vocab or tokenizer.bos_id() < 0:
        parser.error("Tokenizer must match the checkpoint's vocabulary")
    prompts = args.prompt or ["Once upon a time", "The little dog"]
    inputs = [tokenizer.encode(text, out_type=int, add_bos=True) for text in prompts]
    if not (0 < args.context <= weights.limit and args.tokens > 0
            and max(map(len, inputs)) + args.tokens <= args.context):
        parser.error("Prompt plus generation must fit the checkpoint and static window")
    core = Core()
    model = make_model(weights, args.context)
    report = {
        "schema": "myriadforge.ncs2-tiny-llama/1", "model": args.checkpoint.name,
        "checkpoint_sha256": model_fingerprint(args.checkpoint)[".bin"], "parameters": weights.parameters,
        "tokenizer_sha256": model_fingerprint(args.tokenizer).get(args.tokenizer.suffix.lower()),
        "fp16_weight_bytes": weights.parameters * 2, "context": args.context, "runtime": get_version(),
        "partition": args.partition,
        "placement": {"host": ["tokenizer", "token embedding lookup", "greedy selection", "decode loop"],
                      "myriad": ["transformer", "attention", "vocabulary projection"]},
        "prompts": [],
    }
    cpu = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 1}).create_infer_request()
    controls = []
    for prompt, ids in zip(prompts, inputs):
        tokens, durations, logits = generate(weights, args.context, cpu, ids, args.tokens, tokenizer.eos_id())
        controls.append(logits)
        report["prompts"].append({"text": prompt, "input_token_ids": ids, "cpu_token_ids": tokens,
                                  "cpu_text": tokenizer.decode(ids + tokens), "cpu_ms": durations})
    if not args.cpu_only:
        try:
            report["device"] = require_myriad_x(core)
            if args.partition == "head":
                report["placement"]["host"] += ["transformer", "attention"]
                report["placement"]["myriad"] = ["vocabulary projection"]
                hidden = op.parameter([1, weights.dim], np.float32)
                model = Model([dense(hidden, weights.output)], [hidden], "TinyLlama vocabulary projection")
                report["myriad_fp16_weight_bytes"] = weights.output.size * 2
                host = core.compile_model(make_model(weights, args.context, project=False), "CPU",
                                          {"INFERENCE_NUM_THREADS": 1}).create_infer_request()
            else:
                report["myriad_fp16_weight_bytes"] = weights.parameters * 2
            start = perf_counter_ns()
            print(f"Compiling {args.partition} on {report['device']}", flush=True)
            device = core.compile_model(model, "MYRIAD", allocation_properties(16, 16, 1))
            report["compile_ms"] = (perf_counter_ns() - start) / 1e6
            print(f"Compiled in {report['compile_ms']:.1f} ms", flush=True)
            request = DeviceProjection(host, device) if args.partition == "head" else device.create_infer_request()
            for case, ids, logits in zip(report["prompts"], inputs, controls):
                predicted, _, errors = generate(weights, args.context, request, ids, len(logits),
                                                tokenizer.eos_id(), reference=logits)
                generated, times, _ = generate(weights, args.context, request, ids, args.tokens, tokenizer.eos_id())
                case.update(ncs2_token_ids=generated, ncs2_text=tokenizer.decode(ids + generated),
                            ncs2_ms=times, teacher_forced_matches=sum(a == b for a, b in zip(predicted, case["cpu_token_ids"])),
                            max_abs_logit_error=max(errors))
            warm = [ms for case in report["prompts"] for ms in case["ncs2_ms"][1:]]
            report["warm_median_ms_per_token"] = median(warm) if warm else None
            if args.partition == "head" and len(request.samples) > 1:
                report["warm_median_projection_ms"] = median(request.samples[1:])
            exact = all(case["ncs2_token_ids"] == case["cpu_token_ids"]
                        and case["teacher_forced_matches"] == len(case["cpu_token_ids"])
                        for case in report["prompts"])
            report["status"] = "ok" if exact else "parity_failed"
        except (ArithmeticError, RuntimeError) as error:
            report.update(status="error", detail=str(error).strip()[-1800:])
    else:
        report["status"] = "cpu_only"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(report["status"], report.get("warm_median_ms_per_token"), report.get("detail", ""), flush=True)
    return 0 if report["status"] in ("ok", "cpu_only") else 1


if __name__ == "__main__":
    sys.exit(main())
