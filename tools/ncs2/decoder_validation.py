"""Generate and validate cached Llama tokens on the physical NCS2."""

from argparse import ArgumentParser, SUPPRESS
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import median
import sys
from time import monotonic, perf_counter_ns

import numpy as np
from openvino.runtime import get_version

from cached_decoder import (
    ContextLimit, DeviceDecoder, ReferenceDecoder, decoder_fingerprint, load_weights, partitions, weight_bytes,
)
from decoder_checkpoint import digest
from myriadforge.myriad_x import allocation_properties
from worker_process import watch_decoder

DEFAULT = (16, 16, 1)
PROFILE_SCHEMA = "myriadforge.ncs2-decoder-profile/1"
LEADS = (
    "Once upon a time there was a little girl named Lily who lived near the forest.",
    "The little dog walked through the garden and found a green ball under the bench.",
    "In the morning, Sam and his sister went outside to see where the birds were singing.",
    "Mia opened the door and saw that the rain had washed the little red bridge away.",
)
BODY = (
    "The sky was bright, and a soft wind moved through the trees. They walked down the path by a stream "
    "and listened to the water running over the stones. When they reached the old wooden fence, they saw "
    "a small bird sitting in the grass. It looked up at them and hopped closer. They waited quietly until "
    "it flew into a tree. Then they followed the path home, stopping by some flowers to count the bees. "
)


class Tokenizer:
    def __init__(self, path):
        if path.suffix == ".json":
            from tokenizers import Tokenizer as FastTokenizer

            self.tokenizer = FastTokenizer.from_file(str(path))
            self.vocab = self.tokenizer.get_vocab_size(with_added_tokens=False)
            self.encode = lambda text: self.tokenizer.encode(text).ids
            self.decode = self.tokenizer.decode
        else:
            import sentencepiece as spm

            self.tokenizer = spm.SentencePieceProcessor(model_file=str(path))
            self.vocab = self.tokenizer.vocab_size()
            self.encode = lambda text: self.tokenizer.encode(text, out_type=int, add_bos=True)
            self.decode = self.tokenizer.decode


def fixture(tokenizer, context):
    lengths = (8, context // 2, context - 16) if context >= 128 else (4, 8, context - 16)
    result = []
    for length in dict.fromkeys(lengths):
        for title in LEADS:
            ids = tokenizer.encode(title + " " + BODY * (1 + length // 72))[:length]
            if len(ids) != length:
                raise ValueError("Fixture did not reach the intended token length")
            result.append(ids)
    return result


def edge_prompt(tokenizer, context):
    return tokenizer.encode((LEADS[0] + " " + BODY) * (1 + context // 64))[:context - 16]


def timed_step(decoder, token):
    start = perf_counter_ns()
    scores = decoder.step(token)
    guess = int(scores.argmax())
    return scores, guess, (perf_counter_ns() - start) / 1e6


def prefill(decoder, prompt):
    if not prompt:
        raise ValueError("Supply at least one prompt token")
    decoder.reset()
    start = perf_counter_ns()
    for token in prompt:
        scores, guess, _ = timed_step(decoder, token)
    return scores, guess, (perf_counter_ns() - start) / 1e6


def generate(decoder, scores, steps, *, eos=None):
    ids, samples = [], []
    guess = int(scores.argmax())
    for index in range(steps):
        ids.append(guess)
        if guess == eos or index + 1 == steps:
            break
        _, guess, duration = timed_step(decoder, guess)
        samples.append(duration)
    return ids, samples


def snapshot(cache):
    return cache.position, cache.keys.copy(), cache.values.copy()


def restore(cache, state):
    cache.position, cache.keys[:], cache.values[:] = state


def metrics(samples):
    return {"samples": len(samples), "p50": median(samples), "p95": float(np.percentile(samples, 95)),
            "tokens_per_second": 1000 * len(samples) / sum(samples)}


def qualify(weights, tokenizer, context, device, *, mode):
    cpu = ReferenceDecoder(weights, context)
    prompts = fixture(tokenizer, context)
    if mode == "smoke":
        prompts = prompts[:1]
    cases, ncs, host, first_ncs, first_cpu = [], [], [], [], []
    initial_logits = None
    for ids in prompts:
        expected, _, prefill_cpu = prefill(cpu, ids)
        observed, _, prefill_device = prefill(device, ids)
        first_ncs.append(prefill_device)
        first_cpu.append(prefill_cpu)
        if initial_logits is None:
            initial_logits = observed.copy()
        saved_scores = observed
        saved_cache = snapshot(device.cache)
        expected_ids, observed_ids, margins = [], [], []
        for index in range(16):
            want, got = int(expected.argmax()), int(observed.argmax())
            top = np.partition(expected, -2)[-2:]
            margins.append(float(top[1] - top[0]))
            expected_ids.append(want)
            observed_ids.append(got)
            if index + 1 < 16:
                expected, _, cpu_ms = timed_step(cpu, want)
                observed, _, ncs_ms = timed_step(device, want)
                host.append(cpu_ms)
                ncs.append(ncs_ms)
        restore(device.cache, saved_cache)
        first, _ = generate(device, saved_scores, 16)
        restore(device.cache, saved_cache)
        second, _ = generate(device, saved_scores, 16)
        cases.append({"prefix": ids, "cpu": expected_ids, "myriad": observed_ids,
                      "margins": margins, "free": first, "deterministic": first == second})
        print("CASE", len(cases), "/", len(prompts), "PARITY",
              sum(a == b for a, b in zip(expected_ids, observed_ids)), flush=True)

    over_limit = True
    if mode != "smoke":
        ids = edge_prompt(tokenizer, context)
        if len(ids) != context - 16:
            raise ValueError("Boundary prompt did not reach the intended size")
        scores, _, _ = prefill(device, ids)
        for _ in range(16):
            scores = device.step(int(scores.argmax()))
        try:
            device.step(int(scores.argmax()))
        except ContextLimit:
            over_limit = device.cache.position == context
        else:
            over_limit = False
    after, _, _ = prefill(device, prompts[0])
    reset_ok = np.array_equal(initial_logits, after)
    comparisons = [(a == b, gap) for row in cases
                   for a, b, gap in zip(row["cpu"], row["myriad"], row["margins"])]
    correct = sum(match for match, _ in comparisons)
    clear = sum(not match and margin >= .10 for match, margin in comparisons)
    ok = (correct / len(comparisons) >= .99 and clear == 0 and reset_ok and over_limit
          and all(row["deterministic"] for row in cases))
    return {"status": "ok" if ok else "parity_failed", "cases": cases, "matched": correct,
            "comparisons": len(comparisons), "clear_disagreements": clear, "reset_ok": reset_ok,
            "context_limit_ok": over_limit if mode != "smoke" else None,
            "decode_ms": {"myriad": metrics(ncs), "cpu": metrics(host)},
            "first_token_ms": {"myriad": metrics(first_ncs), "cpu": metrics(first_cpu)}}


def benchmark(tokenizer, context, device, duration, qualification=None, *, screening=False):
    prompts = fixture(tokenizer, context)
    sample_indices = (0,) if screening else (0, 4, 8)
    selected = [prompts[i] for i in sample_indices]
    controls, saved, prefill_samples = [], [], []
    parity = []
    for index, prompt in enumerate(selected):
        scores, _, ttft = prefill(device, prompt)
        prefill_samples.append(ttft)
        saved.append((scores, snapshot(device.cache)))
        controls.append(generate(device, scores, 16)[0])
        if qualification:
            reference = qualification["cases"][sample_indices[index]]
            if prompt != reference["prefix"]:
                raise ValueError("The tokenized benchmark fixtures changed")
            restore(device.cache, saved[-1][1])
            for step, (want, margin) in enumerate(zip(reference["cpu"], reference["margins"])):
                parity.append((int(scores.argmax()) == want, margin))
                if step + 1 < len(reference["cpu"]):
                    scores = device.step(want)
    if parity and (sum(match for match, _ in parity) + 1 < len(parity)
                   or any(not match and margin >= .10 for match, margin in parity)):
        raise ArithmeticError("Configuration disagrees with qualified CPU controls")
    samples, initial, rounds = [], [], 0
    stop = monotonic() + duration
    minimum_rounds = 4 if screening else 6
    while rounds < minimum_rounds or monotonic() < stop:
        index = rounds % len(selected)
        if duration:
            scores, _, ttft = prefill(device, selected[index])
            initial.append(ttft)
        else:
            scores, state = saved[index]
            restore(device.cache, state)
        sequence, times = generate(device, scores, 16)
        if sequence != controls[index]:
            raise ArithmeticError("Generated sequence changed during repeated replay")
        samples.extend(times)
        rounds += 1
        if rounds % 6 == 0:
            print(f"REPLAY {rounds} rounds", flush=True)
    return {"status": "ok", "rounds": rounds, "decode_ms": metrics(samples),
            "first_token_ms": metrics(initial or prefill_samples), "parity_matches": sum(match for match, _ in parity),
            "parity_checks": len(parity)}


def parse_allocation(text):
    result = tuple(map(int, text.split(":")))
    if len(result) != 3:
        raise ValueError("Use SHAVES:CMX:STREAMS")
    allocation_properties(*result)
    return result


def concurrent_benchmark(tokenizer, context, device):
    prompts = fixture(tokenizer, context)
    sessions = (device, device.new_session())
    trials = []
    for session, ids in zip(sessions, (prompts[0], prompts[8])):
        logits, _, _ = prefill(session, ids)
        state = snapshot(session.cache)
        expected, _ = generate(session, logits, 16)
        trials.append((session, logits, state, expected))

    def decode_session(trial):
        session, logits, state, expected = trial
        durations = []
        for _ in range(10):
            restore(session.cache, state)
            actual, samples = generate(session, logits, 16)
            if actual != expected:
                raise ArithmeticError("Concurrent generation changed the token sequence")
            durations.extend(samples)
        return durations

    start = perf_counter_ns()
    with ThreadPoolExecutor(max_workers=2) as pool:
        samples = [ms for row in pool.map(decode_session, trials) for ms in row]
    elapsed = (perf_counter_ns() - start) / 1e6
    return {"status": "ok", "sessions": 2, "per_token_ms": metrics(samples),
            "aggregate_tokens_per_second": len(samples) * 1000 / elapsed}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="llama2.c .bin or prepared Llama directory")
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--graphs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--allocation", type=parse_allocation, default=DEFAULT)
    parser.add_argument("--attention", choices=("reduce", "gemm", "grouped"), default="reduce")
    parser.add_argument("--layout", choices=("packed", "layer", "aligned", "float32"), default="packed")
    parser.add_argument("--norm", choices=("standard", "scaled"), default="standard")
    parser.add_argument("--shave-only", action="store_true", help="Disable the convolution accelerator")
    parser.add_argument("--mode", choices=("smoke", "qualify", "bench", "throughput", "generate"), default="qualify")
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--screening", action="store_true", help="Use the short CPU control for an initial sweep")
    parser.add_argument("--controls", type=Path, help="Successful qualification JSON for benchmark parity checks")
    parser.add_argument("--prompt", default=LEADS[0])
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=SUPPRESS)
    args = parser.parse_args()
    if not args.seconds >= 0:
        parser.error("Benchmark duration must be nonnegative")
    if not args.worker:
        output = args.out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        status, failure = watch_decoder([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]], output)
        if failure:
            output.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf8")
        return status
    weights = load_weights(args.checkpoint)
    tokenizer = Tokenizer(args.tokenizer)
    if not 32 <= args.context <= weights.limit or tokenizer.vocab != weights.vocab:
        parser.error("Supply a matching tokenizer and a window from 32 to the checkpoint limit")
    source = args.checkpoint if args.checkpoint.is_file() else args.checkpoint / "checkpoint.json"
    binding = {"checkpoint_sha256": digest(source), "tokenizer_sha256": digest(args.tokenizer),
               "context": args.context, "runtime": get_version()}
    controls = None
    if args.controls:
        controls = json.loads(args.controls.read_text(encoding="utf8"))
        if (controls.get("status") != "ok" or controls.get("mode") != "qualify"
                or any(controls.get(name) != value for name, value in binding.items())):
            parser.error("Supply passing controls for the same checkpoint, tokenizer, context, and runtime")
    if args.profile:
        from openvino.runtime import Core
        from cached_decoder import require_myriad_x

        profile = json.loads(args.profile.read_text(encoding="utf8"))
        expected = {**binding, "device": require_myriad_x(Core())}
        if profile.get("schema") != PROFILE_SCHEMA or profile.get("binding") != expected:
            parser.error("Profile does not match this checkpoint, tokenizer, context, device, or runtime")
        args.allocation = tuple(profile["allocation"])
        args.graphs, args.attention, args.layout = profile["graphs"], profile["attention"], profile["layout"]
        args.norm, args.shave_only = profile["norm"], profile["shave_only"]
        if (profile.get("decoder_sha256") != decoder_fingerprint()
                or profile.get("graph_ranges") != [list(pair) for pair in partitions(weights, args.graphs)]):
            parser.error("Profile does not match the decoder implementation or graph split")
        allocation_properties(*args.allocation)
    ids = tokenizer.encode(args.prompt) if args.mode == "generate" else None
    if ids is not None and (not ids or args.new_tokens < 1 or len(ids) + args.new_tokens > args.context):
        parser.error("Prompt and requested generation must fit the context window")
    cache_rows = args.context if args.layout == "aligned" else args.context - 1
    cache_in_bytes = 4 if args.layout == "float32" else 2
    report = {**binding, "model": args.checkpoint.name, "parameters": weights.parameters,
              "myriad_weight_bytes": weight_bytes(weights, 0, weights.layers, True),
              "placement": {"myriad": ["all transformer blocks", "attention", "vocabulary projection"],
                            "host": ["tokenization", "embedding lookup", "KV storage and transfer", "sampling"]},
              "allocation": args.allocation, "attention": args.attention, "layout": args.layout, "mode": args.mode,
              "norm": args.norm, "shave_only": args.shave_only,
              "kv_bytes_per_step": 2 * weights.layers * weights.kv_heads * weights.head_dim
              * (cache_in_bytes * cache_rows + 2)}
    try:
        device = DeviceDecoder(weights, args.context, args.graphs, args.allocation,
                               attention=args.attention, layout=args.layout,
                               norm=args.norm, hardware=not args.shave_only)
        report.update(device=device.device, graphs=device.graphs, load_ms=device.load_ms)
        if args.mode in ("qualify", "smoke"):
            report.update(qualify(weights, tokenizer, args.context, device, mode=args.mode))
        elif args.mode == "bench":
            report.update(benchmark(tokenizer, args.context, device, args.seconds, controls,
                                    screening=args.screening))
        elif args.mode == "throughput":
            report.update(concurrent_benchmark(tokenizer, args.context, device))
        else:
            scores, _, ttft = prefill(device, ids)
            tokens, times = generate(device, scores, args.new_tokens, eos=getattr(weights, "eos", 2))
            report.update(status="ok", text=tokenizer.decode(ids + tokens), token_ids=tokens,
                          first_token_ms=ttft, warm_ms=times)
    except (ArithmeticError, ValueError, RuntimeError) as error:
        report.update(status="error", detail=str(error).strip()[-1800:])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(report["status"], report.get("matched"), report.get("comparisons"), report.get("detail", ""), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
