# Rack Lab industry expansion plan

## Baseline

The local wave covers eleven workflows across automotive, facilities, and retail. On 66 frozen authored evaluation cases, keyword routing matched 14, the learned router matched 42, and the selected handoff pipeline matched 40 decisions. It made zero proposals in 33 blocked cases. Lexical retrieval returned every target in the top three for 62 cases. The fitted reranker reached 58. RailShot passed eleven restart checks and BreakShot matched 32 of 33 synthetic speech and ambient labels. These pilot numbers set the next work priorities.

## 1. Repair intake quality and acquire evidence

- **Automotive.** Start with identification and sales, recalls and bulletins, parts and warranty review, depot and rental handoffs, and EV charging support. Add supplier quality and lot traceability, pre-delivery inspection, used-vehicle reconditioning, and parts disposition in that order. Use NHTSA and DOE records for public reference targets. Use approved operator records to label company-specific routing and warranty decisions.
- **Facilities.** Start with HVAC work orders, warehouse equipment checks, and guest or contractor handoffs. Add preventive maintenance, energy alerts, multi-site dispatch, and work-order closeout. Preserve a source label for field-measured, lab, simulated, and authored examples.
- **Retail and distribution.** Start with receiving discrepancies, product recalls and returns, and POS or recorder support. Add inventory reconciliation, fulfillment exceptions, and vendor claims. Preserve device variants, look-alike product IDs, and missing serials as hard cases.
- For each initial workflow, collect at least 30 reviewed notes for development and 20 for a fresh evaluation. Include positive, ambiguous, and absent-evidence cases. Split by source family, site, or equipment before training. Each decision label needs the policy revision and each reference label needs the exact source record. Keep raw customer and site material in its approved workspace.

## 2. Select the strongest primary build per model

| Model | Work | Evidence to record |
|---|---|---|
| MagicEightBall | Train a small text-to-facts adapter where the exact route or identifier rules miss. Keep approvals, holds, and vetoes in explicit policy. Exercise the 131,072-token ingest path with relevant evidence at the beginning, middle, and end. | Exact route and field accuracy, coverage on ambiguous notes, zero proposals on blocked controls, and end-to-end latency and peak memory at each tested input length |
| BankShot | Expand the source catalogs. Start with field-aware lexical retrieval and promote a fitted reranker only after paired improvement on separate data. | All-target top-three recall, MRR, abstention errors, index size, and full-query latency |
| CueBall | Add pronunciation lexicons for product codes and equipment IDs. Compare the current full voice with a stronger quality candidate before distilling. | Human review of critical words and numbers, intelligibility, time to first audio, real-time factor |
| BreakShot | Collect annotated site audio from depots, mechanical rooms, and checkout areas. Tune thresholds before adapting a small readout. | Speech-event recall, false wakes per hour, onset delay, and the noisy retail miss |
| RailShot | Exercise larger event streams with duplicates, gaps, conflicting retries, restarts, and late holds. | Exact replay and recovery, zero duplicate commits, and throughput under load |

Freeze the chosen models, policy, labels, and thresholds before the new evaluation. Keep the keyword and lexical controls in every report and retain the earlier pilot as a regression set. Promote a workflow after blocked proposals and restart/retry controls pass and each measured component meets the thresholds recorded for that pack.

## 3. Publish measured examples and compatibility builds

1. Update the shared Break Room gallery after each measured wave. Show the input, expected reference or action, actual output, provenance, and misses.
2. Give each model a dedicated Space with a version-linked model card and a runnable or clearly labeled recorded example. BankShot already has its own Space. Add MagicEightBall's approval/reversal flow, RailShot's restart replay, CueBall playback, and BreakShot's speech/ambient timeline next.
3. Release primary CPU or modern accelerator artifacts on their own evidence. Compare smaller variants on the same frozen cases. Record quality, size, latency, and actual stage placement. Use a physical run for any end-to-end NCS1 claim.
4. Compare the primary build with a representative peer and the easiest baseline for each task. Publish input packs, hardware and runtime versions, warm and cold timings, peak memory, and quality on common inputs. For ONNX, GGUF, full-precision, and legacy builds, publish each format only after inference parity and a documented download-and-run check.
5. Log reviewed misses by workflow and keep a versioned correction queue. Refit from reviewed batches and test candidate improvements against frozen references, policy controls, and cross-industry regressions before promoting an adapter.

Use the [fine-tuning CLI](RACKLAB_FINE_TUNING.md) for the first routing and retrieval adaptation. Keep a named evidence bundle for each pack and update the published card from its raw report.
