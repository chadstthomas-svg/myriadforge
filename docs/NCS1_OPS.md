# Myriad 2 op-support matrix (empirical)

Generated 2026-11-20 by `tools/ncs1/op_matrix.py` on real ma2450 silicon —
OpenVINO 2020.3.2, ONNX opset 11, one minimal graph per op, subprocess-per-op with a
75s hang timeout. `MO_FAIL` = Model Optimizer could not convert the graph itself;
`REJECT` = converted fine but the MYRIAD plugin refused to compile/infer it; `HANG` = never returned;
`DEVICE_WEDGE` = the host USB path wedged before a verdict could be taken (transient, re-probe later).

**Summary:** {"OK": 42, "CHECK_FAIL": 1, "MO_FAIL": 15, "REJECT": 7, "DEVICE_WEDGE": 3} of 68 ops.

| Op | Family | Status | Detail |
|---|---|---|---|
| `Conv` | conv/pool | OK |  |
| `Relu` | activation | OK |  |
| `Sigmoid` | activation | OK |  |
| `Tanh` | activation | OK |  |
| `LeakyRelu` | activation | OK |  |
| `Prelu` | activation | CHECK_FAIL | No Op registered for Prelu with domain_version of 11

==> Context: Bad node spec |
| `Elu` | activation | OK |  |
| `Selu` | activation | MO_FAIL | mo_onnx conversion failed for Selu.onnx: ptimizer with --log_level=DEBUG for mor |
| `HardSigmoid` | activation | REJECT | Failed to compile layer "output/HardSigmoid_": unsupported layer type "HardSigmo |
| `Softplus` | activation | MO_FAIL | mo_onnx conversion failed for Softplus.onnx: ptimizer with --log_level=DEBUG for |
| `Softsign` | activation | REJECT | Failed to compile layer "abs_output": unsupported layer type "Abs" |
| `Softmax` | activation | OK |  |
| `LogSoftmax` | activation | MO_FAIL | mo_onnx conversion failed for LogSoftmax.onnx: ptimizer with --log_level=DEBUG f |
| `Erf` | activation | OK |  |
| `Sign` | activation | MO_FAIL | mo_onnx conversion failed for Sign.onnx: ptimizer with --log_level=DEBUG for mor |
| `Add` | math | OK |  |
| `Mul` | math | OK |  |
| `Sub` | math | OK |  |
| `Div` | math | OK |  |
| `Pow` | math | OK |  |
| `Exp` | math | OK |  |
| `Log` | math | OK |  |
| `Sqrt` | math | OK |  |
| `Abs` | math | REJECT | Failed to compile layer "output": unsupported layer type "Abs" |
| `Neg` | math | OK |  |
| `Floor` | math | OK |  |
| `Round` | math | MO_FAIL | mo_onnx conversion failed for Round.onnx: ptimizer with --log_level=DEBUG for mo |
| `MatMul` | math | OK |  |
| `Gemm` | math | OK |  |
| `Transpose` | tensor | OK |  |
| `Reshape` | tensor | OK |  |
| `Flatten` | tensor | OK |  |
| `Squeeze` | tensor | OK |  |
| `Unsqueeze` | tensor | OK |  |
| `Concat` | tensor | OK |  |
| `Slice11` | tensor | OK |  |
| `Cast` | tensor | OK |  |
| `Gather` | tensor | OK |  |
| `Tile11` | tensor | OK |  |
| `Expand8` | tensor | REJECT | Check 'arg_shape[i - start_axis] == 1 // target_shape[i] == 1 // arg_shape[i - s |
| `Shape` | tensor | REJECT | Unexpected network batch size : 0 |
| `Pad11` | tensor | MO_FAIL | mo_onnx conversion failed for Pad11.onnx: [ ERROR ]  Exception occurred during r |
| `Transpose4d` | tensor | OK |  |
| `MaxPool` | conv/pool | OK |  |
| `AveragePool` | conv/pool | OK |  |
| `GlobalAveragePool` | conv/pool | OK |  |
| `GlobalMaxPool` | conv/pool | OK |  |
| `ConvTranspose` | conv/pool | OK |  |
| `DepthToSpace` | conv/pool | MO_FAIL | mo_onnx conversion failed for DepthToSpace.onnx: ptimizer with --log_level=DEBUG |
| `SpaceToDepth` | conv/pool | MO_FAIL | mo_onnx conversion failed for SpaceToDepth.onnx: ptimizer with --log_level=DEBUG |
| `LpNormalization` | conv/pool | OK |  |
| `LRN` | conv/pool | DEVICE_WEDGE | deterministic: re-probed 2026-11-20 after recovery, wedged again (outer kill at  |
| `BatchNormalization` | conv/pool | DEVICE_WEDGE | deterministic: re-probed 2026-11-20 after recovery, wedged again (outer kill at  |
| `Resize11` | conv/pool | MO_FAIL | mo_onnx conversion failed for Resize11.onnx: ptimizer with --log_level=DEBUG for |
| `ReduceMean` | reduction | OK |  |
| `ReduceSum` | reduction | OK |  |
| `ReduceMax` | reduction | OK |  |
| `ArgMax` | reduction | OK |  |
| `ArgMin` | reduction | MO_FAIL | mo_onnx conversion failed for ArgMin.onnx: ptimizer with --log_level=DEBUG for m |
| `TopK10` | data-dependent | OK |  |
| `CumSum` | data-dependent | MO_FAIL | mo_onnx conversion failed for CumSum.onnx: ptimizer with --log_level=DEBUG for m |
| `NonZero` | data-dependent | MO_FAIL | mo_onnx conversion failed for NonZero.onnx:  for node output with type=<UNKNOWN> |
| `Where` | data-dependent | MO_FAIL | mo_onnx conversion failed for Where.onnx: ptimizer with --log_level=DEBUG for mo |
| `ScatterND` | data-dependent | MO_FAIL | mo_onnx conversion failed for ScatterND.onnx: ptimizer with --log_level=DEBUG fo |
| `Clip_afterConv` | data-dependent | REJECT | Clamp layer output has incorrect number of inputs (MO-generated Clamp broken in  |
| `Expand_afterConv` | tensor | MO_FAIL | MO 2020.3 cannot convert Expand after a Conv |
| `Shape_afterConv` | tensor | DEVICE_WEDGE | deterministic: re-probed 2026-11-20 after recovery, wedged again (outer kill at  |
| `Clip11` | data-dependent | REJECT | Clamp layer output with id: 3 has incorrect number of inputs! |

Unknown ops in a graph should be assumed HETERO-fallback until probed.

_This table is generated - do not edit by hand._
