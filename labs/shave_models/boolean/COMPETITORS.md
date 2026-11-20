# MagicEightBall NCS1 size and competitor comparison

The comparison target is a physically loaded executable for the original Intel Neural Compute Stick. It must evaluate the seven support-policy facts, return all raw rule bits, and preserve the yes/no answer, category and action. The host uses the same exact token count and evidence extractor for the 131,072-token test. Report the executable, resident policy data, application files and cold/warm timing separately.

## Matched decisions

The shipped physical NCS1 Gather and four CPU backends agreed on all 128 fact assignments and the full-context fixtures. The CPU baselines are the same Gather graph, the Boolean interpreter, a 128-byte lookup table and a 46-byte decision diagram. The CPU figure counts decision data, while the 512-byte NCS1 figure counts the native executable. CPU had lower latency for 128 short decisions. Within the NCS1 lane, one batched submission took 6.43 ms median for those decisions, down from 223.70 ms as separate submissions. [Matched measurements](measurements/ncs1-cpu-comparison.json)

The native alternatives include convolution, dense and gates, a phrase scanner, radix Gather and weight-free clause evaluation. MaxPool and ReduceMax kept their truth-table results but increased the compiled size or transferred more data. A constant-free I32 Gather with one input also returned all 128 exact outputs. Splitting that input into the policy words and addresses produced a 136-byte serialized stage that occupied 192 bytes after alignment. The single-input artifact stayed at 512 bytes and required another 32 bytes of policy data. Separate inputs raised the artifact to 576 bytes. [Stock compiler controls](measurements/native-release-and-stock-controls.json) · [Pooling controls](measurements/constant-free-pooling.json) · [Streamed-parameter controls](measurements/streamed-parameters-compatibility.json)

The arithmetic lowering of the 46-byte decision diagram compiled to a **9,600-byte** MA2450 executable with 44 stages. The legacy FP16 IR matched all 128 assignments on the OpenVINO 2020.3.2 CPU plugin and 115 on 2022.3.1 CPU. The `Where` lowering passed the ONNX Runtime test and was rejected by the legacy Model Optimizer. Both remain offline comparisons. [Decision diagram lowering controls](measurements/decision-diagram-native-controls.json)

The resident rank-one Gather occupies 192 bytes of headers, 64 bytes each for input and output descriptors, 128 bytes for its stage and 64 bytes for the constants section in the stock 2020.3.2 exporter. This establishes a floor for that exporter and layout. Other stage types, toolchains and device-accepted layouts remain part of the search. [Intel section serializer](https://github.com/openvinotoolkit/openvino/blob/2020.3.2/inference-engine/src/vpu/graph_transformer/src/backend/serialize.cpp#L150-L175) · [Intel Gather stage](https://github.com/openvinotoolkit/openvino/blob/2020.3.2/inference-engine/src/vpu/graph_transformer/src/stages/gather.cpp#L62-L82)

## Published context

| Reference | What its published evidence covers |
|---|---|
| Intel NCSDK and Neural Compute Application Zoo | Original-stick compiler and application examples. The applications provide VPU compatibility context across model families. [NCSDK](https://github.com/movidius/ncsdk/releases) · [Application zoo](https://github.com/movidius/ncappzoo) |
| Blouw et al. | Physical Movidius NCS and CPU, GPU, Jetson and Loihi comparisons on a trained keyword spotter. [Paper](https://arxiv.org/abs/1812.01739) · [Benchmark source](https://github.com/abr/power_benchmarks) |
| fastText.zip | Learned text classification down to 16 KiB on public datasets. [Table 2](https://arxiv.org/pdf/1612.03651#page=7) |
| Bonsai | Learned feature classifiers at 2 KB and experiments on resource-constrained microcontrollers. [Paper](https://proceedings.mlr.press/v70/kumar17a.html) |
| FastGRNN | A 1 KB trained wake-word model with microcontroller deployment. [Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/ab013ca67cf2d50796b0c11d1b8bc95d-Paper.pdf) |

These public results have different tasks, inputs and storage boundaries. The search across original-stick toolchain examples, published hardware benchmarks and compact decision-model papers found no public artifact with the same support-policy and full-context contract. The defensible size claim is the smallest physically verified native artifact in this comparison. A worldwide record remains unverified.
