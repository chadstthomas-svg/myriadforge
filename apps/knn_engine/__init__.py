"""WS5 kNN-as-MatMul engine: the NCS2's non-vision headline workload.

Brute-force similarity search formulated as one GEMM - the constant vector
table is baked into the graph as MatMul weights, the query is the input,
scores come out of the GEMM, TopK runs on the host. ``capacity`` sizes the
table against the stick's ~1 GB usable budget; ``engine`` builds the ONNX
and benches batched queries on a real device -> queries/sec/stick.
"""
