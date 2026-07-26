import onnx, sys, collections
m = onnx.load(sys.argv[1])
seen = collections.defaultdict(set)
for n in m.graph.node:
    for a in n.attribute:
        val = a.s.decode() if a.type == 3 else (round(a.f, 6) if a.type == 1 else a.i if a.type == 2 else None)
        if val is not None:
            seen[(n.op_type, a.name)].add(val)
for (op, name), vals in sorted(seen.items()):
    if op in ("HardSigmoid", "Resize", "Clip", "LeakyRelu", "Elu", "Selu", "Softmax", "Gemm", "Split", "Pad"):
        print(f"{op:16} {name:32} {sorted(vals)}")
