import onnx, sys, collections
m = onnx.load(sys.argv[1])
auto = collections.Counter()
stride_auto = 0
examples = []
for n in m.graph.node:
    if n.op_type not in ("Conv", "ConvTranspose", "AveragePool", "MaxPool"):
        continue
    ap, strides, pads = None, None, None
    for a in n.attribute:
        if a.name == "auto_pad":
            ap = a.s.decode()
        if a.name == "strides":
            strides = list(a.ints)
        if a.name == "pads":
            pads = list(a.ints)
    auto[(n.op_type, ap or "NOTSET")] += 1
    if ap and ap != "NOTSET" and strides and max(strides) > 1:
        stride_auto += 1
        if len(examples) < 5:
            examples.append((n.name, n.op_type, ap, strides, pads))
print("auto_pad usage:", dict(auto))
print("nodes with auto_pad AND stride>1:", stride_auto)
for e in examples:
    print("  ", e)
