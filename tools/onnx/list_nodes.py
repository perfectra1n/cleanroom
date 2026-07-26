import onnx, sys
m = onnx.load(sys.argv[1])
for i, n in enumerate(m.graph.node):
    print(i, n.op_type, n.name, n.output[0] if n.output else "")
