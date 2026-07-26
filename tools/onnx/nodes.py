import onnx, sys
m = onnx.load(sys.argv[1])
print(",".join(n.name for n in m.graph.node if n.name))
