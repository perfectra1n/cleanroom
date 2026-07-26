"""Extract a sub-model of RVM ending at the output of node `idx`.

The bisection tool. Everything upstream of the probe runs; everything downstream is
discarded. Comparing the same probe on two providers localises the first tensor where
they disagree, which is the operator that introduces the error.
"""
import onnx, sys
from onnx import shape_inference
from onnx.utils import extract_model

src, dst, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = onnx.load(src)
node = m.graph.node[idx]
out = node.output[0]

# Beside the *output*, never beside the input: the input usually lives in the
# user's model directory, and littering that with intermediates is rude.
inferred = dst + ".inferred"
onnx.save(shape_inference.infer_shapes(m), inferred)
ins = [i.name for i in m.graph.input]
try:
    extract_model(inferred, dst, ins, [out])
    print(f"OK idx={idx} op={node.op_type} name={node.name} out={out}")
except Exception as e:
    print(f"SKIP idx={idx} op={node.op_type} name={node.name}: {type(e).__name__}: {e}")
    sys.exit(3)
