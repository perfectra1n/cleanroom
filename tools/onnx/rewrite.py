"""Rewrite suspect operators into provably-equivalent ones.

Both rewrites are exact identities, so any change in output is a statement about
the execution provider, not about the model.

  hardsigmoid : HardSigmoid(x) == Clip(alpha*x + beta, 0, 1).  RVM carries the
                PyTorch alpha of 1/6, where the ONNX *default* is 0.2 — a kernel
                that ignores the attribute is wrong in all 28 squeeze-excite blocks.
  split       : axis=-3 on a 4D tensor is axis=1.  Negative axes are a routine
                place for a kernel to get normalisation wrong.
"""
import onnx, sys
from onnx import helper, numpy_helper, TensorProto
import numpy as np

src, dst, what = sys.argv[1], sys.argv[2], sys.argv[3]
m = onnx.load(src)
g = m.graph
print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])

changed = 0
if what in ("split", "both"):
    for n in g.node:
        if n.op_type == "Split":
            for a in n.attribute:
                if a.name == "axis" and a.i < 0:
                    a.i += 4          # 4D tensors throughout
                    changed += 1

if what in ("hardsigmoid", "both"):
    new_nodes, consts = [], []
    for n in g.node:
        if n.op_type != "HardSigmoid":
            new_nodes.append(n)
            continue
        alpha, beta = 0.2, 0.5
        for a in n.attribute:
            if a.name == "alpha":
                alpha = a.f
            if a.name == "beta":
                beta = a.f
        x, y, p = n.input[0], n.output[0], n.name
        an, bn, zn, on = f"{p}_a", f"{p}_b", f"{p}_zero", f"{p}_one"
        for nm, v in ((an, alpha), (bn, beta), (zn, 0.0), (on, 1.0)):
            consts.append(numpy_helper.from_array(np.array(v, dtype=np.float32), nm))
        new_nodes += [
            helper.make_node("Mul", [x, an], [f"{p}_mul"], name=f"{p}_Mul"),
            helper.make_node("Add", [f"{p}_mul", bn], [f"{p}_add"], name=f"{p}_Add"),
            helper.make_node("Clip", [f"{p}_add", zn, on], [y], name=f"{p}_Clip"),
        ]
        changed += 1
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(consts)

onnx.checker.check_model(m, full_check=False)
onnx.save(m, dst)
print(f"rewrote {changed} nodes ({what}) -> {dst}")
