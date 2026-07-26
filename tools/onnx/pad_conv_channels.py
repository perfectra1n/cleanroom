"""Round every Conv's input channel count up to a multiple of 4.

Works around a correctness bug in ONNX Runtime's WebGPU Conv kernel: a convolution
whose input channel count is divisible by 3 and *not* by 4 returns wrong values
(measured 17-20% off in L1), while every multiple of 4 is exact. C_in=3 is
special-cased upstream and is fine.

The rewrite is an exact identity. Each affected Conv gets:

  * a `Pad` on its input, appending zero channels to reach the next multiple of 4;
  * its weight tensor zero-padded along C_in to match.

Because the appended *weights* are zero, the appended input channels contribute
nothing to the result no matter what they hold — so the output is unchanged, and the
only thing that differs is which kernel path the provider takes.
"""
import onnx, numpy as np, sys
from onnx import helper, numpy_helper, TensorProto as T

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
init = {i.name: i for i in g.initializer}

def faulty(cin):
    return cin % 3 == 0 and cin % 4 != 0 and cin != 3

new_nodes, added, changed = [], [], 0
for n in g.node:
    grp = 1
    for a in n.attribute:
        if a.name == "group":
            grp = a.i
    w = init.get(n.input[1]) if n.op_type == "Conv" and len(n.input) > 1 else None
    if w is None or grp != 1 or not faulty(w.dims[1]):
        new_nodes.append(n)
        continue

    cin = w.dims[1]
    target = ((cin + 3) // 4) * 4
    extra = target - cin

    arr = numpy_helper.to_array(w)
    padded = np.zeros((arr.shape[0], target) + arr.shape[2:], dtype=arr.dtype)
    padded[:, :cin] = arr
    g.initializer.remove(w)
    added.append(numpy_helper.from_array(padded, n.input[1]))

    # NCHW: [b0, c0, h0, w0, b1, c1, h1, w1] — pad only the channel end.
    pads_name = f"{n.name}_chpad"
    added.append(numpy_helper.from_array(
        np.array([0, 0, 0, 0, 0, extra, 0, 0], dtype=np.int64), pads_name))
    padded_in = f"{n.name}_padded"
    new_nodes.append(helper.make_node(
        "Pad", [n.input[0], pads_name], [padded_in], name=f"{n.name}_Pad", mode="constant"))
    n.input[0] = padded_in
    new_nodes.append(n)
    changed += 1
    print(f"  {n.name}: C_in {cin} -> {target}")

del g.node[:]
g.node.extend(new_nodes)
g.initializer.extend(added)
onnx.checker.check_model(m, full_check=False)
onnx.save(m, dst)
print(f"padded {changed} conv(s) -> {dst}")
