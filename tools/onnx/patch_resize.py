"""Rewrite RVM's Resize coordinate mode, and save a patched copy.

`pytorch_half_pixel` and `half_pixel` differ only when the resized length is 1
(ONNX Resize spec); for every size this model actually uses they compute the
identical mapping. So this is a semantic no-op that changes only which kernel
path an execution provider takes.
"""
import onnx, sys

src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
m = onnx.load(src)
n_changed = 0
for node in m.graph.node:
    if node.op_type != "Resize":
        continue
    for a in node.attribute:
        if a.name == "coordinate_transformation_mode":
            a.s = mode.encode()
            n_changed += 1
onnx.save(m, dst)
print(f"rewrote {n_changed} Resize nodes to {mode} -> {dst}")
