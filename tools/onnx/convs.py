import onnx, sys, collections
m = onnx.load(sys.argv[1])
init = {i.name: i for i in m.graph.initializer}
groups = collections.Counter()
depthwise = 0
for n in m.graph.node:
    if n.op_type != "Conv":
        continue
    g = 1
    for a in n.attribute:
        if a.name == "group":
            g = a.i
    w = init.get(n.input[1])
    shape = tuple(w.dims) if w is not None else None
    groups[g] += 1
    if shape and g > 1 and shape[1] == 1:
        depthwise += 1
print("Conv count by `group`:", dict(sorted(groups.items())))
print("depthwise convs (group==channels, weight C_in/group==1):", depthwise)
