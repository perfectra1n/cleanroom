import onnx, sys
m = onnx.load(sys.argv[1])
init = {i.name: i for i in m.graph.initializer}
bad = []
for n in m.graph.node:
    if n.op_type != "Conv":
        continue
    g = 1
    for a in n.attribute:
        if a.name == "group":
            g = a.i
    w = init.get(n.input[1])
    if w is None:
        continue
    cin = w.dims[1] * g        # true input channels
    # The measured rule: divisible by 3, not divisible by 4, and not the special-cased 3.
    if cin % 3 == 0 and cin % 4 != 0 and cin != 3:
        bad.append((n.name, cin, list(w.dims), g))
print(f"convs hitting the faulty shape: {len(bad)}")
for name, cin, dims, g in bad:
    print(f"  {name:<12} C_in={cin:<5} W={dims} group={g} -> pad to {((cin+3)//4)*4}")
