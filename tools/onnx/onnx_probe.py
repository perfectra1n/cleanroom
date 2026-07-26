import onnx, collections, sys
m = onnx.load(sys.argv[1])
ops = collections.Counter(n.op_type for n in m.graph.node)
print('nodes:', len(m.graph.node), '| named:', sum(1 for n in m.graph.node if n.name))
print('top ops:', ops.most_common(12))
for n in m.graph.node:
    if n.op_type == 'Resize':
        attrs = {a.name: (a.s.decode() if a.type == 3 else (a.f if a.type == 1 else a.i)) for a in n.attribute}
        print('Resize', repr(n.name), attrs)
