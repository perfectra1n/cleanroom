import onnx, numpy as np, sys
from onnx import helper, numpy_helper, TensorProto as T
vals = [int(v) for v in sys.argv[1:]]
for cin in vals:
    rng = np.random.default_rng(cin)
    W = (rng.standard_normal((16, cin, 3, 3)) * 0.2).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Conv", ["X", "W"], ["Y"], name="c",
                          group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])],
        f"c{cin}",
        [helper.make_tensor_value_info("X", T.FLOAT, [1, cin, 16, 16])],
        [helper.make_tensor_value_info("Y", T.FLOAT, [1, 16, 16, 16])],
        [numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 12)]); m.ir_version = 8
    onnx.save(m, f"chan_{cin}.onnx")
