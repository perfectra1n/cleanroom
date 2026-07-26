# ONNX graph tools

Throwaway-looking scripts kept deliberately, because they are the evidence behind
`docs/pitfalls.md`'s claim that ONNX Runtime's WebGPU provider miscomputes RVM — and
because the next person to re-test that claim on a newer `ort` should not have to
rebuild them.

Each rewrite is an **exact identity**. That is the whole method: change nothing about
what the model computes, then any change in output is a statement about the execution
provider rather than about the model. Every rewrite here was confirmed a no-op by
re-running it on the CPU provider and checking the result was unchanged.

```sh
# they need onnx, which is not in the dev shell
NIXPY='(builtins.getFlake "nixpkgs").legacyPackages.x86_64-linux.python3.withPackages (ps: [ ps.onnx ps.numpy ])'
run() { nix shell --impure --expr "$NIXPY" -c python3 "$@"; }

run tools/onnx/onnx_probe.py ~/.local/share/cleanroom/rvm_mobilenetv3_fp32.onnx   # op census + Resize attrs
run tools/onnx/attrs.py      ~/.local/share/cleanroom/rvm_mobilenetv3_fp32.onnx   # attribute values by op
run tools/onnx/nodes.py      ~/.local/share/cleanroom/rvm_mobilenetv3_fp32.onnx   # every node name, comma separated
run tools/onnx/rewrite.py    <in.onnx> <out.onnx> hardsigmoid|split|both
run tools/onnx/patch_resize.py <in.onnx> <out.onnx> half_pixel|asymmetric
```

Then point the matting crate at the result and compare providers:

```sh
CLEANROOM_RVM_MODEL=out.onnx \
  cargo run --release -p cleanroom-matting --example matte_sweep -- frame.png webgpu 1.0 512 288 10
```

`matte_sweep` also takes `CR_LAYOUT`, `CR_OPT`, `CR_GRAPHCAP`, `CR_BUFCACHE`,
`CR_VALIDATION` and `CR_FORCECPU`; see its module docs. Note that `CR_FORCECPU`
(`forceCpuNodeNames`) is inert on this ORT build — forcing all 353 nodes to the CPU still
returns the GPU's wrong answer at GPU speed, which is the control proving the option is
ignored rather than that the nodes are innocent.

## The fix: `pad_conv_channels.py`

This is the one that matters. ONNX Runtime's WebGPU provider computes `Conv` wrongly when
the input channel count is divisible by 3 and not by 4; RVM hits it in exactly one node.

```sh
run tools/onnx/pad_conv_channels.py \
  ~/.local/share/cleanroom/rvm_mobilenetv3_fp32.onnx \
  ~/.local/share/cleanroom/rvm_mobilenetv3_fp32.padded.onnx
```

`find_model()` prefers `*.padded.onnx` over the stock name in every search directory, so
writing it there is all that is needed — nothing else has to be configured.

Measured in the live daemon at 1080p30, RTX 5090:

| matting backend | daemon CPU | fps | matting | matte resolution |
|-----------------|-----------|-----|---------|------------------|
| gpu (padded model) | **46%** | 30.1 | 9.91 ms | 512x288 |
| cpu | 136% | 30.0 | 11.21 ms | 320x180 |
