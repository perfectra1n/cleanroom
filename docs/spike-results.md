# M0a spike results

Three Day-1 go/no-go probes, each with a pre-agreed fallback so a failure would cost
hours rather than weeks. All three passed. This document records what was measured, on
what hardware, and — more usefully — the things that were **wrong in the plan** and had
to be corrected against reality.

Measured 2026-07-25 on: NixOS 26.11, kernel 7.1.4-cachyos, Hyprland 0.56.0 / Wayland,
Ryzen 9 9900X, RTX 5090 (driver 610.43.03) + AMD Granite Ridge iGPU (RADV / Mesa 26.1.5).

---

## Spike 3 — `ort` + WebGPU EP running RVM: **PASS**

The highest-risk item, so it ran first.

| Question | Answer |
|---|---|
| Is the WebGPU EP in a linux-x64 prebuilt at all? | **Yes** — `ort`'s `download-binaries` ships it, with `libwebgpu_dawn.so` alongside |
| Does RVM load through it? | **Yes** |
| Fast enough? | **4.53–4.73 ms** mean, 7.87 ms worst @ 512×288 on the RTX 5090 |

### Vendor neutrality is proven, not assumed

The *same binary*, selected only by `VK_DRIVER_FILES`:

| Adapter | Mean | Verdict |
|---|---|---|
| RTX 5090 (Blackwell, `nvidia_icd.json`) | **4.53 ms** | PASS — comfortably inside the <10 ms target |
| AMD Granite Ridge iGPU (RADV, `radeon_icd.x86_64.json`) | **38.77 ms** | Runs correctly; slow as expected for a 2-CU RDNA2-class iGPU |

No CUDA toolkit, no ROCm, no cuDNN, no TensorRT is installed on this machine. The AMD
result is a *correctness* pass — that part is the designated slow-GPU conformance target,
not a performance target.

### Corrections to the plan

**The plan was wrong that the nix store's ONNX Runtime would serve this.** It claimed
"ORT already in your store, so `load-dynamic` slots straight in". It does not: nixpkgs
builds onnxruntime **without** the WebGPU EP. Verified two ways —
`nix eval --json nixpkgs#onnxruntime.override.__functionArgs` exposes only
`coremlSupport`/`cudaSupport`/`ncclSupport`/`openvinoSupport`/`rocmSupport`, and the
built 1.27.1 store path contains only `libonnxruntime_providers_openvino.so` and
`libonnxruntime_providers_shared.so`. There is no Dawn anywhere in the store. The WebGPU
EP comes from `ort`'s own prebuilt, and that is now the supported path.

**The TensorRT symbolic-dimension hazard is real but WebGPU tolerates it.** The plan
budgeted for rewriting RVM's reused symbolic dim names. The session dump confirms the
hazard exists — `r1i` … `r4i` all carry
`dimension_symbols: ["batch_size", "channels", "height", "width"]`, sharing `height` and
`width` with `src` — which is exactly the pattern TensorRT read as equality constraints
and refused to build. ORT's WebGPU shape inference does not. **The workaround is not
needed.** Recurrent state auto-shaped correctly on frame 0:
`[[1,16,36,64], [1,20,18,32], [1,40,9,16], [1,64,5,8]]`.

### Two real defects found

**1. Segfault on teardown — and it would have caused a restart loop.**
Dropping an ORT session that owns a WebGPU/Dawn context segfaults reproducibly, on *both*
adapters, *after* all work completes successfully. This is not cosmetic: a systemd unit
with `Restart=on-failure` would read the segfault as a failed exit and restart the daemon
forever. Leaking the session (`std::mem::forget`) avoids it and exits `rc=0`. The daemon
needs either that, or a shutdown path that tears the GPU context down in a controlled
order — decide before wiring up `Restart=on-failure`.

**2. The degenerate-alpha guard was stated wrong.** The first version warned on any flat
matte and so cried wolf on every run. The failure it exists to catch is a flat **high**
alpha — RVM fed a blank frame emits a near-uniform ~0.96, which composites as a one-frame
"effect off" flash. A flat **low** alpha is the network correctly reporting *no subject*,
which is the right answer for this spike's synthetic gradient (RVM is trained on people).
The guard now checks `spread < 0.005 && min > 0.5`.

> Still outstanding: matte *quality* is unvalidated. A synthetic gradient proves shapes
> and speed, not correctness. Numeric comparison against the CPU EP on real footage with
> a person in it is a separate job.

---

## Spike 1 — Slint on Hyprland: **PASS**

egui was ruled out on two open, unfixed bugs; this probe existed to check Slint does not
share them, since both use winit.

| Failure being probed | Result |
|---|---|
| egui#8249 — keyboard input completely dead on Hyprland | **Not present.** Typed characters reach the app (confirmed interactively) |
| egui#8314 — drag-resize pins a core and stops the UI thread | **Not present.** 366 samples at a steady 60–62 fps; the only dips were startup (4.7) and a single 49.0 during resize, with immediate recovery |

Native Wayland (`xwayland: False`), no XWayland fallback.

### Finding: Slint sets no `app_id`

`hyprctl clients` reports `class: ''` for the spike window. This confirms, on this exact
stack, the trap flagged during planning: without an explicit app_id **matching the
`.desktop` basename byte-for-byte**, the app gets a generic icon in every taskbar, dock
and alt-tab, and portal features that require a valid app ID will refuse. It is a Slint
default, not an egui-specific problem, and the GUI must set it deliberately.

---

## Spike 2 — Slint wgpu texture interop: **PASS**

Marked "unverified, web budget exhausted" in the plan, with a memfd-upload fallback
assumed. The fallback is not needed.

`slint::Image::try_from(wgpu::Texture)` imports a texture we allocated and filled
ourselves. Confirmed at runtime, not just on paper:

```
[spike2] got Slint's wgpu device/queue, allocated our own texture
[spike2] Image::try_from(wgpu::Texture) succeeded — interop works
```

Two hard requirements, both enforced in `i-slint-core/graphics/wgpu_29.rs` and both
failing loudly rather than rendering wrong:

* format must be `Rgba8Unorm` or `Rgba8UnormSrgb`
* usage must include `TEXTURE_BINDING | RENDER_ATTACHMENT`

Uploading a fresh 640×360 RGBA texture every 33 ms and re-importing it costs nothing
measurable: the window holds a steady **60 fps** in release with the interop live. (A
debug build shows ~22 fps, but that is entirely the CPU-side test-pattern generator —
230k pixels of `sin`/`cos` unoptimised — not the interop. Worth stating because it would
be easy to misread as an interop cost.)

Device sharing works in **both** directions:

* `WGPUConfiguration::Manual { instance, adapter, device, queue }` — hand Slint a device
  *we* created, e.g. one pinned to a specific DRM render node.
* `Window::set_rendering_notifier` → `GraphicsAPI::WGPU29 { device, queue, .. }` — take
  Slint's. Simpler, and sufficient for a GUI whose job is only to display frames.

### Version constraint this pins

Slint 1.17.1 offers `unstable-wgpu-28` and `unstable-wgpu-29` and nothing newer. wgpu
30.0 shipped 2026-07-02. **The whole workspace is therefore pinned to wgpu 29** — the
types must unify across `cleanroom-gpu` and the GUI.

Also worth knowing: Slint 1.17's *default* features already include `system-tray` and
`accessibility`, so the separate `ksni` dependency may be unnecessary. Not yet verified.

---

## Environment notes worth keeping

Things that cost time here and will cost it again.

* **`ort` needs `tls-rustls`, not the default `tls-native`.** The build script downloads
  the prebuilt over HTTPS; linking that against system OpenSSL fails on NixOS with
  `libssl.so.3: cannot open shared object file`. rustls is pure Rust and works outside
  the dev shell too.
* **`copy-dylibs` puts `libonnxruntime.so` and `libwebgpu_dawn.so` next to the binary but
  sets no `$ORIGIN` rpath**, so they are not found at run time. The dev shell adds
  `target/debug` and `target/release` to `LD_LIBRARY_PATH`.
* **Do not glob `/nix/store/*-vulkan-loader-*/lib` by hand.** Several entries are 32-bit;
  picking one yields `Couldn't load Vulkan: libvulkan.so.1: wrong ELF class: ELFCLASS32`
  from inside Dawn, which reads as "no GPU present" rather than as a path bug. Nix's
  `makeLibraryPath` always resolves the right architecture.
* **Nix flakes cannot see untracked files.** `git add` before `nix develop`, or the
  flake is simply "not tracked by Git".
* **`${...}` inside a Nix `''` string is antiquotation even in what looks like a
  comment.** A `#` line inside `shellHook` is a *shell* comment, not a Nix one.
* **`cargo build ... | tail` masks the exit code** — the pipeline returns `tail`'s status.
  Use `set -o pipefail` or check `${PIPESTATUS[0]}`.

## Prior art re-confirmed

[`funinkina/openeffects`](https://github.com/funinkina/openeffects) independently chose
`ort` over `burn` for the same problem, which is a useful second vote for the reversal
this project made during its gap audit.
