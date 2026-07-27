# Pitfalls

Every trap that cost real time building Cleanroom, with the code that resolves it. Each
entry states the **symptom first**, because in almost every case the symptom pointed
somewhere other than the cause.

Companion to [`spike-results.md`](spike-results.md), which has the measured numbers.

---

## Build and environment

### `libssl.so.3: cannot open shared object file` while building `ort`

`ort`'s build script downloads a prebuilt over HTTPS and, by default, links that download
against system OpenSSL — which does not exist on NixOS.

```toml
# Not the default tls-native. rustls is pure Rust, so the build script works outside the
# dev shell too.
ort = { version = "=2.0.0-rc.12", default-features = false, features = [
    "std", "ndarray", "tracing", "download-binaries",
    "tls-rustls",          # <- this line
    "copy-dylibs", "api-24", "webgpu",
] }
```

### `turbojpeg-sys` build fails in cmake

It defaults to *building* its own bundled libjpeg-turbo, which needs nasm for SIMD.

```sh
export TURBOJPEG_SOURCE=pkg-config   # set in flake.nix and .mise/config.toml
```

The system copy is also faster: nixpkgs builds it with SIMD enabled.

### `libonnxruntime.so` / `libwebgpu_dawn.so` not found at run time

`ort`'s `copy-dylibs` drops both next to the built binary but sets no `$ORIGIN` rpath.

```sh
LD_LIBRARY_PATH="$PWD/target/debug:$PWD/target/release:$LD_LIBRARY_PATH"
```

### `Couldn't load Vulkan: libvulkan.so.1: wrong ELF class: ELFCLASS32`

Reported from **inside Dawn**, so it reads as "no GPU present" rather than as a path bug.
The cause is globbing the nix store by hand and hitting a 32-bit build:

```sh
# WRONG — roughly half the matches are 32-bit
VKLOADER=$(ls -d /nix/store/*-vulkan-loader-*/lib | head -1)
```

Let nix resolve it. `pkgs.lib.makeLibraryPath` always picks the right architecture; the
flake's `runtimeLibs` list is the only correct source.

### `error: Path 'flake.nix' ... is not tracked by Git`

Nix flakes cannot see untracked files. `git add` before `nix develop`.

### Nix flake syntax error pointing at a comment

Inside a `''…''` string, `${…}` is antiquotation **even on a line starting with `#`** —
that `#` is a *shell* comment, not a Nix one. This breaks the flake:

```nix
shellHook = ''
  # ${pkgs.lib.makeLibraryPath ...} always resolves the right architecture
'';
```

### `cargo build … | tail` reports success on failure

The pipeline returns `tail`'s status. Use `set -o pipefail` or check `${PIPESTATUS[0]}`.

---

## GPU

### wgpu 29 API differences from older examples

```rust
// InstanceDescriptor has no Default, and `display` is required.
let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    backends: wgpu::Backends::VULKAN,
    flags: wgpu::InstanceFlags::default(),
    memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
    backend_options: wgpu::BackendOptions::default(),
    display: None,                       // headless: we never present to a surface
});

// enumerate_adapters is async in 29.
let adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN).await;

// DeviceDescriptor needs experimental_features.
adapter.request_device(&wgpu::DeviceDescriptor {
    /* … */
    trace: wgpu::Trace::Off,
    experimental_features: wgpu::ExperimentalFeatures::disabled(),
}).await?;

// PollType::Wait is a struct variant, not a unit one.
device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
```

**wgpu is pinned to 29 by Slint**, which offers `unstable-wgpu-28` and `-29` and nothing
newer. Bumping it without bumping Slint stops the types unifying across the GUI boundary.

### Never take "adapter 0"

On the reference machine Vulkan offers an RTX 5090, a 2-CU Radeon iGPU and llvmpipe, and
the measured matting gap between the first two is **4.53 ms against 38.77 ms** — the same
binary, the same model. Rank explicitly, and report what was chosen:

```rust
described.sort_by_key(|(a, _)| match a.get_info().device_type {
    wgpu::DeviceType::DiscreteGpu => 0,
    wgpu::DeviceType::IntegratedGpu => 1,
    wgpu::DeviceType::VirtualGpu => 2,
    wgpu::DeviceType::Other => 3,
    wgpu::DeviceType::Cpu => 4,   // lavapipe: last, but kept for headless CI
});
```

A pin that cannot be honoured is an **error**, never a fallback — someone who pinned a GPU
would otherwise never learn they were ignored.

### One *declaration* per binding slot, not one entry point per file

A binding slot can only be declared once in a module, so `@group(0) @binding(0)` written
twice fails to compile. That is the actual constraint, and it is easy to over-read as "one
entry point per file" — `blur.wgsl` has three, sharing the same declarations.

What matters when entry points in one module need *different* bindings: with `layout: None`,
wgpu derives each pipeline's bind group layout from the resources that entry point actually
uses, not from everything declared in the module. So `blur.wgsl` can declare a matte at
binding 4 that only `down_weighted` reads, and the `down` and `up` pipelines validate without
it rather than demanding a binding they never touch. Verified on this stack; do not assume it
without checking, because the failure would be a validation error at pipeline creation.

Genuinely shared *code* still has to be concatenated by the host — the colour matrix lives in
`shaders/colour.wgsl` and is prepended, because naga has no `include`.

### Readback rows must be 256-byte aligned

```rust
let padded = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
    * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
// …then un-pad row by row when copying out.
```

### A ping-pong off-by-one produces a plausible, wrong picture

The blur ran correctly and was then **discarded**, because the code derived which texture
held the result from `passes % 2` and was wrong by one — so the composite sampled the
*input* to the final pass. It showed up as a 7% variance drop where there should have been
14%: no error, no crash, just a slightly-too-sharp background. Track it explicitly:

```rust
let mut blur_in_a = true;
for _ in 1..passes {
    let (src, dst) = if blur_in_a { (&a, &b) } else { (&b, &a) };
    blur_pass(src, dst);
    blur_in_a = !blur_in_a;
}
let bg = if blur_in_a { &a } else { &b };
```

### Out-of-gamut test colours read as a colour-space bug

A round-trip test built in YUV directly produced max luma error 15 and looked like a broken
matrix. It wasn't: arbitrary Y/Cb/Cr combinations fall outside the RGB gamut, the shader
clamps them correctly, and the round-trip cannot recover them. Build test frames from **RGB**
and convert with the same matrix — then the error is **0**.

### Blurring the whole frame puts the subject inside its own background

Reported as "a smeared copy of me slides around behind me when I move quickly".

The obvious way to build a background blur is: blur the frame, then composite the sharp
subject over it with the matte. That is wrong, and it is wrong in a way that only shows up
under motion. The blurred plane contains a low-frequency copy of the *subject*, so when they
move, their own smear slides around behind them. It also haloes: the subject's colours bleed
outward across the silhouette, which is the bright ring people notice against a dark wall.

Quantified here by mutation — a white subject against a mid-grey backdrop, sampling the
background just outside the silhouette:

|  | background luma | true backdrop |
|---|---|---|
| blur the whole frame | **173** | 125 |
| weighted | **125** | 125 |

Forty-eight levels of subject were being mixed into its own background.

**The fix is normalized convolution, and on a pyramid it is nearly free.** Accumulate each
tap as `(rgb * (1 - alpha), 1 - alpha)` and divide the weight back out at the end. Two things
make it cheap: the pyramid textures are already RGBA with an unused alpha channel, and every
pass after the first sums `vec4` linearly, so the weight propagates through the rest of the
pyramid with no code at all. Only level 0 — the one pass that reads the full-resolution
frame — needs a weighted variant.

Two things that are easy to get wrong:

**The pyramid has to be `Rgba16Float`, not `Rgba8Unorm`.** The weight is divided back out, so
quantisation error is amplified by `1/w`. An isolated background pixel in a narrow gap — say
between an arm and the torso — sits near `w = 0.05`, where 8-bit recovers with about ten
levels of error and bands visibly across a region that is supposed to be smooth.

**Guard the zero-weight case.** With no matting model loaded the matte is a single opaque
texel, so every tap has zero background weight and the divide falls on its `max()` floor.
That is harmless *only* because alpha is 1 there and the composite takes the foreground
whole — the meaningless background is never sampled. Worth an assertion rather than a
comment: a regression would black the entire frame the moment the model failed to load.

The exclusion mask can come from the raw matte rather than the guided filter's refined
alpha. It is about to be blurred by the whole pyramid, so its edge precision is irrelevant,
and a bilinearly-stretched matte gives a *softer* exclusion than a hard silhouette would.

### Adapter selection means throughput checks measure the wrong GPU

`gpu_check` called `Gpu::new(None)`, which uses the same "prefer discrete" selection the
daemon does. On the reference machine that always picks the RTX 5090, so the slow-hardware
conformance target — the Radeon iGPU, 3-4x the frame time — was never exercised and
"fits 30fps" was being answered for the wrong device. It takes an optional render node now:

```sh
nix develop -c cargo run --release -p cleanroom-gpu --example gpu_check -- /dev/dri/renderD129
```

The difference is the whole point of measuring. Adding the weighted blur above cost nothing
measurable on the 5090 (0.86 → 0.85 ms at max blur, inside noise) and 0.29 ms on the iGPU
(2.77 → 3.06). Both fit; only one of them tells you anything.

---

## ONNX Runtime / matting

### The matte was a frame behind the image, and every throughput metric said "fine"

Reported as "the model doesn't seem to be working that quickly — weird fringing and lagging
when I move around". The obvious reading is that inference is missing its budget. It was
not: the daemon reported **29.9 fps, 0 dropped, decode 7.47 ms + gpu 1.46 ms + matting
10.40 ms = 19.3 ms** inside a 33.3 ms frame. Fourteen milliseconds of headroom.

The defect was not speed but **alignment**. `run_once` composited the frame *first* and
inferred *afterwards*, so the alpha produced from frame N was uploaded for frame N+1. At
30 fps that is 33.3 ms of misalignment between an image and its own alpha:

- **Leading edge of motion** — the subject has moved into pixels the matte does not cover,
  so they get blurred along with the room.
- **Trailing edge** — the matte still claims foreground where the subject has left, so a
  sharp band of stale background is keyed in and drags behind the movement.

It was justified in a comment as invisible, and as avoiding a pipeline stall. Neither held.
It was not invisible — it is the whole reported symptom. And it avoided no stall: `process`
already blocked on a readback and `read_matte_input` blocked on a second, for **three**
queue submissions per frame.

**The guided filter amplifies it rather than hiding it.** `run_guided` fits its
coefficients on the correct pair — the guidance image and the matte from the *same* frame —
but the composite *consumed* them a frame later, evaluating `alpha = a*I + b` against the
next frame's luma. `a` is large at an edge by construction, because that is what makes the
alpha track luminance. A stale high-gain local linear model does not decay into soft
ghosting; it extrapolates, and a moving edge picks up a luminance-keyed fringe. That is why
the artifact reads as *weird* rather than merely delayed.

**The fix is pure reordering** — `unpack -> downscale -> infer -> composite`, as
`begin_frame` / `set_matte` / `finish_frame`. Same passes, same readback bytes, same two
stalls, one *fewer* submission, and no added end-to-end latency because `sink.write`
already ran after inference. Measured before and after on the reference machine:

| | fps | decode | gpu | matting | total | dropped |
|---|-----|--------|-----|---------|-------|---------|
| before | 29.9 | 7.47 | 1.46 | 10.40 | 19.33 | 0 |
| after | 29.9–30.1 | 7.43 | 1.65 | 9.91 | 18.99 | 0 |

`gpu` rises and `matting` falls by about the same amount: the guided pass moved out of the
matting span and into the composite's command buffer, where it belongs.

`finish_frame` deliberately takes **no frame argument**. The bug was never in
`FramePipeline`, which faithfully applied whatever matte was set — it was in the caller's
ordering. Removing the frame parameter is what stops that ordering being something a caller
can get wrong, and `read_matte_input` is gone so "composite now, infer later" no longer has
a way to spell itself.

**The lesson worth keeping:** fps and dropped-frame counters cannot see a frame-offset bug.
Both were perfect throughout. A pipeline can be exactly on time and still be wrong.

### Temporal smoothing was not the culprit, and measuring in frames says otherwise

The obvious second suspect was the alpha EMA (`ALPHA_FALL = 0.22`, `MOTION_FULL = 0.25`),
which does lag: 1.27 frames on a slow trailing edge. That number is misleading on its own.
What anyone *sees* is the edge in the wrong place, and `pixels = frames * speed` — the two
terms move in opposite directions, so the slowest motion lags the most frames and the
fewest pixels:

| speed (matte px/frame) | fall lag | displacement | 1-frame-late matte |
|---|---|---|---|
| 0.25 | 1.27 fr | **0.32 px** | 0.25 px |
| 1.0 | 0.34 fr | **0.34 px** | 1.00 px |
| 4.0 | 0.00 fr | **0.00 px** | 4.00 px |

Worst case the filter displaces the edge by 0.38 matte pixels — about 1.4 px at 1080p —
against a full `speed` pixels for the misalignment above. Two orders of magnitude apart at
any realistic speed. The constants were left alone and exposed as `video.matte_fade_*` for
taste; `temporal_smoothing_costs_well_under_a_pixel` is the guard against retuning them
into a smear. **Pick the unit the user actually perceives before concluding anything.**

### `tighten` sharpens the edge while eroding it, which is the opposite of softening

`alpha = (alpha - t) / (1 - t)` is a shift *and* a rescale, so its gain is `1/(1 - t)`. A
hand-tuned `matte_tighten = 0.34` — reached for to fight the fringing above — was also
making the ramp 51% steeper. Anyone chasing a softer edge with this knob is turning the
wrong one, and turning it the wrong way.

`video.matte_feather` is the width control: it widens the ramp about whatever crossing
`tighten` chose, leaving the crossing alone. It lives in the composite shader rather than
in the matting crate on purpose — anything softened at 512x288 is re-sharpened by
`a*I + b` on the way up to full resolution.

### You cannot feather an edge by remapping alpha values

The correction to the claim above, which was true in intent and false as shipped. The first
implementation of `matte_feather` widened the ramp in *alpha space*:

```wgsl
let c = (1.0 + comp.tighten) * 0.5;
let w = clamp((1.0 - comp.tighten) * 0.5 * (1.0 + 3.0 * comp.feather), 0.01, 0.5);
alpha = smoothstep(c - w, c + w, alpha);
```

Evaluated across the slider's actual range, `w` never moves:

| feather | tighten=0.0 | tighten=0.12 | tighten=0.34 |
|---|---|---|---|
| 0.05 | w=0.5 | w=0.5 | w=0.3795 |
| 0.1 | w=0.5 | w=0.5 | w=0.429 |
| 1.0 | w=0.5 | w=0.5 | w=0.5 |

At `tighten = 0` — the default for blur — `(1 - 0) * 0.5` already sits on the clamp ceiling,
so **every non-zero feather produces the identical curve**. The control had two states, off
and one fixed shape. For replace it saturated by about 0.05, so 95% of the travel was inert.

And the shape it reached was the wrong one. `smoothstep(0, 1, a)` has gradient 1.5 at the
midpoint, so the knob labelled "feather" made the edge **harder** through the transition and
softened only the tails.

**The clamp was not the bug; it is the boundary of the technique.** Remapping alpha *values*
can move where the edge sits and reshape its profile, but the transition still lands on
exactly the same pixels. `w = 0.5` centred at `0.5` already spans the whole `[0,1]` alpha
range — there is nothing left to widen. Feathering is inherently a *spatial* operation: a
pixel's alpha has to be influenced by its neighbours' or the edge cannot get wider.

So it is now an average of the resolved alpha over a disc, in the composite, at full
resolution — full resolution for the reason in the entry above, that anything softened at
512x288 is re-sharpened on the way up.

The sample pattern took two attempts, and the failure is the interesting part. A hexagonal
ring of six taps plus the centre *looks* like a disc, but projected onto an edge normal it
collapses to three distinct offsets — `0`, `±r/2`, `±r` — with a hole between them. Measured
on a hard test edge that gave alpha 0.14 / 0.43 / 0.57 / 0.86 and nothing in between: a wide
edge made of four coarse steps, which reads as banding, not softness. A twelve-tap Vogel
disc (radius `sqrt((i + 0.5)/N)`, golden-angle rotation) spreads taps evenly, so every normal
direction sees twelve distinct offsets:

```
feather 0     width 0   235 235 235 235 145 145 145 145
feather 0.25  width 2   235 235 235 217 160 145 145 145
feather 0.5   width 4   235 230 203 170 151 145 145 145
feather 1     width 6   234 226 214 193 178 162 154 147
```

**Measure the profile, not just the width.** A ramp that is wide but takes three values is
not a soft edge, and a width figure alone cannot tell the two apart.

One trap in testing it: the split white/black frame the other GPU tests use puts a *colour*
edge in the same place as the alpha edge, and `Remove` then mixes flat green against black,
so composited luma dips **below** the background plateau — 97 in the middle of a 145..235
ramp. Any threshold-based band detector silently loses half the transition. Use a uniform
frame and let luma be a direct readout of alpha.

### A knob that writes, persists and reads back, and still does nothing

`video.guided_filter`, `video.guided_radius` and `video.guided_eps` were write-only for
their whole existence. `pipe.set_guided(...)` was called exactly once, before the frame loop,
from the config snapshot `run_once` opened with, and never again.

This is worse than a control that plainly does nothing, because every check a user can run
says it worked: the value validates, persists to `config.toml`, and `cleanroom-ctl get` and
the GUI slider both read it back correctly. It starts taking effect only when something
*unrelated* restarts the pipeline — so toggling blur off and on makes it appear to work.

The gap is structural, and worth understanding before adding a knob. `needs_restart` is a
**deny-list**: a setting it does not name is *assumed* live. For almost everything that
assumption holds for free, because `Look` and `Smoothing` are rebuilt from the live config
every frame and cost nothing to pass again. Anything that instead *reconfigures* the
pipeline needs its previous value remembered so a change can be detected — which
`audio_pipeline` has always done correctly for `attenuation_db`:

```rust
if c.audio.denoise.attenuation_db != applied_atten {
    applied_atten = c.audio.denoise.attenuation_db;
    d.set_attenuation(applied_atten);
}
```

`LiveSettings` in `video_pipeline.rs` is that, for the guided fields. Note what the fix is
*not*: adding them to `needs_restart`. A restart drops the loopback fd, and with
`exclusive_caps=1` the node reverts to output-only, so every app in the call loses the
camera — see the entry on that below. Reopening the devices to change a filter radius would
be a spectacular price.

The guard is `every_video_setting_is_classified_as_live_restart_or_inert`: it walks
`settings::keys()` and fails unless every `video.*` key is named in exactly one of three
lists. Adding a knob now fails the build until somebody classifies it. Its limit is worth
stating — it proves a key was *classified*, not that the frame loop reads it.

### Adding a config field is not a safe *downgrade*

`Config` is `deny_unknown_fields`, and the daemon refuses to start on a config it cannot
parse rather than overwrite it with defaults. Both are right. Together they mean a **newer
daemon writing a new key takes the older binary down**, hard:

```
config is corrupt; attempting to recover from backup
backup is also unparseable
Error: unknown field `matte_fade_fall`, expected one of `enabled`, `device`, ...
```

Hit for real while A/B-testing this change: the dev build wrote `matte_fade_*` into
`~/.config/cleanroom/config.toml`, and handing the devices back to the packaged daemon left
no camera at all. The backup does not help, because it gets rewritten too.

Adding fields is still forward-compatible — `#[serde(default)]` on every new one means an
*older* config loads fine in a newer daemon, which is the direction that matters for
upgrades. Just know that rolling back needs the new keys removed first:

```
sed -i '/^matte_feather = /d; /^matte_fade_/d; /^matte_motion_release = /d' \
    ~/.config/cleanroom/config.toml
```

### A Conv with C_in divisible by 3 but not by 4 is computed wrong on the WebGPU EP

**This is the root cause of the entry below, found after every whole-model hypothesis had
been eliminated. Fix first, then read the rest as the trail.**

ONNX Runtime's WebGPU provider returns wrong values from `Conv` when the **input channel
count is divisible by 3 and not divisible by 4**. Minimal reproducer: one dense 3x3 Conv,
16 output channels, 16x16 spatial, varying only `C_in`:

| C_in | 3 | 4 | 5 | **6** | 7 | 8 | 12 | 59 | 107 | **171** | 172 |
|------|---|---|---|-------|---|---|----|----|-----|---------|-----|
|      | ok | ok | ok | **wrong** | ok | ok | ok | ok | ok | **wrong** | ok |

Across a 1..176 sweep the failing set is exactly `6, 9, 15, 18, 21, 27, 30, 33, 39, 171` —
17-20% off in L1. Every multiple of 4 is exact. `C_in = 3` is special-cased upstream and
works. Reproduced identically on ORT **1.24.2 and 1.27.0**.

RVM walks into this once: its decoder concatenates the 3-channel source image onto its skip
connections, and `Conv_200` (`W = [80, 171, 3, 3]`) lands on 171. **One** conv out of 353
nodes, and it destroys the whole matte.

**The fix is an exact identity.** Pad the input channels up to the next multiple of 4 with
an ONNX `Pad`, and zero-pad the weight tensor along `C_in` to match. The appended weights
are zero, so the appended channels contribute nothing whatever they hold; only the kernel
path the provider takes changes.

```sh
python3 tools/onnx/pad_conv_channels.py in.onnx out.onnx    # 171 -> 172
```

Measured on the same frame, 512x288, 10 passes, RTX 5090:

| model | provider | fg>0.5 | mean | max | ms/frame |
|-------|----------|--------|------|-----|----------|
| stock | WebGPU | 0.000 | 0.002 | 0.185 | 6.71 |
| padded | **WebGPU** | **0.227** | **0.227** | **1.000** | **7.34** |
| padded | CPU | 0.227 | 0.227 | 1.000 | 41.06 |

Padded-WebGPU matches CPU exactly — the rewrite really is an identity — and is 5.6x faster
than the CPU provider. GPU matting needs no CUDA and no loss of vendor neutrality.

**How it was found**, because the method generalises: sub-model extraction with
`onnx.utils.extract_model`, comparing each probe across providers
(`tools/onnx/extract.py`, `examples/ep_diff.rs`). Every probe matched to ~1e-7 up to and
including `Concat_199` — the failing conv's own *input* — and then `Conv_200`'s output was
44% off. One caveat that wasted a pass: compare with a **relative tolerance** (~1e-3).
Floating-point reassociation alone moves an L1 sum by ~1e-7, so exact equality marks every
probe as divergent and localises nothing.

### The WebGPU EP runs RVM fast and returns a matte that decays to nothing

**Symptom:** background blur appears not to work. Look closely and it *is* working — the
whole frame is blurred, the subject included. It looks fine while somebody walks around the
room and fails once they sit still, which is to say it fails during an actual video call and
passes every casual test.

**Cause:** ONNX Runtime's WebGPU provider computes this model wrong. The same still frame,
fed repeatedly through each provider:

| pass | 1 | 2 | 3 | 5 | 10 |
|------|---|---|---|---|----|
| WebGPU mean alpha | 0.039 | 0.027 | 0.018 | 0.005 | **0.002** |
| CPU mean alpha | 0.227 | 0.227 | 0.227 | 0.227 | **0.227** |

Read the first column before the trend: **pass 1 is already wrong**, with an all-zero
`(1,1,1,1)` state that both providers share, so this is a fault in the *forward* pass and
not in the recurrent hand-off. The recurrence only feeds an already-wrong result back into
itself, which is what drives it to zero over the following seconds — and that is why the
symptom looks like "it works until you sit still".

Alpha ≈ 0 means "every pixel is background", so the composite blurs everything. Nothing
errors, the session builds, and `matting_ms` looks excellent — 7 ms against the CPU's 39 ms.

**A newer runtime does not fix it.** ONNX Runtime 1.27.0 — a native WebGPU build with Dawn
statically linked, taken from the `onnxruntime-webgpu` PyPI manylinux wheel and loaded
through `ort`'s `load-dynamic` with `ORT_DYLIB_PATH` — returns **bit-identical** wrong
output to the pinned 1.24.2. Three minor versions apart, the same 0.002 / 0.185. That wheel
is the cheapest way to re-test this in future: no building, no Dawn, just point the env var
at the `.so` and run `matte_sweep`.

Ruled out, each by measurement rather than reasoning: `downsample_ratio` (wrong at 1.0, 0.5
and 0.25), input resolution (wrong at 512x288 and 1920x1080), and adapter selection
(identical with `VK_DRIVER_FILES` pinned to `nvidia_icd.json`).

`.error_on_failure()` cannot catch this, because nothing fails. The only detector is a
second opinion, so `video.matting_backend = auto` watches the matte for as long as the GPU
runs and cross-checks against a CPU session when the subject disappears for 45 consecutive
frames.

**Two things that make the check work, both learned by getting them wrong first:**

* **Measure coverage, not peak alpha.** A dead matte still throws stray texels above 0.5, so
  a peak-based test declared the provider healthy and latched while every frame came out
  uniformly blurred. The fraction of pixels above 0.5 separates cleanly: 22.7% versus 0.0%.
* **Verify continuously, not once at startup.** Early frames still carry signal, so a
  one-shot check passes and then latches — and the defect appears seconds later, exactly
  when the subject stops moving.

`examples/matte_sweep.rs` is the harness: one provider/ratio/resolution per process, since
two ORT sessions in one process cannot be torn down. `examples/matte_probe.rs` prints the
alpha histogram and a spatial map for a single still.

#### What has been ruled out

Everything below produces **bit-identical** wrong output — `fg>0.5` 0.000, mean 0.002,
max 0.185 — on the same frame where the CPU provider returns 0.227 / 1.000:

| tried | values | verdict |
|-------|--------|---------|
| `preferredLayout` | NCHW, NHWC | not layout conversion |
| `GraphOptimizationLevel` | Disable, Level1, default | not a fusion |
| `enableGraphCapture` | off | not command-buffer replay |
| buffer cache modes | disabled | not buffer reuse |
| `downsample_ratio` | 1.0, 0.5, 0.25 | not the ratio |
| `src` size | 512x288, 1920x1080 | not resolution |
| `VK_DRIVER_FILES` | pinned to nvidia_icd | not adapter selection |
| Resize rewritten | `half_pixel`, `asymmetric` | not the coordinate mode |
| HardSigmoid rewritten | `Clip(alpha*x + beta, 0, 1)` | not the non-default alpha (1/6) |
| Split rewritten | axis `-3` normalised to `1` | not negative-axis handling |
| recurrent state seed | full-size zeros instead of `(1,1,1,1)` | not the `Expand` broadcast |
| ONNX Runtime version | 1.24.2 and 1.27.0 | not a fixed-since bug |

`auto_pad` is ruled out by inspection rather than experiment: all 85 Convs use explicit
pads (`NOTSET`) and none combines `auto_pad` with a stride above 1, so ORT issue #26734
does not apply here.

The prime remaining suspect is **depthwise/grouped Conv**. The model has 19 depthwise
convolutions and grouped convolutions at group 4, 16, 64, 72, 120, 184, 200, 240, 480, 672
and 960 — MobileNetV3's inverted residual blocks — and those take a different kernel path
in the WebGPU EP from dense convolution. Upstream has a documented pattern of WebGPU Conv
correctness bugs (microsoft/onnxruntime #26734, #24442, #24070). Confirming it needs a
minimal reproducer: one depthwise Conv in a two-node model, run on both providers.

Each graph rewrite is an exact identity, and each was confirmed a no-op by re-running it on
the CPU provider, which returned 0.227 unchanged. `scratchpad`-style helpers for this live
in the commit history: `rewrite.py` (HardSigmoid/Split), `patch_resize.py`.

Two further notes for whoever picks this up:

* The model's 353 nodes **are** all named (`Resize_3`, `Conv_12`, …) — an earlier claim here
  that they were unnamed was wrong, and came from reading `strings(1)` output instead of the
  graph. But `forceCpuNodeNames` appears to be **inert**: forcing *all 353* nodes to the CPU
  still returns the wrong answer at full GPU speed (7 ms), which is the control that says
  the option is not being honoured. Per-op isolation through that lever is unavailable.
* The output *is* input-dependent — max alpha 0.279 for a portrait, 0.065 for a blurred
  room, 0.000 for a screenshot — so the input tensor does reach the kernels. The fault is a
  systematic attenuation that then compounds through the recurrent state, not a dead input.

The remaining rigorous approach is sub-model extraction: truncate the graph at successive
tensors with `onnx.utils.extract_model`, run each on both providers and find the first
divergence. That needs a generic runner, since `matte_sweep` assumes RVM's exact signature.

TensorRT is separately ruled out for this model by the shared symbolic dimension names
described at the top of `cleanroom-matting`'s module docs.

Re-test on a new `ort` or a new Dawn before trusting the GPU path again.

### `gpu_ms` used to include `matting_ms`

The GPU timer spanned the whole match arm, which contains the matting block, so the two
stats double-counted: the GPU column read 10.8 ms against ~0.65 ms of real GPU work and
looked like a GPU problem. It is subtracted now. Worth remembering before optimising
anything on the strength of one of these counters.

### Run the probe *inside* `nix develop`, or it measures nothing

Running `target/release/examples/matte_sweep` directly with only `LD_LIBRARY_PATH` set to
`target/release` fails to find `libvulkan.so.1`, so the WebGPU provider never registers.
`.error_on_failure()` turns that into a panic — and the panic then *hangs* rather than
exiting, because Dawn is mid-initialisation, which reads as "inference is extremely slow"
rather than as "there is no GPU here". Two of these were misread as a hung model before the
stderr was looked at directly instead of through a `grep`.

### nixpkgs' onnxruntime has no WebGPU EP

```sh
nix eval --json nixpkgs#onnxruntime.override.__functionArgs
# coremlSupport, cudaSupport, ncclSupport, openvinoSupport, rocmSupport … no webgpuSupport
```

There is no Dawn in the store either. Pointing `ORT_DYLIB_PATH` at it silently gives **CPU**
inference. `ort`'s `download-binaries` prebuilt is the supported path and does contain
WebGPU.

### Always `.error_on_failure()`

Without it, `ort` silently registers nothing and runs on the CPU:

```rust
use ort::ep::{ExecutionProvider, WebGPU};

let session = Session::builder()?
    .with_execution_providers([WebGPU::default().build().error_on_failure()])?
    .commit_from_file(model)?;
```

This caught a real fault within an hour of being written (Dawn could not find
`libvulkan.so.1`) that would otherwise have produced a fabricated GPU number.

Note `ort::ep`, not `ort::execution_providers` (deprecated), and `error_on_failure()` is on
the **dispatch** returned by `build()`, not on the EP.

### Dropping a session that owns a Dawn context segfaults — *at process exit*, not in `drop`

Reproducible on both Nvidia and AMD, **after** all inference succeeds. Under
`Restart=on-failure` a segfault at exit is a failed exit, so every clean shutdown becomes a
restart loop.

```rust
impl Drop for Matter {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            std::mem::forget(session);   // deliberate; OS reclaims at exit
        }
    }
}
```

Measured precisely by `examples/teardown_check.rs`, which exists so this can be re-tested
against a new driver, `ort` or Dawn without editing library code:

```sh
nix develop -c cargo run --release -p cleanroom-matting --example teardown_check
# ...
# dropping the session for real...
# INFO ort::logging: WebGPU device lost (2): Device was destroyed.
# SURVIVED — a controlled teardown works; the leak in Drop can be removed.
# exit: 139
```

Note what that says. The **drop itself completes** — Dawn reports the device destroyed and
the program runs to the end of `main`. The SIGSEGV lands *afterwards*, during process
teardown, in global destructors. So `mem::forget` does not avoid a crash inside `drop`; it
prevents ORT's destructor from ever running, which leaves Dawn's own exit-time cleanup with
nothing to trip over.

That distinction matters if you try to fix this: an ordered shutdown inside the daemon will
not help, because the fault is not in the ordering of *our* drops. Control: the default
(leaking) path exits `0` — `cargo run --example two_sessions` confirms it.

`CLEANROOM_DROP_ORT_SESSION=1` switches any binary to the honest path for testing.

### Creating two such sessions *concurrently* aborts (SIGABRT)

Sequentially is fine — `crates/cleanroom-matting/examples/two_sessions.rs` proves it. But
cargo runs tests on parallel threads, so **the two model-dependent tests are deliberately
merged into one**. Splitting them reintroduces the crash.

A comment cannot enforce that, so `Matter::new` now takes a process-wide mutex across
construction only:

```rust
static SESSION_INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
// in Matter::new, before building the session:
let _serialise = SESSION_INIT.lock().unwrap_or_else(|e| e.into_inner());
```

An abort is not catchable and unwinds nothing, so the failure mode is a CI job that dies
with no stack and no test name. Serialising makes a second caller slow rather than fatal.

### RVM's reused symbolic dimensions

The export gives `src` and every `r1i`…`r4i` the *same* symbolic H/W names:

```
src  Tensor { shape: [-1, 3, -1, -1], dimension_symbols: ["batch_size", "", "height", "width"] }
r1i  Tensor { shape: [-1, -1, -1, -1], dimension_symbols: ["batch_size", "channels", "height", "width"] }
```

TensorRT reads those as equality constraints and refuses to build an engine. **WebGPU's
shape inference does not**, so no rewriting is needed here — but it will bite anyone trying
a different EP.

### Recurrent state hand-off

```rust
// Seed: a (1,1,1,1) zero tensor tells RVM to auto-shape its state on the first frame.
let mut state: Vec<(Vec<i64>, Vec<f32>)> =
    (0..4).map(|_| (vec![1, 1, 1, 1], vec![0.0f32])).collect();

// Each frame: r*o becomes the next frame's r*i. This is what makes edges stable.
for (i, name) in ["r1o", "r2o", "r3o", "r4o"].iter().enumerate() {
    let (shape, data) = outputs[*name].try_extract_tensor::<f32>()?;
    state[i] = (shape.iter().copied().collect(), data.to_vec());
}
```

Observed shapes at 512×288: `[[1,16,36,64], [1,20,18,32], [1,40,9,16], [1,64,5,8]]`.
Reset to the seed on any geometry change, or the next frame is a shape error.

### The degenerate-alpha guard tests flat **and high**

Fed a blank frame, RVM emits a near-uniform *high* alpha, which composites as a one-frame
"effect off" flash. A flat *low* alpha is not a fault — that is the network correctly
reporting no subject, and it is the right answer for a synthetic test frame.

```rust
let degenerate = frames > 0 && (hi - lo) < 0.005 && lo > 0.5;
```

An earlier version tested spread alone and cried wolf on every run.

---

## V4L2

### Read `device_caps`, not `capabilities`

`capabilities` is the union across **every** node the driver owns, so a metadata node
cheerfully reports `VIDEO_CAPTURE` because a sibling has it. Only `device_caps` describes
the node you opened, and only when `V4L2_CAP_DEVICE_CAPS` is set:

```rust
let effective = if caps.capabilities & V4L2_CAP_DEVICE_CAPS != 0 {
    caps.device_caps
} else {
    caps.capabilities
};
```

Read the wrong one and every UVC camera appears twice with half the entries broken.
`/dev/video1` on the C922 is a metadata node; sysfs cannot tell you this, so the ioctl is
mandatory.

### MJPEG is the *preferred* format, not a fallback

On the C922, 1080p is 30 fps over MJPG and **5 fps** over YUYV, because raw 4:2:2 saturates
USB 2. The mode ladder ranks `Mjpeg` first for exactly this reason.

### `S_FMT` before creating the stream

v4l2loopback only sets the buffer `length` field once a format is set, so creating the
stream first yields zero-sized buffers and a silent black camera. The `v4l` crate's own
example flags this under "BEWARE OF DRAGONS".

### The `exclusive_caps=1` deadlock — prime the sink on open

A loopback node advertises `VIDEO_OUTPUT` until a producer **actually starts streaming**;
holding the fd open is not enough. Combined with power save this deadlocks: idle at startup
means no frames are written, so the node never flips to `VIDEO_CAPTURE`, so no consumer can
open it (`Not a video capture device`), so no consumer event ever fires to wake anything.

```rust
// In LoopbackSink::open, after S_FMT — NOT optional.
sink.write_placeholder()?;
```

It also makes the camera present *before* Zoom, Chrome and Discord launch, which matters
because they enumerate cameras once at startup and never look again.

### Publish YUY2

The one format both Firefox and Chromium accept without complaint. NV12-only loses Firefox;
I420-only makes Chromium convert. A test guards the constant.

### Consumer detection has exactly one correct mechanism

- `fuser` / scanning `/proc/*/fd` returns a confident **"no consumers"** inside a bubblewrap
  namespace, because the `/proc/PID/fd` magic links fail the kernel's ptrace-mode check.
- inotify open/close counting **drifts low**: the kernel coalesces adjacent identical
  events, and browsers probe-open cameras while a capture fd is already open.
- `V4L2_EVENT_PRI_CLIENT_USAGE` is an absolute count from the kernel via the device fd, and
  fires on `STREAMON`/`STREAMOFF` rather than `open()` — so probe-opens correctly do not count.

```rust
// Defined in v4l2loopback.c, not in any UAPI header — must be hardcoded.
const V4L2_EVENT_PRI_CLIENT_USAGE: u32 = 0x0800_0000 + 0x08E0_0000 + 1;  // 0x10E00001
// Ubuntu shipped a downstream variant at the bare private-start value; try new then old.
const V4L2_EVENT_PRI_CLIENT_USAGE_LEGACY: u32 = 0x0800_0000;
const V4L2_EVENT_SUB_FL_SEND_INITIAL: u32 = 0x1;   // or you sit at "unknown" until a change
```

Events arrive as **POLLPRI**, not ordinary readable. `count` is at byte 8 of
`struct v4l2_event` and `pending` at byte 72; `sizeof` is 136 on 64-bit — pinned by a test,
because if that drifts the count becomes plausible nonsense rather than an obvious failure.

**Contract: unknown counts as in-use.** A detection failure must never blank a camera
mid-call.

### A solid green frame means "zeroed buffer", not "green-screen mode"

All-zero YUY2 decodes to mid-green through BT.601 limited range, because `Y=0` clamps to
black luma while `U=V=0` puts both chroma channels at −128:

```
G = 1.164(0-16) - 0.813(0-128) - 0.391(0-128) ≈ 135     R, B clamp to 0
```

This is almost exactly what `BackgroundMode::Remove` produces on purpose, so the two are
easy to confuse. If a consumer shows green, check whether it is receiving buffers the
producer never wrote before assuming the mode is wrong.

### Changing the capture geometry green-screens consumers that are already streaming

V4L2 has no mid-stream format renegotiation. An app that negotiated 1920x1080 and is
mid-`STREAMON` cannot follow the producer to 1280x720; it keeps dequeuing buffers that are
never filled and shows the green above — measured at **207 of 360 frames, with no
recovery** across a `set video.width` while `ffmpeg` was reading.

Same-geometry restarts are fine. So the restart predicate must stay as narrow as possible
(`needs_restart` in `video_pipeline.rs`, unit-tested both ways): blur strength, mirror,
power save and Blur/Replace/Remove are all read per-frame and must never restart, because a
restart is visible to everyone in the call.

### Testing a v4l2loopback device wedges it

Rapid producer open/close cycles leave the node in a state where a consumer opens it
successfully, receives **exactly one frame**, then hits EOF — while the daemon reports "no
consumers" and never wakes from power save. It looks exactly like a broken change.

It is not: an A/B/A against an unmodified `HEAD` build in a throwaway `git worktree` gave
300 frames / 1 frame / 300 frames for the same binaries, i.e. the variable was the device,
not the code. Before concluding a change broke frame delivery, re-run the *old* binary
under the *current* device state.

### `/dev/v4l2loopback` is root-only

```
crw------- 1 root root 10, 263 /dev/v4l2loopback
```

A user daemon cannot create devices, so the plan's "allocate at runtime" is impossible.
Devices are provisioned at boot and the daemon *selects* a free one — which is better
anyway: no privilege escalation, and no fight with OBS. "Free" is detectable precisely
because of the `exclusive_caps` flip above.

`options` lines in `/etc/modprobe.d` are **per-module and global**: modprobe merges every
file into one argument list where `video_nr`, `card_label` and `exclusive_caps` are parallel
arrays, so two packages each declaring a device usually yields neither.

---

## PipeWire

### The v4l2 SPA node cannot deliver MJPEG

Its `EnumFormat` advertises YUY2 only and does not pass a UVC camera's MJPG modes through,
so anything capturing via PipeWire is structurally pinned to ~5 fps at 1080p. There is no
"prefer MJPEG" knob. This is why Cleanroom opens `/dev/video*` directly.

### The WirePlumber rule MUST match `node.nick`

```
monitor.v4l2.rules = [
  { matches = [ { node.nick = "C922 Pro Stream Webcam" } ]
    actions = { update-props = { node.disabled = true } } }
]
```

`media.class` is **not** set when the v4l2 `create-node` hook evaluates rules, so a rule
matching it silently never fires. Valid keys there: `node.name`, `node.nick`,
`node.description`, `api.v4l2.path`.

Also disable the libcamera monitor: it double-enumerates UVC cameras, exposes only
RAW/YUYV, and holds the device fd open.

### pipewire-rs 0.10 uses `Rc` variants, behind a version feature

```toml
# Without a version feature the constructors have different signatures entirely and
# TARGET_OBJECT does not exist.
pipewire = { version = "0.10", features = ["v1_2_0"] }
```

```rust
let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
let context  = pipewire::context::ContextRc::new(&mainloop, None)?;
let core     = context.connect_rc(None)?;
let stream   = pipewire::stream::StreamRc::new(core.clone(), "name", props)?;
```

`pipewire::filter::Filter` has **never** been wrapped — requested since 2021, three unmerged
attempts. Use two `Stream`s and a ring buffer, which is what `module-loopback` does anyway.

### The properties that make a virtual microphone

```rust
properties! {
    *pipewire::keys::MEDIA_TYPE     => "Audio",
    *pipewire::keys::MEDIA_CATEGORY => "Playback",     // we PRODUCE, even as a source
    *pipewire::keys::MEDIA_CLASS    => "Audio/Source", // NOT Audio/Source/Virtual
    *pipewire::keys::NODE_NAME      => "cleanroom_mic",
}
// …and Direction::Output, because producing data means Output even for a *source* node.
```

`Audio/Source/Virtual` has the opposite port direction and is invisible to QtWebEngine and
Electron clients. `AUTOCONNECT` is inert on a source: WirePlumber classifies it as a
*device*, so it **is** a link target rather than something seeking one.

### `node.passive = true` gives a permanently silent microphone

It was set on the capture stream intending "do not hold the mic when nobody is listening".
A passive node will not drive the graph — and because the capture and source streams are two
**independent** nodes joined only by a userspace ring buffer, not by a PipeWire link,
nothing else was ever going to drive it. It sat paused forever while the hardware mic read a
healthy −40 dBFS.

Use `"node.always-process" => "true"`. Releasing the mic when idle needs a registry watcher
on the source node's links, not a property.

### Advertise one fully-fixed format

WirePlumber treats `Audio/Source` as a device, so it picks a format from `EnumFormat`,
fixates it, pushes it back — and **suspends the node to do so**. One option means nothing to
negotiate. Expect `param_changed` with a NULL format when idle: WirePlumber suspends idle
nodes after 5 s.

### A daemon-config filter-chain can take down all audio

A `filter-chain` in PipeWire's *daemon* config is a mandatory module: if the plugin fails to
load, PipeWire aborts with **exit 254** and takes all audio with it — hence `nofail` in the
old NixOS config. Cleanroom is a *client*, so this failure class is gone.

---

## DeepFilterNet

### The crates.io `deep_filter` is a different, useless crate

0.2.5 (2022) contains only an HDF5 training dataloader: no `DfTract`, no weights, 58 KB.

```toml
# The library name is `df`, not deep_filter.
# default-features = false drops vorbis/flac and, crucially, the `dataset` feature, which
# pulls an unpublishable git hdf5 dependency. `default-model` is deliberately NOT enabled.
deep_filter = { git = "https://github.com/Rikorose/DeepFilterNet", tag = "v0.5.6",
                default-features = false, features = ["tract"] }
ndarray = "0.15"   # must match — ArrayView2 is in the public call path
```

### An attenuation limit of ≥ 100 dB means *no limit at all*

Not "maximum suppression". Upstream maps `>= 100.0` to `atten_lim: None`, and `< 0.01` to
passthrough. The widely-copied `Attenuation Limit (dB) = 100` in PipeWire filter-chain
configs therefore switches the limiter **off**.

```rust
fn clamp_attenuation(db: f32) -> f32 { db.clamp(0.1, 99.9) }
```

### Feed exactly 480 samples

Upstream guards the hop with a `debug_assert`, so a **release** build silently produces
garbage on a wrong length. PipeWire's quantum is 1024 here and renegotiates whenever another
client joins the graph, so a ring buffer is mandatory — and priming it with one hop of
silence is what makes every subsequent block fill completely:

> The deficit at any moment is `samples_in mod 480`, strictly less than one hop. So one hop
> of slack covers it for any quantum, forever — 10 ms of latency for a permanent fix.

### The weights carry no licence grant

The README licenses "all **code** in this repository"; weights are not code.
[Issue #697](https://github.com/Rikorose/DeepFilterNet/issues/697) asks exactly this and is
unanswered on a repo dormant since Oct 2024. Debian and nixpkgs both redistribute them, so
practical risk is low — but that is inference from silence. Load from a path, never
`include_bytes!`.

---

## GUI (Slint)

### `zbus`'s `tokio` feature breaks accessibility, workspace-wide

Symptom, on the main thread before any window appears:

```
thread '<unnamed>' panicked at zbus/src/abstractions/executor.rs:190
there is no reactor running, must be called from the context of a Tokio 1.x runtime
```

Cargo unifies features across the whole build graph, so `features = ["tokio"]` anywhere
switches zbus's executor **globally**. Slint pulls zbus in twice on its own account —
`i-slint-backend-winit` directly and `accesskit_unix` for the AT-SPI bridge — and
`accesskit_unix` runs its own `async-io` executor on its own thread and calls zbus from
there. The backtrace names `accesskit_unix::context::get_or_init_messages` as the caller.

**Fix: do not enable the feature.** zbus does not need it; on its default executor it
coexists fine with a tokio runtime. Re-adding it silently disables screen-reader support.

### `set_xdg_app_id` goes *after* the window is created

The docs say "before the window is shown", which reads like "before it exists". The Slint
platform is initialised lazily by the **first window**, and `set_xdg_app_id` explicitly
refuses to initialise one:

```
Error: No default Slint platform was selected, and no Slint platform was initialized
```

```rust
let ui = AppWindow::new()?;          // creates the platform
slint::set_xdg_app_id(APP_ID)?;      // then this
ui.show()?;                          // before this
```

### Slint sets no `app_id` at all by default

Confirmed: `hyprctl clients` reports `class: ''`. Wayland has no `WM_CLASS`, so `app_id` is
the only identity a compositor gets, and it looks up `<app_id>.desktop` for the icon. The
`.desktop` **basename**, its `Icon=`, its `StartupWMClass=` and the app_id must all be the
same string.

### The tray is decoration, never the only entry point

GNOME ships no SNI host (the AppIndicator extension is third-party; the official Status
Icons extension is XEmbed-only and will not support SNI). A bare wlroots session often has
none either, since the host is whichever bar the user runs. So:

- closing the window **quits the GUI** rather than hiding into a tray that may not exist;
- everything the tray menu does is reachable from `cleanroom-ctl`.

Verified working on quickshell: item `Active`, 256×256 ARGB pixmap, `/MenuBar` exported with
four live items. Slint hardcodes `Id = "slint-tray"` and leaves `Title` empty; `ToolTip` is
correct and is what hosts show.

`Id` and `Title` are not configurable — they are literals inside `i-slint-backend-winit`,
not defaults we can override from the `SystemTrayIcon` element. Most hosts key off `ToolTip`
for display, so the visible result is right, but a host that matches on `Id` sees
`slint-tray` and cannot tell Cleanroom from any other Slint app.

Living with it is deliberate. The alternative is driving `ksni` directly, which means
re-implementing the menu by hand and giving up the live property binding that keeps the tray
labels agreeing with the window — a real regression in exchange for a cosmetic fix.

### Reading the tray menu to check it

```sh
dbus-send --session --print-reply --dest=org.kde.StatusNotifierItem-<pid>-1 \
  /MenuBar com.canonical.dbusmenu.GetLayout int32:0 int32:-1 array:string:label
```

`busctl` and `gdbus` both parse the `-1` as an option and fail.

### `grim` captures screen regions, not windows

A window on another workspace yields whatever is currently at those coordinates — which is
how an attempt to screenshot the app captured an unrelated desktop instead. Check the window
is on the active workspace before capturing, or do not capture.

---

## Desktop integration

### `graphical-session.target` may never activate

On Hyprland launched from a TTY it is **inactive**, and Hyprland runs no XDG autostart
either ([#5169](https://github.com/hyprwm/Hyprland/issues/5169), closed *not planned*). So
`WantedBy=graphical-session.target` silently never starts, and
`~/.config/autostart/*.desktop` is ignored. Autostart must be a three-way decision made at
toggle time, with `Type=dbus` activation underneath as the always-works path.

Do **not** use `WantedBy=default.target`: with lingering that starts a device-holding daemon
at boot with no session.

### `.desktop` keys that break things

- **No `OnlyShowIn` / `NotShowIn`.** The registry has no entry for Hyprland, sway, niri or
  river, so either key excludes every wlroots compositor.
- **No `X-GNOME-Autostart-Phase`.** Any value but `Application` is fatal on GNOME 49+, and
  it makes systemd's xdg-autostart generator emit no unit at all.
- Absolute `Exec=` — the generator silently emits nothing for a relative path.

### A `.desktop` pointing into `target/` launches nothing, silently

**Symptom.** Clicking Cleanroom in the launcher does nothing at all — no window, no error,
no process. Running the same binary from a terminal inside `nix develop` works perfectly.

**Cause.** `wgpu` dlopens `libvulkan.so.1` and Slint/winit dlopen `libwayland-client.so`.
Because they are dlopened, they are not in `DT_NEEDED`, so `ldd` reports the binary as
**fully resolved** and the `RUNPATH` cargo baked in lists only the real link-time deps
(pipewire, libjpeg-turbo, fontconfig). The libraries are found only via the
`LD_LIBRARY_PATH` the dev shell exports. A desktop entry inherits none of it, so the
process dies immediately:

```
Error: Could not initialize backend.
Error from Winit backend: ... The wayland library could not be loaded
```

and with `Terminal=false` nobody ever sees that line.

**Fix.** Desktop entries must point at a *packaged* binary. `nix profile install .#cleanroom`
wraps all three with the right `LD_LIBRARY_PATH`; deb/rpm/AUR install to `/usr/bin`. From a
checkout, use `mise run gui`, which goes through the dev shell. Never point `Exec=` at
`target/debug` or `target/release`.

### `$XDG_DATA_HOME` outranks the package's own entry

A leftover `~/.local/share/applications/io.github.perfectra1n.Cleanroom.desktop` **shadows**
the packaged one, because `XDG_DATA_HOME` is searched before every entry in
`XDG_DATA_DIRS`. Installing the package correctly is not enough if a stale hand-written
entry is still sitting there — remove it, or it keeps winning.

### Nix: user units go in `share/systemd/user`, not `lib/systemd/user`

systemd's **user** manager searches `$XDG_DATA_DIRS/systemd/user`, to which a nix profile
contributes `~/.nix-profile/share`. It never looks under `lib/`, which is a *system*-manager
path. A unit installed to `$out/lib/systemd/user` is simply invisible — and because the
D-Bus service file delegates with `SystemdService=cleanroomd.service`, that silently breaks
on-demand activation too, which is the only start mechanism that works when
`graphical-session.target` never activates.

Check with `systemctl --user cat cleanroomd.service`: it must resolve to a real path.

### The shipped unit files are FHS, and Nix has no `/usr/bin`

`packaging/systemd/*.service` and `packaging/desktop/*.desktop` hardcode
`/usr/bin/cleanroomd` and `Exec=cleanroom-gui`, which is right for deb, rpm and the AUR
package and wrong for Nix. `flake.nix`'s `postInstall` rewrites them with
`substituteInPlace --replace-fail`. Use `--replace-fail`, never `--replace`: if the source
files are edited later the build then fails loudly instead of shipping a unit that points at
a binary that does not exist.

### `/proc/self/exe` under Nix is `.name-wrapped`, which must not be handed to the user

`makeWrapper` installs the real ELF as `.cleanroomd-wrapped` beside a shell script that sets
`LD_LIBRARY_PATH`, and `/proc/self/exe` names the ELF, never the script. `cleanroom-ctl
autostart` printed that path as its `exec-once =` line, so pasting it would have started the
daemon *bypassing the wrapper* — dying at the first `dlopen` of libvulkan, which is exactly
the silent non-start `autostart.rs` exists to prevent, one layer down. `own_exe()` now maps
`.<name>-wrapped` back to the sibling wrapper when one exists.

Note the printed path is a `/nix/store` path, which changes on every rebuild: an `exec-once`
line pasted into a compositor config will keep running the *old* build after
`nix profile upgrade`. Prefer `~/.nix-profile/bin/cleanroomd` there.

### A worker thread that reports its own health cannot report its own death

`run_once` returned `Ok(())` both for "the stop flag is set" and for "config changed, reopen
the devices", and the caller treated every `Ok` as stop. One `cleanroom-ctl set
video.width` therefore ended the video thread permanently: camera and loopback fds dropped,
the node reverted to output-only, and every app lost the camera.

The reason it went unnoticed for so long is the reporting, not the control flow. Health is
only ever *written* by that thread, so once it was gone the daemon kept serving the last
value it had published — `[idle] no consumers; camera released (virtual camera still
present)` — indefinitely. Plausible, reassuring, and false.

```rust
enum Outcome { Stopped, Restart }   // not Ok(())
```

The general lesson: a component whose liveness is only visible through state *it* publishes
has no way to say "I died". Either the type system distinguishes the exits, or something
outside the thread has to notice it stopped.

### `Restart=on-failure`, never `always`

The single-instance guard exits **0** when it loses the name race, because a second launch is
a normal thing for a user to do. `always` turns that into an endless restart loop.

### RT priority and systemd user units

PAM's `limits.conf` does **not** apply to systemd user units, so `RLIMIT_RTPRIO` of 0 is
normal there and is not itself the fault. rtkit or the Realtime portal is the supported
route. Always set `RLIMIT_RTTIME`, or the kernel SIGKILLs a spinning RT thread. Verify with
`sched_getscheduler()` afterwards rather than assuming.

### Holding `/dev/nvidia*` open blocks system suspend

A 24/7 daemon *is* that process. Subscribe to logind `PrepareForSleep` and release the GPU
on the way down.

---

## Browsers

| | v4l2loopback | PipeWire `Video/Source` |
|---|---|---|
| Chrome / Electron / Zoom / Discord / Teams | yes, needs `exclusive_caps=1` | no — `kWebRtcPipeWireCamera` is `FEATURE_DISABLED_BY_DEFAULT` |
| Firefox upstream | yes | no — `media.webrtc.camera.allow-pipewire = false` |
| Firefox on Fedora 41+ | often invisible ([PipeWire #3659](https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/3659)) | yes |
| Flatpak / portal apps | no | yes |

Neither transport alone reaches everyone, which is why both are planned. The portal path
additionally requires **both** `media.class = Video/Source` *and* `media.role = Camera` —
`xdg-desktop-portal`'s `camera.c` and WirePlumber's `find-portal-access.lua` each check both
independently, and upstream's own `video-src.c` sets only the class and would not be seen.

---

## Prior art worth reading

- [`funinkina/openeffects`](https://github.com/funinkina/openeffects) —
  `daemon/src/pipeline/provider.rs` is a working PipeWire `Video/Source` producer. Same
  workspace shape; independently chose `ort` over `burn`. Do **not** copy its output
  strategy: PipeWire-native only makes it invisible to an out-of-the-box browser.
- [`Marko19907/vcam-rs-master-project`](https://github.com/Marko19907/vcam-rs-master-project)
  — wgpu filter pipeline into a v4l2loopback sink, including the `MmapStream` lifetime
  workaround.
- [`sky0hunter/remote-mic`](https://github.com/sky0hunter/remote-mic) — production
  `Audio/Source` producer in Rust.
