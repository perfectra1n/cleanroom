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

### One WGSL entry point per file

Bindings cannot be shared across entry points in a single module, so `@group(0) @binding(0)`
declared twice fails to compile. The colour matrix lives in `shaders/colour.wgsl` and is
concatenated by the host, because naga has no `include`.

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

---

## ONNX Runtime / matting

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
