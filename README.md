# Cleanroom

Linux-native webcam and microphone effects — background blur/replace and neural noise
suppression — as a background daemon with a small GUI. Think NVIDIA Broadcast, except it
runs on **AMD and Nvidia alike**, needs no vendor SDK (no CUDA, no ROCm, no TensorRT, no
Maxine), and does not lie to you about whether the GPU is being used.

> **Status: alpha.** Everything below works on the reference machine (NixOS, RTX 5090 +
> Radeon iGPU, Hyprland, Logitech C922, Focusrite Scarlett Solo). It has been run on
> exactly one machine, so treat anything about *your* hardware as untested.

| Subsystem | State |
|---|---|
| Virtual camera | `/dev/video0` → `/dev/video10`, MJPG 1080p30, power-save on real consumer events |
| PipeWire camera | `cleanroom_cam` published as `Video/Source` + `media.role=Camera` for Flatpak and portal apps |
| Background | Blur, replace (cover-fitted image), green key; guided-filter matte upsample |
| Virtual microphone | `cleanroom_mic` with DeepFilterNet, ~40 dB measured suppression, released when nothing is listening |
| GPU | RVM matting at 9.6 ms/frame on an RTX 5090, 30 fps sustained end to end |
| Desktop | GUI with tray, preview, device pickers; autostart; suspend/resume; `cleanroom-ctl` parity |

Model weights are not bundled — run `cleanroom-ctl fetch-models` once. See
[Licence](#licence).

## Why

The existing Linux options are either Nvidia-locked, or a pile of shell scripts around
`v4l2loopback`, or a static PipeWire `filter-chain` you have to edit a config file and
restart your session to change. Cleanroom aims to be the thing you configure once, leave
running, and forget about.

## Design commitments

These are the non-negotiables the rest of the design falls out of.

**No silent degradation.** Every fallback is an explicit daemon state, reported over
D-Bus and shown in the UI. If the GPU path fails you are told, loudly. The prior art this
project replaces has a three-stage demotion ladder that quietly lands on the CPU, which
makes it impossible to tell whether it is working — that is the single behaviour we are
most determined not to reproduce.

**GPU required, vendor-neutral.** Inference runs on a portable GPU path. There is no CPU
inference fallback, because a CPU fallback nobody notices is worse than an error.

**The daemon owns the devices.** Exactly one process opens the camera. Effects survive
closing the window. The GUI, the CLI and `busctl` are equal citizens on the same D-Bus
interface — anything the UI can do is scriptable.

**Both virtual-camera transports.** `v4l2loopback` reaches Chrome, Electron, Zoom,
Discord and OBS. A PipeWire `Video/Source` node reaches Flatpak and portal-aware apps, and
Firefox where distros have flipped on PipeWire camera support. Neither alone reaches
everybody, so we publish both.

## Layout

```
crates/
  cleanroom-core/     Config schema and non-destructive persistence
  cleanroom-video/    V4L2 capture, decode, v4l2loopback sink, PipeWire Video/Source
  cleanroom-audio/    PipeWire virtual mic, DeepFilterNet, registry watcher
  cleanroom-gpu/      wgpu pipeline: colour conversion, blur, guided filter, composite
  cleanroom-matting/  Robust Video Matting on the WebGPU execution provider
  cleanroom-ipc/      The D-Bus surface, shared by every client
  cleanroomd/         The daemon: owns the camera, the mic and the GPU
  cleanroom-ctl/      CLI. Everything the GUI can do
  cleanroom-gui/      Slint control panel and tray

spikes/       Day-1 go/no-go probes. Kept because a proof that no longer compiles
              has stopped being one.
  ort-rvm/          Can we run Robust Video Matting on a vendor-neutral GPU EP?
  slint-hyprland/   Does the GUI toolkit actually work on Wayland/Hyprland?

docs/
  pitfalls.md       Every trap that cost real time, with the code that resolves it
  spike-results.md  The measured numbers behind the design decisions
  shortcuts.md      Binding compositor keys to cleanroom-ctl
```

## Using it

```sh
cleanroom-ctl fetch-models        # once; not bundled, see Licence
cleanroom-ctl doctor              # checks the things that usually go wrong
cleanroomd                        # or let D-Bus activation start it
cleanroom-ctl set video.background blur
cleanroom-gui                     # optional; the daemon does not need it
```

## Building

```sh
nix develop          # rust, pipewire, vulkan, libjpeg-turbo, onnxruntime, libclang
cargo build
```

The dev shell matters more than usual here: `pipewire-sys` and `v4l2-sys-mit` both run
bindgen over system headers and need `LIBCLANG_PATH`, and wgpu dlopens `libvulkan.so.1`
at runtime rather than linking it.

## Documentation

* [docs/pitfalls.md](docs/pitfalls.md) — every trap that cost real time, symptom first,
  with the code that resolves it. Read the zbus, WirePlumber and DeepFilterNet-attenuation
  entries before touching those areas: all three fail *silently* when got wrong.
* [docs/spike-results.md](docs/spike-results.md) — the measured numbers behind the design
  decisions, and the two plan claims the spikes disproved.
* [docs/shortcuts.md](docs/shortcuts.md) — binding compositor keys, and why global
  shortcuts stay out of process.

## Licence

GPL-3.0-only. Robust Video Matting's weights are GPL-3.0, which sets the floor.

Model weights are **not** vendored — they are loaded from a path at runtime. For RVM that
is a size decision; for DeepFilterNet it is a licensing one, since its weights carry no
license grant at all ([upstream issue #697](https://github.com/Rikorose/DeepFilterNet/issues/697),
unanswered).
