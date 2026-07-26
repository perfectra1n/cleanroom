# Cleanroom

Linux-native webcam and microphone effects — background blur/replace and neural noise
suppression — as a background daemon with a small GUI. Think NVIDIA Broadcast, except it
runs on **AMD and Nvidia alike**, needs no vendor SDK (no CUDA, no ROCm, no TensorRT, no
Maxine), and does not lie to you about whether the GPU is being used.

> **Status: pre-alpha.** Nothing works yet. The repository currently contains only the
> Day-1 spikes that de-risk the architecture.

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
spikes/       Day-1 go/no-go probes. Each has a pre-agreed fallback.
  ort-rvm/          Can we run Robust Video Matting on a vendor-neutral GPU EP?
  slint-hyprland/   Does the GUI toolkit actually work on Wayland/Hyprland?
```

## Building

```sh
nix develop          # rust, pipewire, vulkan, libjpeg-turbo, onnxruntime, libclang
cargo build
```

The dev shell matters more than usual here: `pipewire-sys` and `v4l2-sys-mit` both run
bindgen over system headers and need `LIBCLANG_PATH`, and wgpu dlopens `libvulkan.so.1`
at runtime rather than linking it.

## Licence

GPL-3.0-only. Robust Video Matting's weights are GPL-3.0, which sets the floor.

Model weights are **not** vendored — they are loaded from a path at runtime. For RVM that
is a size decision; for DeepFilterNet it is a licensing one, since its weights carry no
license grant at all ([upstream issue #697](https://github.com/Rikorose/DeepFilterNet/issues/697),
unanswered).
