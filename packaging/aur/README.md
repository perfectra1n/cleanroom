# Arch packaging

**Unbuilt and untested.** This PKGBUILD was written on NixOS, where `makepkg` does not
run, so it has never been executed. It needs one real build before it goes anywhere near
the AUR.

Two things to check first:

* **`LIBCLANG_PATH`** is guessed as `/usr/lib`. Arch puts libclang there today, but the
  build fails confusingly if it is wrong — `bindgen` reports missing system headers rather
  than a missing libclang.
* **The `ort` shared objects.** `ort` downloads a prebuilt ONNX Runtime because neither
  Arch's nor nixpkgs' onnxruntime is built with the WebGPU execution provider, and drops
  `libonnxruntime.so` and `libwebgpu_dawn.so` next to the binary with no `$ORIGIN` rpath.
  They are installed to `/usr/lib/cleanroom/`, which means the binaries need either an
  rpath set at build time or a wrapper — **this is the part most likely to be wrong.**

Model weights are not packaged. Run `cleanroom-ctl fetch-models` after installing;
DeepFilterNet's weights carry no licence grant, so redistributing them in a package is a
decision the user should make rather than the packager.
