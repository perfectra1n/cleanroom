{
  description = "Cleanroom — Linux-native, vendor-neutral webcam & microphone effects";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Shared library path for the dev shell.
        #
        # vulkan-loader          : wgpu dlopens libvulkan.so.1 at runtime, it is not a link-time dep.
        # wayland/libxkbcommon   : winit/Slint dlopen these; missing them fails at window creation, not link.
        # stdenv.cc.cc.lib       : `ort` with the default `download-binaries` feature fetches a prebuilt
        #                          libonnxruntime.so built against a normal FHS distro. It has no ELF
        #                          interpreter to patch (it is a .so, not an executable), but its
        #                          DT_NEEDED libstdc++/libgcc_s must resolve from somewhere on NixOS.
        runtimeLibs = with pkgs; [
          vulkan-loader
          wayland
          libxkbcommon
          libGL
          fontconfig
          freetype
          stdenv.cc.cc.lib
        ];

        # ONNX Runtime with the WebGPU execution provider, as a fixed-output derivation.
        #
        # This is the only way to get it into a sandboxed build. `ort`'s build script
        # normally downloads this itself, which nix forbids, and nixpkgs' own onnxruntime
        # is not a substitute: it is built without the WebGPU EP, and there is no Dawn in
        # the store at all. Pointing at it would give CPU inference that silently misses
        # the frame budget by 10x.
        #
        # Update both the version and the hash together; a stale hash fails the build
        # loudly, which is the intended behaviour.
        # ONNX Runtime with the WebGPU execution provider, as a fixed-output derivation.
        #
        # This is the exact artefact `ort` would download itself, taken from its own
        # dist table (ort-sys/build/download/dist.txt, row: feature set "wgpu", target
        # "x86_64-unknown-linux-gnu"). Matching it exactly matters:
        #
        #   * nix sandboxes network access, so ort's build script cannot fetch during a
        #     build; ORT_LIB_LOCATION points it at this instead.
        #   * nixpkgs' onnxruntime is NOT a substitute. It is built without the WebGPU EP
        #     and there is no Dawn in the store at all, so pointing at it gives silent CPU
        #     inference — the one failure mode this project refuses to ship.
        #   * neither is Microsoft's own `onnxruntime-linux-x64-gpu` release tarball, which
        #     is the CUDA/TensorRT build and also has no WebGPU EP. That mistake is
        #     particularly easy to make because it downloads and builds perfectly.
        #
        # The hash is ort's own, straight from the dist table, so it is verified upstream
        # rather than by us pinning whatever we happened to receive.
        ortDist = {
          version = "1.24.2";
          url = "https://cdn.pyke.io/0/pyke:ort-rs/ms@1.24.2/x86_64-unknown-linux-gnu+wgpu.tar.lzma2";
          sha256 = "e9aa41101eacde0bf8f832f28c06a8bf3d0f7896a463e0b2d3550563583262b9";
        };

        ortPrebuilt = pkgs.stdenv.mkDerivation {
          pname = "onnxruntime-webgpu-prebuilt";
          version = ortDist.version;

          src = pkgs.fetchurl {
            inherit (ortDist) url sha256;
          };

          nativeBuildInputs = with pkgs; [ xz autoPatchelfHook ];
          buildInputs = [ pkgs.stdenv.cc.cc.lib ];

          # A *raw* LZMA2 stream, not a .lzma or .xz container, so nix's unpackPhase does
          # not recognise it and `xz --format=auto` reports "File format not recognized".
          # It needs the filter chain spelled out, and the dictionary size has to match
          # what it was compressed with — 64 MiB; smaller values fail outright rather than
          # producing partial output.
          unpackPhase = ''
            runHook preUnpack
            xz --format=raw --lzma2=dict=64MiB --decompress --stdout "$src" > dist.tar
            tar xf dist.tar
            runHook postUnpack
          '';

          # Name the two artefacts rather than copying the build directory wholesale.
          # `cp -r . $out/` also captured dist.tar — the 109 MiB intermediate this phase
          # had just extracted, i.e. half the closure was a second copy of its own input —
          # plus the builder's env-vars dump. Listing them explicitly also means a change
          # in the upstream archive layout fails the build instead of silently producing
          # an output with no libraries in it.
          installPhase = ''
            runHook preInstall
            install -Dm755 -t "$out" libwebgpu_dawn.so
            install -Dm644 -t "$out" libonnxruntime.a
            runHook postInstall
          '';
        };

      in
      {
        # --- the package --------------------------------------------------------------
        #
        # packaging/nix/module.nix declares a mandatory `package` option and nothing
        # supplied it, so the module could not be used at all. This is that package.
        packages.cleanroom = pkgs.rustPlatform.buildRustPackage rec {
          pname = "cleanroom";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
            # DeepFilterNet is a git dependency: the crates.io `deep_filter` is a different,
            # useless crate — 0.2.5 from 2022, an HDF5 training dataloader with no DfTract
            # and no weights.
            outputHashes = {
              # Same story as the ort hash: obviously wrong until one real build fills it in.
              "deep_filter-0.5.6" = "sha256-5bYbfO1kmduNm9YV5niaaPvRIDRmPt4QOX7eKpK+sWY=";
            };
          };

          nativeBuildInputs = with pkgs; [
            pkg-config
            llvmPackages.clang
            rustPlatform.bindgenHook
            makeWrapper
          ];

          buildInputs = with pkgs; [
            pipewire
            libjpeg_turbo
            vulkan-loader
            fontconfig
            freetype
            libxkbcommon
            wayland
          ];

          # Only the three shipped binaries. The spikes are workspace members — they are
          # the vendor-neutrality and Slint-interop proofs and belong in the repo — but they
          # are not products, and building them here just adds a second ORT link and a
          # second Slint compile to every package build.
          cargoBuildFlags = [
            "-p" "cleanroomd"
            "-p" "cleanroom-ctl"
            "-p" "cleanroom-gui"
          ];

          # turbojpeg-sys otherwise builds its own bundled libjpeg-turbo with cmake and
          # nasm. The system copy is also faster, being built with SIMD enabled.
          TURBOJPEG_SOURCE = "pkg-config";

          # ort's prebuilt, fetched as a fixed-output derivation.
          #
          # It cannot be fetched during the build: nix sandboxes network access, which is
          # the entire point. And nixpkgs' onnxruntime cannot substitute for it — it is
          # built WITHOUT the WebGPU execution provider and there is no Dawn anywhere in
          # the store, so pointing ORT_LIB_LOCATION at it gives silent CPU inference, which
          # is the one failure mode this project refuses to ship.
          ORT_LIB_LOCATION = ortPrebuilt;
          # Stops ort's build script reaching for the network even with the location set.
          ORT_STRATEGY = "system";

          # The distribution ships libonnxruntime.a alongside libwebgpu_dawn.so, so the
          # static ONNX Runtime resolves its Dawn symbols only if that shared object is on
          # the link line. Without this the build gets all the way to linking and then
          # emits several hundred undefined references to wgpu* symbols.
          RUSTFLAGS = "-L native=${ortPrebuilt} -l dylib=webgpu_dawn";

          # The prebuilt .so files have no $ORIGIN rpath, and wgpu dlopens libvulkan.so.1
          # rather than linking it, so neither is found without help.
          postInstall = ''
            for bin in cleanroomd cleanroom-ctl cleanroom-gui; do
              wrapProgram "$out/bin/$bin" \
                --prefix LD_LIBRARY_PATH : "${ortPrebuilt}:${pkgs.lib.makeLibraryPath runtimeLibs}"
            done

            # share/systemd/user, not lib/systemd/user. systemd's *user* manager searches
            # $XDG_DATA_DIRS/systemd/user, and a nix profile contributes
            # ~/.nix-profile/share — it never looks under lib/, which is a system-manager
            # path. Installed to lib/ the unit is simply invisible, and because the D-Bus
            # service file below delegates activation with SystemdService=, that also
            # silently breaks starting the daemon on demand. The NixOS module declares its
            # own unit and is unaffected either way.
            install -Dm644 packaging/systemd/cleanroomd.service \
              "$out/share/systemd/user/cleanroomd.service"
            install -Dm644 packaging/systemd/io.github.perfectra1n.Cleanroom.service \
              "$out/share/dbus-1/services/io.github.perfectra1n.Cleanroom.service"
            install -Dm644 packaging/desktop/io.github.perfectra1n.Cleanroom.desktop \
              "$out/share/applications/io.github.perfectra1n.Cleanroom.desktop"

            # The three unit/entry files are written for FHS distros, where /usr/bin is
            # both correct and stable — deb, rpm and the AUR package all install there.
            # Nothing is ever at /usr/bin on NixOS, so installing them verbatim gives a
            # systemd unit and a D-Bus service that fail with "No such file or directory"
            # and a launcher entry that silently does nothing. Rewrite the paths to this
            # derivation's own wrappers, which is also what makes LD_LIBRARY_PATH reach
            # the daemon: the bare ELF next to the wrapper would start and then fail to
            # dlopen libvulkan.
            substituteInPlace \
              "$out/share/systemd/user/cleanroomd.service" \
              "$out/share/dbus-1/services/io.github.perfectra1n.Cleanroom.service" \
              --replace-fail /usr/bin/cleanroomd "$out/bin/cleanroomd"

            # Exec=cleanroom-gui relies on PATH, which a .desktop launched by the
            # compositor does not reliably inherit — `nix profile` puts the binary in
            # ~/.nix-profile/bin, which is on an interactive shell's PATH but not
            # necessarily on the session bus activation environment's.
            substituteInPlace \
              "$out/share/applications/io.github.perfectra1n.Cleanroom.desktop" \
              --replace-fail "Exec=cleanroom-gui" "Exec=$out/bin/cleanroom-gui"

            for size in 16 24 32 48 64 128 256; do
              install -Dm644 \
                "packaging/desktop/icons/hicolor/''${size}x''${size}/apps/io.github.perfectra1n.Cleanroom.png" \
                "$out/share/icons/hicolor/''${size}x''${size}/apps/io.github.perfectra1n.Cleanroom.png"
            done

            for conf in packaging/wireplumber/*.conf; do
              install -Dm644 "$conf" "$out/share/wireplumber/wireplumber.conf.d/$(basename "$conf")"
            done
          ''
          ;

          # The model-dependent tests need weights that are deliberately not vendored, and
          # the GPU tests need a real adapter. Neither exists in the sandbox.
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Linux-native webcam and microphone effects";
            homepage = "https://github.com/perfectra1n/cleanroom";
            license = licenses.gpl3Only;
            platforms = platforms.linux;
            mainProgram = "cleanroomd";
          };
        };

        packages.default = self.packages.${system}.cleanroom;

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            # Rust. Pinned via nixpkgs rather than a toolchain file so the shell is
            # self-contained; swap for fenix/rust-overlay if we need a specific nightly.
            rustc
            cargo
            rustfmt
            clippy
            rust-analyzer

            pkg-config
            # bindgen needs libclang. `pipewire-sys` and `v4l2-sys-mit` both run bindgen
            # over system headers, and neither builds without this.
            llvmPackages.clang
            llvmPackages.libclang
          ];

          buildInputs =
            with pkgs;
            [
              # --- audio ---
              pipewire # .dev output carries the headers pipewire-sys binds
              pipewire.dev

              # --- video ---
              libjpeg_turbo # `turbojpeg` crate: decompress_to_yuv on the MJPEG critical path
              v4l-utils # v4l2-ctl for manual probing + the headers v4l2-sys-mit binds

              # --- gpu ---
              vulkan-loader
              vulkan-tools # vulkaninfo, for `doctor` parity and manual checks
              vulkan-validation-layers

              # --- inference ---
              # Present so the `load-dynamic` path is testable, but note: nixpkgs
              # onnxruntime is built WITHOUT the WebGPU EP (override args expose only
              # coreml/cuda/nccl/openvino/rocm). See spikes/ort-rvm.
              onnxruntime

              # --- gui ---
              fontconfig
              freetype
              libxkbcommon
              wayland
              libGL
            ]
            ++ runtimeLibs;

          shellHook = ''
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"

            # `ort`'s `copy-dylibs` feature drops the prebuilt libonnxruntime.so AND
            # libwebgpu_dawn.so next to the built binary, but nothing sets an rpath of
            # $ORIGIN, so they are not found at run time. Put the target dirs on the
            # path rather than fighting rpath.
            #
            # Order matters: nix's own libs come FIRST. Picking a 32-bit vulkan-loader
            # out of the store by accident yields
            #   "Couldn't load Vulkan: libvulkan.so.1: wrong ELF class: ELFCLASS32"
            # from inside Dawn, which reads as "no GPU" rather than as a path bug.
            # makeLibraryPath always resolves the right architecture; globbing
            # /nix/store/*-vulkan-loader-*/lib by hand does not.
            #
            # (NB: everything in this heredoc is a shell comment, not a Nix one --
            # a bare dollar-brace here would still be Nix antiquotation.)
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$PWD/target/debug:$PWD/target/release:/run/opengl-driver/lib:$LD_LIBRARY_PATH"

            # turbojpeg-sys defaults to *building* libjpeg-turbo from its bundled source
            # with cmake, which needs nasm for SIMD and fails without it. We already have
            # libturbojpeg 3.1.4.1 in the shell, so point the crate at it. Building a
            # second copy would also be slower than the system one, not faster: nixpkgs
            # builds it with SIMD enabled.
            export TURBOJPEG_SOURCE=pkg-config

            # The nixpkgs ONNX Runtime, for the `load-dynamic` comparison arm.
            # CPU + OpenVINO only — deliberately NOT the WebGPU path. Verified:
            # `nix eval nixpkgs#onnxruntime.override.__functionArgs` has no webgpuSupport,
            # and there is no Dawn in the store. The WebGPU EP comes from ort's prebuilt.
            export ORT_SYS_NIXPKGS_DYLIB="${pkgs.onnxruntime}/lib/libonnxruntime.so"

            echo "cleanroom devshell"
            echo "  rustc            $(rustc --version | cut -d' ' -f2)"
            echo "  vulkan devices   $(vulkaninfo --summary 2>/dev/null | grep -c deviceName || echo '?')"
            echo "  nixpkgs ORT      $ORT_SYS_NIXPKGS_DYLIB  (CPU/OpenVINO only)"
          '';
        };
      }
    )
    // {
      # System-independent, so it sits outside eachDefaultSystem.
      #
      # The module previously declared a mandatory `package` option with nothing able to
      # supply it, which made it unusable on its own. It now defaults to this flake's own
      # package for the evaluating system.
      nixosModules.cleanroom =
        { pkgs, lib, ... }:
        {
          imports = [ ./packaging/nix/module.nix ];
          services.cleanroom.package = lib.mkDefault self.packages.${pkgs.stdenv.hostPlatform.system}.cleanroom;
        };
      nixosModules.default = self.nixosModules.cleanroom;
    };
}
