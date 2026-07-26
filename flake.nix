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
      in
      {
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
    );
}
