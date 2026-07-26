{
  config,
  lib,
  pkgs,
  ...
}:
# NixOS module for Cleanroom.
#
# Replaces the older `services.deepfilterNoiseSuppression` filter-chain approach. Worth
# noting why that is an upgrade beyond having a UI: a filter-chain declared in PipeWire's
# *daemon config* is a mandatory module, so if the LADSPA plugin fails to load PipeWire
# aborts with exit 254 and takes ALL audio down with it — which is why that config needs
# `nofail`. Cleanroom is a PipeWire *client*, so that entire failure mode is gone.
let
  cfg = config.services.cleanroom;
in
{
  options.services.cleanroom = {
    enable = lib.mkEnableOption "Cleanroom camera and microphone effects";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The cleanroom package to use.";
    };

    autostart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Start the daemon with the graphical session.

        Note this binds to `graphical-session.target`, which some sessions never reach —
        Hyprland launched directly from a TTY is one, and it also runs no XDG autostart.
        On such a session, use `exec-once = cleanroomd` in the compositor config instead,
        or rely on D-Bus activation: any `cleanroom-ctl` or GUI invocation starts the
        daemon on demand regardless of targets.
      '';
    };

    loopbackDevices = lib.mkOption {
      type = lib.types.ints.positive;
      default = 2;
      description = ''
        How many v4l2loopback devices to provision.

        Two by default so Cleanroom and OBS can each own one. Cleanroom selects a free
        device at runtime rather than claiming a fixed number, so it will not fight another
        producer for a node — but there does have to be a spare one to select.
      '';
    };

    releaseCamera = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Stop PipeWire from holding the physical camera, so Cleanroom can open it directly
          and get MJPEG.

          PipeWire's v4l2 node advertises YUY2 only and does not pass a UVC camera's MJPG
          modes through, and raw 4:2:2 at 1080p saturates USB 2 — so capturing through
          PipeWire is structurally limited to about 5 fps at 1080p. Disable this only if you
          have another consumer that needs the camera through PipeWire.
        '';
      };

      cameraNick = lib.mkOption {
        type = lib.types.str;
        default = "~.*(Webcam|Camera).*";
        example = "C922 Pro Stream Webcam";
        description = ''
          Which camera to hide from PipeWire, matched against `node.nick`.

          Matching on `node.nick` is not a stylistic choice: WirePlumber's v4l2
          create-node hook evaluates rules against the properties that exist at creation
          time, and `media.class` is not among them — so a rule matching `media.class`
          silently never fires. Find the value with `wpctl status`.
        '';
      };
    };

    denoiseModel = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = ''
        Path to `DeepFilterNet3_onnx.tar.gz`.

        Not bundled, and that is a licensing decision rather than a size one: DeepFilterNet
        licenses "all code in this repository", weights are not code, and the upstream issue
        asking for a weights licence has been unanswered since July 2026 on a repo dormant
        since October 2024. Without this the microphone still works, as a *reported*
        passthrough rather than a silent one.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];

    # --- virtual camera -------------------------------------------------------------
    boot.extraModulePackages = [ config.boot.kernelPackages.v4l2loopback ];
    boot.kernelModules = [ "v4l2loopback" ];
    boot.extraModprobeConfig = ''
      # exclusive_caps=1 is load-bearing: Chromium, Electron, Zoom and Teams all reject a
      # loopback node advertising both output and capture caps.
      #
      # No video_nr: Cleanroom picks a free device at runtime, which is what keeps it from
      # colliding with OBS's virtual camera.
      options v4l2loopback devices=${toString cfg.loopbackDevices} exclusive_caps=1 max_buffers=4
    '';

    # --- audio ------------------------------------------------------------------------
    # The daemon is a PipeWire client, so nothing needs to be declared in PipeWire's own
    # config. This only makes sure PipeWire is there to be a client of.
    services.pipewire.enable = lib.mkDefault true;

    # --- camera ownership -------------------------------------------------------------
    services.pipewire.wireplumber.extraConfig = lib.mkMerge [
      {
        # libcamera double-enumerates UVC cameras, exposes only RAW/YUYV, and tends to hold
        # the device fd open continuously.
        "50-cleanroom-disable-libcamera" = {
          "wireplumber.profiles".main."monitor.libcamera" = "disabled";
        };
      }
      (lib.mkIf cfg.releaseCamera.enable {
        "51-cleanroom-camera" = {
          "monitor.v4l2.rules" = [
            {
              matches = [ { "node.nick" = cfg.releaseCamera.cameraNick; } ];
              actions.update-props."node.disabled" = true;
            }
          ];
        };
      })
    ];

    # --- realtime ---------------------------------------------------------------------
    # PAM's limits.conf does NOT apply to systemd user units, so an RLIMIT_RTPRIO of 0 is
    # normal there. rtkit is the supported route, and the audio thread asks through it.
    security.rtkit.enable = lib.mkDefault true;

    # --- session --------------------------------------------------------------------
    systemd.user.services.cleanroomd = {
      description = "Cleanroom camera and microphone effects daemon";
      # After as well as PartOf: Wants/WantedBy imply no ordering, so without this the unit
      # can start before the session environment and the PipeWire socket exist.
      partOf = [ "graphical-session.target" ];
      after = [ "graphical-session.target" ];
      wantedBy = lib.mkIf cfg.autostart [ "graphical-session.target" ];
      serviceConfig = {
        Type = "dbus";
        BusName = "io.github.perfectra1n.Cleanroom";
        ExecStart = "${cfg.package}/bin/cleanroomd";
        # on-failure, not always: the single-instance guard exits 0 when it loses the name
        # race, and `always` would turn that normal outcome into a restart loop.
        Restart = "on-failure";
        RestartSec = 3;
        Slice = "background.slice";
      };
      environment = lib.mkIf (cfg.denoiseModel != null) {
        CLEANROOM_DFN_MODEL = toString cfg.denoiseModel;
      };
    };
  };
}
