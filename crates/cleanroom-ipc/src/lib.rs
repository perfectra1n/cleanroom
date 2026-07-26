//! Cleanroom's control surface.
//!
//! The daemon exposes everything over the D-Bus session bus, and the GUI, the CLI and
//! `busctl` are all equal citizens on it. That is a deliberate design commitment: any
//! setting the UI can change must be scriptable, and any state the UI can show must be
//! queryable without one.
//!
//! Choosing D-Bus over a bespoke socket buys three things for free: activation (the
//! daemon starts when something first talks to it), a CLI via `busctl` before we write
//! one, and a natural place for the tray to live.

use serde::{Deserialize, Serialize};
use zbus::zvariant::{OwnedValue, Type, Value};

/// The well-known bus name. Reverse-DNS, and — importantly — matching the `.desktop`
/// basename, because on Wayland `app_id` is the only identity signal a compositor has
/// and it must agree with the desktop entry for the app to get its icon.
pub const BUS_NAME: &str = "io.github.perfectra1n.Cleanroom";

/// The GUI's own name, distinct from the daemon's. Held so a second launch can detect
/// the first and ask it to raise its window rather than opening a duplicate.
pub const GUI_BUS_NAME: &str = "io.github.perfectra1n.Cleanroom.Gui";

pub const OBJECT_PATH: &str = "/io/github/perfectra1n/Cleanroom";
pub const INTERFACE: &str = "io.github.perfectra1n.Cleanroom1";

/// Bumped on a breaking change to the interface. Exposed as a property so a mismatched
/// GUI can say "update me" rather than failing in some obscure way at the first call.
///
/// 2: `PipelineStats` gained `matte_rejected` and `Status` gained `pw_node`, both of
///    which change D-Bus signatures.
pub const INTERFACE_VERSION: u32 = 3;

/// How healthy the pipeline is.
///
/// This type is the enforcement mechanism for the project's central rule: **no silent
/// degradation.** The prior art had a three-stage CPU demotion ladder that made it
/// impossible to tell whether the GPU was doing anything. Here, anything short of
/// nominal is a value the daemon must publish, the CLI prints, and the GUI shows as a
/// banner. Degrading quietly is not representable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Type, Value, OwnedValue)]
#[zvariant(signature = "s")]
pub enum Health {
    /// Everything is running as configured.
    Nominal,
    /// Running, but not as configured, and the user needs to know why.
    Degraded,
    /// Not running. `detail` says what stopped it.
    Failed,
    /// Deliberately idle — e.g. camera released because nothing is consuming the
    /// virtual camera. Not a fault.
    Idle,
}

impl std::fmt::Display for Health {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Health::Nominal => "nominal",
            Health::Degraded => "degraded",
            Health::Failed => "failed",
            Health::Idle => "idle",
        })
    }
}

/// A capture device offered to the user.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct DeviceInfo {
    /// Stable identifier to store in config. For video this is a `/dev/video*` path;
    /// for audio a PipeWire `node.name`.
    pub id: String,
    /// Human-readable name for a picker.
    pub description: String,
    /// Whether this device is currently usable. A camera held by another process is
    /// listed but not available, which is more useful than hiding it.
    pub available: bool,
}

/// Per-second pipeline telemetry, emitted on the `Stats` signal.
///
/// The GUI's perf HUD reads this. It exists so "is the GPU actually being used" has an
/// answer you can look at, rather than being inferred from how the picture looks.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Type)]
pub struct PipelineStats {
    /// Frames delivered to the virtual camera in the last second.
    pub fps: f64,
    /// Mean CPU decode time per frame, milliseconds. MJPEG decode is the one CPU step
    /// in the video path, so it gets its own bucket — folding it into gpu_ms would hide
    /// which half of the pipeline is actually slow.
    pub decode_ms: f64,
    /// Mean GPU time per frame, milliseconds. Covers colour conversion, matting and
    /// compositing — everything between upload and readback.
    pub gpu_ms: f64,
    /// Mean matting inference time, milliseconds. A subset of `gpu_ms`.
    ///
    /// Covers the matte readback as well as inference, so it is deliberately *not*
    /// comparable to a bare model benchmark — it is the cost the frame loop actually pays.
    pub matting_ms: f64,
    /// Mattes rejected by the degenerate-alpha guard since startup.
    ///
    /// Cumulative, not per-second, because what matters is the trend: a handful over a
    /// session is the guard doing its job, a number that climbs with the frame counter
    /// means the guard's threshold is wrong for this footage and every matte is being
    /// thrown away. Without this on the wire, that failure looks like "matting is on but
    /// does nothing", which is indistinguishable from having no model at all.
    pub matte_rejected: u64,
    /// Frames the *driver* produced that we never collected, counted from gaps in the V4L2
    /// sequence number. This is the camera outrunning the pipeline, and it is the number
    /// that goes up first when a frame's total work exceeds the frame interval.
    pub dropped: u64,
    /// How many processes are currently reading the virtual camera. Drives power save.
    /// Note this counts *streaming* consumers, not opens: browsers probe-open cameras
    /// without streaming and must not be counted.
    pub vcam_consumers: u32,
    /// Microphone level pre-denoise, dBFS.
    pub mic_level_db: f32,
    /// Microphone level post-denoise, dBFS. The gap is the suppression you are getting.
    pub mic_level_out_db: f32,
}

/// A snapshot of what the daemon is doing, for `cleanroom-ctl status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Type)]
pub struct Status {
    pub video_health: Health,
    pub video_detail: String,
    pub audio_health: Health,
    pub audio_detail: String,
    /// The adapter actually in use, e.g. "NVIDIA GeForce RTX 5090 (/dev/dri/renderD128)".
    /// Printed rather than assumed, because a dual-GPU machine will silently hand you
    /// the wrong one.
    pub gpu_adapter: String,
    /// Path of the v4l2loopback node we are producing into. Allocated at runtime via the
    /// control device rather than hardcoded, so we never fight OBS over /dev/video10.
    pub vcam_path: String,
    /// The PipeWire node name we publish as, or empty when that transport is off or has
    /// failed. Reported separately from `vcam_path` because the two transports succeed and
    /// fail independently, and "the camera works" is a different claim for each.
    pub pw_node: String,
    /// Which execution provider is running the matting network, and why if it is not the
    /// one that was asked for — e.g. "cpu (the GPU provider found 0.0% of the frame to be
    /// subject where the CPU found 8.8%)".
    ///
    /// Reported for the same reason `gpu_adapter` is: the fast path and the correct path
    /// turned out to be different providers on the reference machine, and a matte that
    /// quietly decays to nothing is indistinguishable from "the effect is off" unless
    /// something says which engine produced it.
    pub matting_engine: String,
    pub stats: PipelineStats,
}

#[zbus::proxy(
    interface = "io.github.perfectra1n.Cleanroom1",
    default_service = "io.github.perfectra1n.Cleanroom",
    default_path = "/io/github/perfectra1n/Cleanroom"
)]
pub trait Cleanroom {
    /// A full snapshot. Cheap; safe to poll.
    fn status(&self) -> zbus::Result<Status>;

    /// Cameras we could open. Excludes virtual cameras and metadata-only nodes.
    fn list_cameras(&self) -> zbus::Result<Vec<DeviceInfo>>;

    /// Hardware microphones. Never includes our own virtual node — offering it would
    /// invite the user to create a feedback loop.
    fn list_microphones(&self) -> zbus::Result<Vec<DeviceInfo>>;

    /// Which autostart mechanism applies, whether it is on, and any manual step.
    ///
    /// Evaluated at call time rather than stored, because the answer depends on the
    /// session that happens to be running — the same machine can boot into GNOME one day
    /// and bare Hyprland the next, and only one of those honours XDG autostart.
    fn autostart(&self) -> zbus::Result<(String, String, bool)>;

    /// Turn autostart on or off. Returns the mechanism used and any manual step left.
    fn set_autostart(&self, on: bool) -> zbus::Result<(String, String)>;

    /// Read a setting by dotted path, e.g. `video.blur_strength`.
    ///
    /// Stringly-typed on purpose: it keeps the CLI and any shell script honest without
    /// a generated binding per field, and the daemon validates against the real schema.
    fn get(&self, key: &str) -> zbus::Result<String>;

    /// Write a setting by dotted path. Applied to the running pipeline immediately and
    /// persisted atomically. Changing a value must never interrupt audio or video.
    fn set(&self, key: &str, value: &str) -> zbus::Result<()>;

    /// Every settable key with its current value, for shell completion and `ctl get`.
    fn keys(&self) -> zbus::Result<Vec<(String, String)>>;

    /// Re-read the config from disk, discarding unsaved runtime changes.
    fn reload(&self) -> zbus::Result<()>;

    /// Environment checks: Secure Boot, module presence, GPU adapter, RT priority,
    /// WirePlumber holding the camera, browser prefs. Returns human-readable lines.
    fn doctor(&self) -> zbus::Result<Vec<String>>;

    /// Ask the daemon to shut down cleanly. Needed because the GPU context has to be
    /// torn down in a controlled order — dropping an ONNX Runtime session that owns a
    /// Dawn context segfaults, which under `Restart=on-failure` becomes a restart loop.
    fn shutdown(&self) -> zbus::Result<()>;

    #[zbus(property)]
    fn interface_version(&self) -> zbus::Result<u32>;

    /// Emitted when health changes. The GUI shows a banner on anything but Nominal.
    #[zbus(signal)]
    fn state_changed(&self, status: Status) -> zbus::Result<()>;

    /// Roughly once a second while running.
    #[zbus(signal)]
    fn stats(&self, stats: PipelineStats) -> zbus::Result<()>;
}
