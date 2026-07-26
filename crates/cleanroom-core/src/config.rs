//! The configuration schema.
//!
//! Design rule: **sane defaults, everything customisable.** Every field here has a
//! default that produces a working setup on a machine nobody has configured, and every
//! field is reachable over D-Bus at runtime without a restart.

use crate::node::CaptureTarget;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Bumped whenever a field changes meaning in a way a migration must handle. Adding a
/// field with a `#[serde(default)]` does not need a bump; changing units or semantics
/// does. Written into every saved file so an old config is recognisable rather than
/// silently misread.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub video: VideoConfig,
    pub audio: AudioConfig,
    pub gpu: GpuConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            video: VideoConfig::default(),
            audio: AudioConfig::default(),
            gpu: GpuConfig::default(),
        }
    }
}

// --- video ------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VideoConfig {
    pub enabled: bool,

    /// Which camera to open. `None` means "pick the first usable one".
    ///
    /// Enumeration deliberately filters on **Device Caps, not Driver Caps** (Driver Caps
    /// reports a union across nodes and lies), skips `Metadata Capture`-without-
    /// `Video Capture` nodes — that is what `/dev/video1` is for most UVC cameras — and
    /// skips virtual cameras so we can never chain into ourselves or into OBS.
    pub device: Option<String>,

    pub width: u32,
    pub height: u32,
    pub fps: u32,

    pub background: BackgroundMode,

    /// Gaussian-equivalent blur radius, 0.0..=1.0, mapped to a sigma internally.
    pub blur_strength: f32,

    /// Path to a still image for [`BackgroundMode::Replace`]. Cover-fitted to frame.
    pub background_image: Option<PathBuf>,

    pub mirror: bool,

    /// Also publish a PipeWire `Video/Source` node alongside the v4l2loopback device.
    ///
    /// Both transports, because neither reaches everyone. v4l2loopback is what Chrome,
    /// Electron, Zoom, Discord and OBS see. Flatpak and portal-aware apps can only reach a
    /// PipeWire node — and on Fedora, where Firefox ships with PipeWire camera support
    /// patched on, the loopback device may never appear at all.
    pub pipewire_source: bool,

    /// Release the camera when nothing is consuming the virtual camera: LED off, no
    /// decode, no inference. The v4l2loopback producer stays attached throughout, so
    /// meeting apps never see the device disappear — they enumerate cameras once at
    /// startup and would not notice it coming back.
    pub power_save: bool,

    /// What consumers see in their camera picker.
    pub card_label: String,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device: None,
            // 1080p30 requires MJPEG on USB 2 cameras; YUYV at this size is ~5fps.
            width: 1920,
            height: 1080,
            fps: 30,
            background: BackgroundMode::Blur,
            blur_strength: 0.6,
            background_image: None,
            mirror: false,
            pipewire_source: true,
            power_save: true,
            card_label: "Cleanroom Camera".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackgroundMode {
    /// Pass the camera through untouched. Still goes through the GPU pipeline, so
    /// switching modes is instant and the virtual camera never drops a frame.
    Off,
    Blur,
    /// Composite over [`VideoConfig::background_image`].
    Replace,
    /// Solid key colour, for chroma-keying downstream in OBS.
    Remove,
}

// --- audio ------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    pub enabled: bool,

    /// Which hardware microphone to capture. `None` resolves, at bind time, to the
    /// current system default **excluding any node we publish** — see
    /// [`crate::node::CaptureTarget`] for why that exclusion is load-bearing.
    pub device: Option<CaptureTarget>,

    pub denoise: DenoiseConfig,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device: None,
            denoise: DenoiseConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DenoiseConfig {
    pub enabled: bool,

    /// Maximum attenuation applied to detected noise, in dB.
    ///
    /// **Careful:** DeepFilterNet treats any value `>= 100.0` as *no limit at all*
    /// (`atten_lim: None`), not as "100 dB of suppression". Values below 0.01 make it
    /// short-circuit to passthrough. So the useful range is roughly 6..=60, and the
    /// commonly-copied `100` setting means the limiter is simply switched off.
    pub attenuation_db: f32,

    /// DeepFilterNet post-filter beta. Slightly more aggressive on residual noise at
    /// some cost in naturalness. 0.0 disables it.
    pub post_filter_beta: f32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // A real limit, unlike the 100 that silently means "unlimited". Strong
            // enough to kill keyboard and fan noise while leaving speech natural.
            attenuation_db: 40.0,
            post_filter_beta: 0.02,
        }
    }
}

// --- gpu --------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuConfig {
    /// DRM render node to run the pipeline on, e.g. `/dev/dri/renderD128`.
    ///
    /// `None` means "choose automatically", which prefers a discrete GPU. Never
    /// "adapter 0" — a machine with a dGPU and an iGPU will happily hand you the iGPU,
    /// and on this project's reference machine that is an 8x difference (4.5ms on the
    /// RTX 5090 vs 38.8ms on the Radeon iGPU, measured; see docs/spike-results.md).
    pub render_node: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_through_toml() {
        let c = Config::default();
        let s = toml::to_string_pretty(&c).expect("serialise");
        let back: Config = toml::from_str(&s).expect("deserialise");
        assert_eq!(c, back);
    }

    #[test]
    fn partial_config_fills_in_defaults() {
        // A user hand-editing one value must not have to write the whole file.
        let s = r#"
            schema_version = 1
            [video]
            blur_strength = 0.9
        "#;
        let c: Config = toml::from_str(s).expect("partial config must load");
        assert_eq!(c.video.blur_strength, 0.9);
        assert_eq!(c.video.fps, VideoConfig::default().fps);
        assert_eq!(
            c.audio.denoise.attenuation_db,
            DenoiseConfig::default().attenuation_db
        );
    }

    #[test]
    fn unknown_keys_are_rejected_not_ignored() {
        // A typo should be reported, not silently dropped — the user would otherwise
        // change a setting, see no effect, and have no way to find out why.
        let s = r#"
            [video]
            blur_strenght = 0.9
        "#;
        assert!(toml::from_str::<Config>(s).is_err());
    }

    #[test]
    fn default_attenuation_is_an_actual_limit() {
        // Regression guard: >= 100.0 means "no limit" to DeepFilterNet, which is
        // almost certainly not what a default should express.
        let d = DenoiseConfig::default();
        assert!(d.attenuation_db < 100.0, "default must be a real limit");
        assert!(
            d.attenuation_db > 0.01,
            "below 0.01 short-circuits to passthrough"
        );
    }

    #[test]
    fn a_config_naming_our_own_mic_is_rejected() {
        let s = format!("[audio]\ndevice = \"{}\"\n", crate::node::VIRTUAL_MIC_NODE);
        assert!(
            toml::from_str::<Config>(&s).is_err(),
            "a hand-edited config must not be able to create a feedback loop"
        );
    }
}
