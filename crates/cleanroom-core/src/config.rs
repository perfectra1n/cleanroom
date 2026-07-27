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

    /// Pull the background toward grey, 0.0..=1.0. Applied to the background plane only,
    /// so the subject keeps its colour. Blur alone leaves a busy room still legible as
    /// colour and motion; a little desaturation is what makes it read as "not the point".
    pub background_desaturate: f32,

    /// Darken the background, 0.0..=1.0. Same idea as `background_desaturate`, and the two
    /// together approximate the "spotlight" look without a second render pass.
    pub background_dim: f32,

    /// Pull the alpha edge inward, 0.0..=0.9. `None` picks per mode: nothing for blur,
    /// where a generous silhouette against a blurred copy of the same room is invisible,
    /// and a little for replace, where the same generosity is a bright halo tracing the
    /// shoulders and ears. Set a number to force one value for every mode.
    ///
    /// Note this also *sharpens*: the remap has gain `1/(1 - tighten)`, so 0.34 is a 51%
    /// steeper ramp as well as a tighter cut. To soften, reach for [`matte_feather`].
    pub matte_tighten: Option<f32>,

    /// Widen the alpha ramp, 0.0..=1.0, without moving where it crosses 0.5.
    ///
    /// The knob for "the cut-out looks like a sticker". [`matte_tighten`] decides *where*
    /// the silhouette ends; this decides how abruptly it gets there. 0.0 is the historical
    /// behaviour exactly, so an existing config composites unchanged.
    #[serde(default)]
    pub matte_feather: f32,

    /// How readily the matte follows alpha *increasing* at a pixel, 0.01..=1.0.
    ///
    /// Higher follows the network more closely and shimmers more; lower is calmer and
    /// slower. Rising is allowed to move faster than falling by default, because gaining a
    /// little subject early is invisible where losing it early punches a hole in a limb.
    #[serde(default = "default_fade_rise")]
    pub matte_fade_rise: f32,

    /// How readily the matte follows alpha *decreasing* at a pixel, 0.01..=1.0.
    ///
    /// The trailing edge of anything that moves. See [`matte_fade_rise`].
    #[serde(default = "default_fade_fall")]
    pub matte_fade_fall: f32,

    /// The per-pixel alpha change at which the fade damping is fully released, 0.01..=1.0.
    ///
    /// Below it a change is treated as noise and averaged; at or above it the new value is
    /// taken essentially whole, because a jump that large is the network reporting real
    /// motion and averaging that is what produces ghost trails. Lower reacts sooner.
    #[serde(default = "default_motion_release")]
    pub matte_motion_release: f32,

    /// Edge-aware upsampling of the matte, instead of a plain bilinear stretch.
    ///
    /// The matte is computed small and shown at frame size, and bilinear does not know
    /// where the subject ends — it smears alpha across the boundary, which is the soft halo
    /// around a shoulder and the background bleeding into hair. Off is faster and worse.
    pub guided_filter: bool,

    /// Guided-filter window radius in matte pixels. Larger is smoother and costs
    /// `(2r+1)^2` taps per pixel.
    pub guided_radius: u32,

    /// Guided-filter regularisation. Larger means more smoothing and less edge-following;
    /// this is the knob that trades a crisp-but-noisy edge against a clean-but-soft one.
    pub guided_eps: f32,

    /// Which execution provider runs the matting network. See [`MattingBackend`] — this is
    /// a correctness setting, not just a speed one.
    pub matting_backend: MattingBackend,

    /// Matting input width. `None` derives it from the provider that ends up running: the
    /// GPU can afford 512 px, the CPU cannot and gets 320. Raising it sharpens the matte
    /// and costs inference time on a budget already shared with decode and compositing.
    pub matting_width: Option<u32>,

    /// Matting input height. `None` derives 16:9 from the width.
    pub matting_height: Option<u32>,

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

// Serde needs these as functions so an existing config.toml that predates the fields still
// deserialises. They mirror `cleanroom_matting::Smoothing::default`, which is the authority
// — this crate cannot depend on that one, so the test below pins the two together.
fn default_fade_rise() -> f32 {
    0.55
}
fn default_fade_fall() -> f32 {
    0.22
}
fn default_motion_release() -> f32 {
    0.25
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
            background_desaturate: 0.0,
            background_dim: 0.0,
            matte_tighten: None,
            matte_feather: 0.0,
            matte_fade_rise: default_fade_rise(),
            matte_fade_fall: default_fade_fall(),
            matte_motion_release: default_motion_release(),
            guided_filter: true,
            guided_radius: 3,
            guided_eps: 1e-4,
            matting_backend: MattingBackend::Auto,
            matting_width: None,
            matting_height: None,
            background_image: None,
            mirror: false,
            pipewire_source: true,
            power_save: true,
            card_label: "Cleanroom Camera".into(),
        }
    }
}

/// Which ONNX Runtime execution provider runs the matting network.
///
/// A correctness setting before it is a speed setting. ONNX Runtime 1.24.2's WebGPU
/// provider runs this model ~4x faster than the CPU provider and returns an alpha matte
/// that is **zero everywhere**, which composites as "the whole frame is background" — the
/// subject gets blurred along with the room. Nothing errors; it is simply wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MattingBackend {
    /// Prefer the GPU, then prove it against the CPU on the first frame with a subject in
    /// it and switch if it disagrees. The only value that cannot silently blur your face.
    Auto,
    /// Force the GPU provider. Fast, and unverified — check the output yourself.
    Gpu,
    /// Force the CPU provider. Correct on every machine measured so far.
    Cpu,
}

impl VideoConfig {
    /// The matting input size actually used, resolving `None` against the provider.
    ///
    /// The two providers have very different budgets — measured on an RTX 5090 with a
    /// 33 ms frame, this model costs 4-18 ms on the GPU and 10.6 ms at 256x144 rising to
    /// 39 ms at 512x288 on the CPU — so a single default cannot serve both. A GPU that
    /// turns out to be wrong and falls back to the CPU would otherwise silently halve the
    /// frame rate instead of the matte resolution.
    ///
    /// An explicit width with no height keeps 16:9, rounded to even so chroma subsampling
    /// downstream never sees an odd dimension.
    pub fn matting_size(&self, on_gpu: bool) -> (u32, u32) {
        let w = self
            .matting_width
            .unwrap_or(if on_gpu { 512 } else { 320 })
            .clamp(64, 1920);
        let h = self
            .matting_height
            .unwrap_or(((w * 9 / 16) + 1) & !1)
            .clamp(64, 1080);
        (w, h)
    }

    /// How far to pull the alpha edge in, resolved per mode. See [`VideoConfig::matte_tighten`].
    pub fn tighten_for(&self, mode: BackgroundMode) -> f32 {
        self.matte_tighten
            .unwrap_or(match mode {
                BackgroundMode::Replace => 0.12,
                _ => 0.0,
            })
            .clamp(0.0, 0.9)
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
