//! Node naming, and the type that makes the virtual-microphone feedback loop
//! unrepresentable.
//!
//! The trap this exists to close, in the prior art's own words: its `mic_device`
//! defaulted to `""`, which made the capture stream open the *system default* source.
//! The moment a user selected the virtual microphone as their default input — which is
//! exactly what you do after picking it once in a meeting app — the pipeline captured
//! its own output and howled.
//!
//! The fix is not a runtime check that someone can forget to call. It is that
//! [`CaptureTarget`] cannot be constructed from a name we own.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The PipeWire `node.name` of the virtual microphone Cleanroom publishes.
///
/// Deliberately a plain `Audio/Source`, not `Audio/Source/Virtual`: the `/Virtual`
/// variant keeps `portconfig_direction = INPUT` (it exposes input ports you feed, the
/// null-sink topology) and, more importantly, QtWebEngine and Electron clients do not
/// list it as a microphone at all.
pub const VIRTUAL_MIC_NODE: &str = "cleanroom_mic";

/// The PipeWire `node.name` of the virtual camera node.
///
/// Note this is the *PipeWire* output. The v4l2loopback device is separate and its
/// index is allocated at runtime rather than hardcoded — see `cleanroom-video`.
pub const VIRTUAL_CAM_NODE: &str = "cleanroom_cam";

/// Every node name Cleanroom publishes. Anything in here is refused as a capture target.
const OWNED_NODES: &[&str] = &[VIRTUAL_MIC_NODE, VIRTUAL_CAM_NODE];

/// A validated capture target: the `node.name` of a real hardware input.
///
/// Constructing one is the only way to tell the audio pipeline what to record from, and
/// it is impossible to construct one naming a node we publish. The feedback loop is a
/// compile-time-shaped problem rather than a runtime one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CaptureTarget(String);

/// Why a proposed capture target was rejected.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CaptureTargetError {
    /// The name refers to a node Cleanroom itself publishes. Binding it would make the
    /// pipeline capture its own output.
    #[error(
        "'{0}' is a node Cleanroom publishes; capturing it would feed the output back \
         into the input. Pick a hardware microphone."
    )]
    SelfReference(String),

    /// An empty name would resolve to the system default source, which may *become* our
    /// own node at any time without warning — the exact trap the prior art fell into.
    #[error(
        "capture target must name a concrete hardware node. An empty target resolves to \
         the system default, which becomes self-capture the moment the user sets \
         Cleanroom as their default input."
    )]
    Empty,
}

impl CaptureTarget {
    /// Validate a PipeWire `node.name` as a capture source.
    pub fn new(name: impl Into<String>) -> Result<Self, CaptureTargetError> {
        let name = name.into();
        let trimmed = name.trim();

        if trimmed.is_empty() {
            return Err(CaptureTargetError::Empty);
        }
        if is_owned_node(trimmed) {
            return Err(CaptureTargetError::SelfReference(trimmed.to_string()));
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// True if `name` is a node Cleanroom publishes.
///
/// Matches the bare name and any PipeWire-style suffixed form (`cleanroom_mic.2`), which
/// is what you get when a node is republished while the old one is still being torn down.
pub fn is_owned_node(name: &str) -> bool {
    OWNED_NODES
        .iter()
        .any(|owned| name == *owned || name.starts_with(&format!("{owned}.")))
}

impl fmt::Display for CaptureTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for CaptureTarget {
    type Error = CaptureTargetError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl From<CaptureTarget> for String {
    fn from(t: CaptureTarget) -> String {
        t.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_real_hardware_node() {
        let t = CaptureTarget::new(
            "alsa_input.usb-Focusrite_Scarlett_Solo_USB_Y7H767R15C5B05-00.HiFi__Mic1__source",
        )
        .expect("a hardware node must be accepted");
        assert!(t.as_str().contains("Scarlett"));
    }

    #[test]
    fn refuses_our_own_microphone() {
        // This is the whole point of the type.
        assert_eq!(
            CaptureTarget::new(VIRTUAL_MIC_NODE),
            Err(CaptureTargetError::SelfReference(VIRTUAL_MIC_NODE.into()))
        );
    }

    #[test]
    fn refuses_suffixed_republished_node() {
        // PipeWire appends .N when a name collides during a republish race.
        assert!(matches!(
            CaptureTarget::new("cleanroom_mic.2"),
            Err(CaptureTargetError::SelfReference(_))
        ));
    }

    #[test]
    fn refuses_empty_because_it_means_system_default() {
        assert_eq!(CaptureTarget::new(""), Err(CaptureTargetError::Empty));
        assert_eq!(CaptureTarget::new("   "), Err(CaptureTargetError::Empty));
    }

    #[test]
    fn deserialising_a_self_reference_fails_rather_than_silently_working() {
        // A hand-edited config must not be able to smuggle in a feedback loop.
        let bad = format!("target = \"{VIRTUAL_MIC_NODE}\"\n");
        #[derive(serde::Deserialize)]
        struct Holder {
            #[allow(dead_code)]
            target: CaptureTarget,
        }
        assert!(toml::from_str::<Holder>(&bad).is_err());
    }

    #[test]
    fn round_trips_through_toml() {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Holder {
            target: CaptureTarget,
        }
        let h = Holder {
            target: CaptureTarget::new("alsa_input.pci-0000_00_1f.3.analog-stereo").unwrap(),
        };
        let s = toml::to_string(&h).unwrap();
        let back: Holder = toml::from_str(&s).unwrap();
        assert_eq!(back.target, h.target);
    }
}
