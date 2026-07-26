//! Cleanroom's configuration schema and shared value types.
//!
//! Pure data. This crate deliberately has no camera, GPU, PipeWire or D-Bus dependency
//! so that the daemon, the CLI and the GUI can all agree on the shape of a setting
//! without dragging in each other's machinery.
//!
//! Two things here are load-bearing beyond being plain structs:
//!
//! * [`node::CaptureTarget`] makes the virtual-microphone feedback loop unrepresentable
//!   rather than merely checked-for.
//! * [`persist`] saves atomically and, critically, refuses to paper over a corrupt
//!   config with defaults.

pub mod config;
pub mod node;
pub mod persist;

pub use config::{
    AudioConfig, BackgroundMode, Config, DenoiseConfig, GpuConfig, MattingBackend, SCHEMA_VERSION,
    VideoConfig,
};
pub use node::{CaptureTarget, CaptureTargetError, VIRTUAL_CAM_NODE, VIRTUAL_MIC_NODE};
pub use persist::{ConfigError, ConfigPaths, LoadOutcome};
