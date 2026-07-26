//! Video capture and the virtual-camera sink.

pub mod device;

pub use device::{NodeKind, VideoDevice, capture_devices, enumerate, probe};
