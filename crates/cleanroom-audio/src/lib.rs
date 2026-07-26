//! PipeWire virtual microphone with neural noise suppression.

pub mod node;
pub mod ringbuf;

pub use node::{AudioError, SharedAudio, VirtualMic, to_dbfs};
pub use ringbuf::{HOP, HopBridge, SampleRing};
