//! PipeWire virtual microphone with neural noise suppression.

pub mod denoise;
pub mod node;
pub mod ringbuf;

pub use denoise::{DenoiseError, Denoiser, find_model};
pub use node::{AudioError, SharedAudio, VirtualMic, to_dbfs};
pub use ringbuf::{HOP, HopBridge, SampleRing};
