//! PipeWire virtual microphone with neural noise suppression.

pub mod denoise;
pub mod node;
pub mod ringbuf;

pub mod registry;

pub use denoise::{DenoiseError, Denoiser, find_model};
pub use node::{AudioError, SharedAudio, VirtualMic, to_dbfs};
pub use registry::{RegistryView, Source};
pub use ringbuf::HOP;
