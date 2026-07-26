//! Video capture, decode, and the virtual-camera sink.

pub mod capture;
pub mod decode;
pub mod device;
pub mod format;
pub mod sink;

pub use capture::{Camera, CaptureError, RawFrame};
pub use decode::{DecodeError, FrameDecoder, Yuy2Frame};
pub use device::{NodeKind, VideoDevice, capture_devices, enumerate, probe};
pub use format::{CaptureMode, PixelFormat};
pub use sink::{LoopbackSink, SinkError, available_devices, select_device};
