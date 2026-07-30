//! Video capture, decode, and the virtual-camera sink.

pub mod capture;
pub mod consumers;
pub mod decode;
pub mod device;
pub mod format;
pub mod holders;
pub mod pw_capture;
pub mod pw_source;
pub mod sink;

pub use capture::{Camera, CaptureError, RawFrame};
pub use consumers::ConsumerWatch;
pub use decode::{DecodeError, FrameDecoder, Yuy2Frame};
pub use device::{NodeKind, VideoDevice, capture_devices, enumerate, probe};
pub use format::{CaptureMode, PixelFormat};
pub use holders::{Holder, holders_of};
pub use pw_capture::{PwCapture, PwCaptureError};
pub use pw_source::{FrameSlot, PwSource, PwSourceError};
pub use sink::{LoopbackSink, SinkError, available_devices, select_device};
