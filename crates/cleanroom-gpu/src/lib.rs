//! GPU device management and the WGSL effect kernels.

pub mod device;
pub mod frame;

pub use device::{AdapterChoice, Gpu, GpuError};
pub use frame::{FramePipeline, Look};
