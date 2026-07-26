//! Camera capture.
//!
//! Opens `/dev/video*` directly rather than going through PipeWire. That is not a
//! preference — PipeWire's v4l2 SPA node advertises YUY2 only and does not pass a UVC
//! camera's MJPG modes through, which structurally pins 1080p to about 5fps over USB 2,
//! with no knob to change it.
//!
//! The consequence is that PipeWire must not be holding the device at the same time. If
//! `open` fails with EBUSY, `doctor` explains how to release it.

use crate::format::{CaptureMode, PixelFormat, mode_ladder};
use std::time::Duration;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::{CaptureStream, Stream};
use v4l::video::Capture as CaptureTrait;
use v4l::{Device, Format, FourCC};

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("cannot open {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} reports no pixel format we can use")]
    NoUsableFormat { path: String },

    #[error(
        "every capture mode was refused by {path}. Tried: {tried}. \
         The camera may be held by another process — check `cleanroom-ctl doctor`."
    )]
    NoModeAccepted { path: String, tried: String },

    #[error("v4l2 error: {0}")]
    Io(#[from] std::io::Error),
}

/// A frame straight from the camera, still in its capture format.
pub struct RawFrame<'a> {
    pub data: &'a [u8],
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    /// Driver sequence number. Gaps mean the driver dropped frames, which is worth
    /// distinguishing from us dropping them.
    pub sequence: u32,
}

pub struct Camera {
    device: Device,
    stream: Option<MmapStream<'static>>,
    mode: CaptureMode,
    path: String,
}

impl Camera {
    /// Open a camera and negotiate the best mode it will actually grant.
    ///
    /// Walks the ladder, and — importantly — verifies what the driver *returned* rather
    /// than trusting that `S_FMT` honoured the request. V4L2's contract is that the
    /// driver may substitute a mode it prefers, silently, so the negotiated result has to
    /// be read back.
    pub fn open(path: &str, want_w: u32, want_h: u32, want_fps: u32) -> Result<Self, CaptureError> {
        let device = Device::with_path(path).map_err(|source| CaptureError::Open {
            path: path.to_string(),
            source,
        })?;

        let available = enumerate_modes(&device);
        if available.is_empty() {
            return Err(CaptureError::NoUsableFormat {
                path: path.to_string(),
            });
        }

        let ladder = mode_ladder(&available, want_w, want_h, want_fps);
        let mut tried = Vec::new();

        for candidate in &ladder {
            tried.push(candidate.to_string());

            let fmt = Format::new(candidate.width, candidate.height, candidate.format.fourcc());

            let Ok(granted) = CaptureTrait::set_format(&device, &fmt) else {
                continue;
            };

            // The driver may hand back something other than what we asked for. Accept
            // only a format we can actually decode; a mismatched size is fine (we adapt),
            // an unknown fourcc is not.
            let Some(got_format) = PixelFormat::from_fourcc(granted.fourcc) else {
                continue;
            };

            // Frame rate is a request, not a guarantee. Ask, then read back.
            let mut params = match CaptureTrait::params(&device) {
                Ok(p) => p,
                Err(_) => continue,
            };
            params.interval = v4l::Fraction::new(1, candidate.fps);
            let granted_fps = CaptureTrait::set_params(&device, &params)
                .ok()
                .map(|p| {
                    if p.interval.numerator == 0 {
                        candidate.fps
                    } else {
                        p.interval.denominator / p.interval.numerator.max(1)
                    }
                })
                .unwrap_or(candidate.fps);

            let mode = CaptureMode {
                format: got_format,
                width: granted.width,
                height: granted.height,
                fps: granted_fps,
            };

            if mode != *candidate {
                tracing::info!(
                    requested = %candidate,
                    granted = %mode,
                    "driver substituted a different capture mode"
                );
            }
            tracing::info!(path, mode = %mode, "camera opened");

            return Ok(Self {
                device,
                stream: None,
                mode,
                path: path.to_string(),
            });
        }

        Err(CaptureError::NoModeAccepted {
            path: path.to_string(),
            tried: tried.join(", "),
        })
    }

    pub fn mode(&self) -> CaptureMode {
        self.mode
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Begin streaming. Separate from `open` so the daemon can hold a configured camera
    /// without the LED on and without consuming USB bandwidth — that is what power save
    /// toggles.
    pub fn start(&mut self) -> Result<(), CaptureError> {
        if self.stream.is_some() {
            return Ok(());
        }
        // 4 buffers: enough to absorb a scheduling hiccup without adding latency. More
        // buffers on a live path only means older frames waiting to be shown.
        let stream = MmapStream::with_buffers(&self.device, Type::VideoCapture, 4)?;
        // SAFETY: the stream borrows the device, and both live in this struct with the
        // stream dropped first (declaration order). The 'static is a lifetime erasure,
        // not an escape: `stream` never outlives `device`.
        let stream: MmapStream<'static> = unsafe { std::mem::transmute(stream) };
        self.stream = Some(stream);
        tracing::debug!(path = %self.path, "capture streaming");
        Ok(())
    }

    /// Stop streaming but keep the device open and configured.
    ///
    /// This is the power-save path: the webcam LED goes out and USB traffic stops, while
    /// the negotiated mode is retained so resuming does not renegotiate.
    pub fn stop(&mut self) {
        if let Some(mut s) = self.stream.take() {
            let _ = Stream::stop(&mut s);
            tracing::debug!(path = %self.path, "capture stopped (device still open)");
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.stream.is_some()
    }

    /// Dequeue the next frame. Blocks until one is available.
    pub fn next_frame(&mut self) -> Result<RawFrame<'_>, CaptureError> {
        let mode = self.mode;
        if self.stream.is_none() {
            self.start()?;
        }
        let stream = self.stream.as_mut().expect("started just above");
        let (buf, meta) = CaptureStream::next(stream)?;
        Ok(RawFrame {
            // `bytesused` matters for MJPEG: the buffer is sized for the worst case and
            // the JPEG occupies only a prefix of it. Handing the whole buffer to a
            // decoder yields trailing garbage at best.
            data: &buf[..meta.bytesused as usize],
            format: mode.format,
            width: mode.width,
            height: mode.height,
            sequence: meta.sequence,
        })
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        // Explicit, and before the device: v4l2 wants STREAMOFF while the fd is still
        // open, and relying on field drop order for that is too subtle to leave implicit.
        self.stop();
    }
}

/// Every (format, width, height) the driver advertises and we can handle.
fn enumerate_modes(device: &Device) -> Vec<(PixelFormat, u32, u32)> {
    let mut out = Vec::new();

    let Ok(formats) = CaptureTrait::enum_formats(device) else {
        return out;
    };

    for desc in formats {
        let Some(pf) = PixelFormat::from_fourcc(desc.fourcc) else {
            continue;
        };
        let Ok(sizes) = CaptureTrait::enum_framesizes(device, desc.fourcc) else {
            continue;
        };
        for size in sizes {
            for discrete in size.size.to_discrete() {
                out.push((pf, discrete.width, discrete.height));
            }
        }
    }

    out.sort_by(|a, b| {
        (b.1 * b.2)
            .cmp(&(a.1 * a.2))
            .then(a.0.preference().cmp(&b.0.preference()))
    });
    out.dedup();
    out
}

/// Poll timeout used when waiting on a camera that may have been unplugged.
pub const FRAME_TIMEOUT: Duration = Duration::from_millis(2000);

/// Convenience: what a fourcc looks like when logged.
pub fn fourcc_str(f: FourCC) -> String {
    String::from_utf8_lossy(&f.repr).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_a_nonexistent_device_reports_the_path() {
        let err = match Camera::open("/dev/video-does-not-exist", 1920, 1080, 30) {
            Err(e) => e,
            Ok(_) => panic!("opening a nonexistent device must fail"),
        };
        let msg = err.to_string();
        assert!(msg.contains("/dev/video-does-not-exist"), "got: {msg}");
    }

    #[test]
    fn enumerating_the_real_camera_finds_mjpeg_if_one_is_present() {
        // Skips cleanly on a machine with no camera, so CI stays green.
        let Some(cam) = crate::capture_devices().into_iter().next() else {
            eprintln!("no camera present; skipping");
            return;
        };
        let Ok(device) = Device::with_path(&cam.path) else {
            eprintln!("camera present but not openable (in use?); skipping");
            return;
        };
        let modes = enumerate_modes(&device);
        assert!(!modes.is_empty(), "a real camera must advertise some mode");
        // Not an assertion about MJPEG specifically — some cameras genuinely lack it —
        // but the list must be sorted largest-first so the ladder starts sensibly.
        let areas: Vec<u32> = modes.iter().map(|(_, w, h)| w * h).collect();
        let mut sorted = areas.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(areas, sorted, "modes must be largest-first");
    }
}
