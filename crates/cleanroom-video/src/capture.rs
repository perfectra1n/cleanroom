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
        "{path} is a virtual camera ({label}), not a capture device — it advertises no \
         capturable format because nothing is producing video into it. Point video.device \
         at a real camera (`cleanroom-ctl devices` lists them)."
    )]
    VirtualDevice { path: String, label: String },

    #[error(
        "{path} was re-formatted underneath us while idle: we negotiated {negotiated}, \
         the driver now grants {granted}. Another process issued S_FMT on the shared \
         node; reopening to renegotiate."
    )]
    Reformatted {
        path: String,
        negotiated: String,
        granted: String,
    },

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
    /// The driver set `V4L2_BUF_FLAG_ERROR` on this buffer: it knows the payload is
    /// damaged (for UVC, typically an MJPG frame truncated by USB bandwidth pressure).
    /// The data is still handed over — V4L2's contract is "may be corrupt", not "is
    /// garbage" — but it must not be trusted as a picture. For MJPEG the decoder would
    /// catch it anyway; for raw formats a truncated payload is full-length and decodes
    /// "successfully" into a torn frame, so this flag is the only warning there is.
    pub corrupt: bool,
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
            return Err(no_usable_format(path));
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
        self.reassert_format()?;
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

    /// Re-apply the negotiated format immediately before STREAMON, and believe the
    /// read-back.
    ///
    /// V4L2 streams whatever the *last* `S_FMT` on the device was, not the one we
    /// negotiated at open — and `open` and `start` can be far apart. The power-save path
    /// calls `stop`/`start` repeatedly, and while we hold the fd without streaming,
    /// other processes can issue their own `S_FMT` on the shared node (WirePlumber's
    /// v4l2 monitor probing devices is the usual suspect). When that happens `self.mode`
    /// lies about what frames will contain: observed live as a C920 negotiated to MJPG
    /// delivering YUYV, which the MJPEG decoder rejected with "Not a JPEG file: starts
    /// with 0x00 0x0a" — raw luma, not a JPEG header.
    ///
    /// Same trust-nothing pattern as `open`: ask, then verify what the driver *returned*
    /// rather than assuming `S_FMT` honoured the request. The policy for a mismatch
    /// lives in [`reconcile_streamon_format`].
    fn reassert_format(&mut self) -> Result<(), CaptureError> {
        let want = Format::new(self.mode.width, self.mode.height, self.mode.format.fourcc());
        let granted = CaptureTrait::set_format(&self.device, &want)?;
        match reconcile_streamon_format(self.mode, &granted) {
            Reassertion::Unchanged => Ok(()),
            Reassertion::Adopt(format) => {
                tracing::warn!(
                    path = %self.path,
                    negotiated = %self.mode,
                    granted = %fourcc_str(granted.fourcc),
                    "another process re-formatted the device while it was idle \
                     (WirePlumber's v4l2 monitor probing is the usual suspect); \
                     adopting the granted format"
                );
                self.mode.format = format;
                Ok(())
            }
            Reassertion::Renegotiate => Err(CaptureError::Reformatted {
                path: self.path.clone(),
                negotiated: self.mode.to_string(),
                granted: format!(
                    "{}@{}x{}",
                    fourcc_str(granted.fourcc),
                    granted.width,
                    granted.height
                ),
            }),
        }
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
            corrupt: meta.flags.contains(v4l::buffer::Flags::ERROR),
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

/// What to do about the format the driver granted at stream start.
///
/// See [`reconcile_streamon_format`] for how one is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reassertion {
    /// The driver still holds what we negotiated. Proceed.
    Unchanged,
    /// Same geometry, different but decodable fourcc. Update `Camera::mode` and proceed:
    /// the decoder dispatches per-frame on `RawFrame::format`, so nothing downstream
    /// cares which of our formats arrives — only that the mode tells the truth.
    Adopt(PixelFormat),
    /// The geometry moved, or the fourcc is one we cannot decode. Fail the start so the
    /// pipeline restarts and renegotiates *everything* — the sink and every buffer
    /// between here and it were sized from the negotiated geometry, so adapting in
    /// place would just move the corruption downstream.
    Renegotiate,
}

/// Decide what to do given the mode negotiated at open and the format `S_FMT` granted
/// at stream start.
///
/// A pure function so the policy is testable without hardware; the driver interaction
/// lives in [`Camera::reassert_format`].
fn reconcile_streamon_format(negotiated: CaptureMode, granted: &Format) -> Reassertion {
    if (granted.width, granted.height) != (negotiated.width, negotiated.height) {
        return Reassertion::Renegotiate;
    }
    match PixelFormat::from_fourcc(granted.fourcc) {
        Some(f) if f == negotiated.format => Reassertion::Unchanged,
        Some(f) => Reassertion::Adopt(f),
        None => Reassertion::Renegotiate,
    }
}

/// The error for a device that advertises nothing we can decode, enriched by what the
/// node actually *is*.
fn no_usable_format(path: &str) -> CaptureError {
    let dev = crate::device::probe(std::path::Path::new(path));
    classify_no_format(path, &dev)
}

/// Turn "no usable format" into an error that names the real mistake when there is one.
///
/// The bare message earned this the hard way: pointed at a v4l2loopback node, `open`
/// failed with "reports no pixel format we can use" — true, and useless. The actionable
/// fact is that the node is a *virtual camera*, so the config points Cleanroom at an
/// output (possibly its own) rather than at hardware. Split from [`no_usable_format`]
/// so the classification policy is testable without a device to probe.
fn classify_no_format(path: &str, dev: &crate::device::VideoDevice) -> CaptureError {
    if dev.is_virtual {
        // The driver name ("v4l2 loopback") identifies the mechanism; the card label is
        // the fallback because the virtual classification can come from either.
        let label = if dev.driver.is_empty() {
            dev.card.clone()
        } else {
            dev.driver.clone()
        };
        return CaptureError::VirtualDevice {
            path: path.to_string(),
            label,
        };
    }
    CaptureError::NoUsableFormat {
        path: path.to_string(),
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

    /// The negotiated mode is what `start` asked for, so the common case must be
    /// completely silent — no mode rewrite, no warning, no restart.
    #[test]
    fn an_identical_grant_at_stream_start_proceeds_unchanged() {
        let negotiated = CaptureMode {
            format: PixelFormat::Mjpeg,
            width: 1920,
            height: 1080,
            fps: 30,
        };
        let granted = Format::new(1920, 1080, FourCC::new(b"MJPG"));
        assert_eq!(
            reconcile_streamon_format(negotiated, &granted),
            Reassertion::Unchanged
        );
    }

    /// The live failure this whole re-assertion exists for: a C920 negotiated MJPG at
    /// open, WirePlumber's monitor issued its own S_FMT while we held the fd idle, and
    /// STREAMON delivered YUYV frames into the MJPEG decode path ("Not a JPEG file:
    /// starts with 0x00 0x0a" — raw luma). Same geometry, decodable format: the decoder
    /// dispatches per-frame on `RawFrame::format`, so adopting it keeps streaming
    /// rather than tearing down the sink over a fourcc.
    #[test]
    fn a_decodable_format_swap_at_the_same_size_is_adopted() {
        let negotiated = CaptureMode {
            format: PixelFormat::Mjpeg,
            width: 1920,
            height: 1080,
            fps: 30,
        };
        let granted = Format::new(1920, 1080, FourCC::new(b"YUYV"));
        assert_eq!(
            reconcile_streamon_format(negotiated, &granted),
            Reassertion::Adopt(PixelFormat::Yuyv),
            "the mode must be corrected to what the frames will actually contain"
        );
    }

    /// Drivers disagree on spelling (YUY2 vs YUYV). An alias of the format we already
    /// negotiated is the same format, and treating it as a change would warn about a
    /// re-format that never happened on every single stream start.
    #[test]
    fn an_aliased_spelling_of_the_negotiated_format_is_not_a_change() {
        let negotiated = CaptureMode {
            format: PixelFormat::Yuyv,
            width: 1280,
            height: 720,
            fps: 30,
        };
        let granted = Format::new(1280, 720, FourCC::new(b"YUY2"));
        assert_eq!(
            reconcile_streamon_format(negotiated, &granted),
            Reassertion::Unchanged
        );
    }

    /// The sink and every buffer between the camera and it were sized from the
    /// negotiated geometry. Adapting a size change in place would only move the
    /// corruption downstream, so it must fail the start and renegotiate everything.
    #[test]
    fn a_size_change_at_stream_start_forces_a_renegotiation() {
        let negotiated = CaptureMode {
            format: PixelFormat::Mjpeg,
            width: 1920,
            height: 1080,
            fps: 30,
        };
        let granted = Format::new(640, 480, FourCC::new(b"MJPG"));
        assert_eq!(
            reconcile_streamon_format(negotiated, &granted),
            Reassertion::Renegotiate
        );
    }

    /// A fourcc we cannot decode must never be adopted — every frame would fail the
    /// decoder and thirty of those in a row is a reopen anyway, minus the diagnosis.
    #[test]
    fn an_undecodable_fourcc_at_stream_start_forces_a_renegotiation() {
        let negotiated = CaptureMode {
            format: PixelFormat::Mjpeg,
            width: 1920,
            height: 1080,
            fps: 30,
        };
        let granted = Format::new(1920, 1080, FourCC::new(b"H264"));
        assert_eq!(
            reconcile_streamon_format(negotiated, &granted),
            Reassertion::Renegotiate
        );
    }

    fn loopback_node(path: &str) -> crate::device::VideoDevice {
        crate::device::VideoDevice {
            path: path.into(),
            card: "Cleanroom Camera".into(),
            driver: "v4l2 loopback".into(),
            kind: crate::device::NodeKind::Output,
            is_virtual: true,
            accessible: true,
        }
    }

    /// The unhelpful error that cost a debugging session: video.device pointed at a
    /// v4l2loopback node, and "reports no pixel format we can use" sent the user
    /// everywhere but at the config. The error must name what the node *is*.
    #[test]
    fn a_virtual_node_with_no_formats_is_named_as_virtual() {
        let err = classify_no_format("/dev/video0", &loopback_node("/dev/video0"));
        let msg = err.to_string();
        assert!(msg.contains("virtual camera"), "got: {msg}");
        assert!(msg.contains("v4l2 loopback"), "got: {msg}");
        assert!(msg.contains("/dev/video0"), "got: {msg}");
    }

    /// A real camera with no decodable format is a different problem (exotic hardware,
    /// missing formats) and must keep the plain message rather than accuse it of being
    /// virtual.
    #[test]
    fn a_real_camera_with_no_formats_keeps_the_plain_message() {
        let mut dev = loopback_node("/dev/video2");
        dev.card = "Weird Industrial Cam".into();
        dev.driver = "uvcvideo".into();
        dev.kind = crate::device::NodeKind::Capture;
        dev.is_virtual = false;
        let msg = classify_no_format("/dev/video2", &dev).to_string();
        assert!(msg.contains("no pixel format we can use"), "got: {msg}");
        assert!(!msg.contains("virtual"), "got: {msg}");
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
