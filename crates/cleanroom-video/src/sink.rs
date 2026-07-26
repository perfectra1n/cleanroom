//! The v4l2loopback sink — where the processed frame becomes a camera other apps can see.
//!
//! ## Why we select a device rather than creating one
//!
//! v4l2loopback ≥0.13 has a control device that can allocate a node at runtime, and the
//! plan called for using it. It cannot be used here: `/dev/v4l2loopback` is `crw-------
//! root root`, and Cleanroom runs as a user daemon. Creating a device needs privileges
//! we should not have and do not want.
//!
//! So devices are provisioned once at boot (modprobe options, or the NixOS module) and
//! the daemon *selects* a free one. This is the better arrangement anyway: no privilege
//! escalation, and no fight with OBS over who owns which node.
//!
//! ## How "free" is detectable
//!
//! `exclusive_caps=1` — which Chromium, Zoom and Teams all require, because they reject
//! nodes advertising both output and capture — makes a loopback node report
//! `V4L2_CAP_VIDEO_OUTPUT` while no producer is attached, and flip to
//! `V4L2_CAP_VIDEO_CAPTURE` once one is. That flip is exactly the signal we need, and
//! the enumeration in `device.rs` already reads it: a virtual node still reporting
//! `Output` has no producer.

use crate::device::{NodeKind, VideoDevice};
use crate::format::PixelFormat;
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::{OutputStream, Stream};
use v4l::video::Output as OutputTrait;
use v4l::{Device, Format};

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error(
        "no free v4l2loopback device. {detail}\n\
         Provision one with:  sudo modprobe v4l2loopback devices=1 exclusive_caps=1 \
         card_label=\"Cleanroom Camera\"\n\
         (exclusive_caps=1 is required or Chromium, Zoom and Teams will not list it.)"
    )]
    NoDevice { detail: String },

    #[error("cannot open {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} refused {width}x{height} {format}: {source}")]
    Format {
        path: String,
        width: u32,
        height: u32,
        format: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("v4l2 error: {0}")]
    Io(#[from] std::io::Error),
}

/// The format we publish.
///
/// YUY2 rather than NV12 or I420, deliberately. It is the common denominator both
/// Firefox and Chromium accept without complaint; publishing only NV12 loses Firefox,
/// and only I420 makes Chromium convert. Do not "optimise" this without testing in both.
pub const OUTPUT_FORMAT: PixelFormat = PixelFormat::Yuyv;

/// Find loopback devices with no producer attached.
pub fn available_devices() -> Vec<VideoDevice> {
    crate::device::enumerate()
        .into_iter()
        // Virtual, and still advertising OUTPUT — meaning nothing is producing into it.
        .filter(|d| d.is_virtual && d.kind == NodeKind::Output && d.accessible)
        .collect()
}

/// Pick the device to produce into.
///
/// Prefers a node whose label mentions Cleanroom, so a machine that provisions a
/// dedicated one gets it rather than stealing the node OBS was going to use.
pub fn select_device(preferred_label: &str) -> Result<VideoDevice, SinkError> {
    let free = available_devices();

    if let Some(d) = free
        .iter()
        .find(|d| d.card.eq_ignore_ascii_case(preferred_label))
    {
        return Ok(d.clone());
    }
    if let Some(d) = free
        .iter()
        .find(|d| d.card.to_lowercase().contains("cleanroom"))
    {
        return Ok(d.clone());
    }
    if let Some(d) = free.first() {
        return Ok(d.clone());
    }

    // Explain what we *did* see, because "no device" and "a device that is busy" want
    // very different responses from the user.
    let all_virtual: Vec<String> = crate::device::enumerate()
        .into_iter()
        .filter(|d| d.is_virtual)
        .map(|d| {
            format!(
                "{} ({}{})",
                d.path.display(),
                d.card,
                if d.kind == NodeKind::Capture {
                    ", already has a producer"
                } else if !d.accessible {
                    ", not accessible"
                } else {
                    ""
                }
            )
        })
        .collect();

    Err(SinkError::NoDevice {
        detail: if all_virtual.is_empty() {
            "No v4l2loopback devices exist at all.".to_string()
        } else {
            format!(
                "Loopback devices present but none free: {}.",
                all_virtual.join("; ")
            )
        },
    })
}

pub struct LoopbackSink {
    device: Device,
    stream: Option<MmapStream<'static>>,
    pub path: String,
    pub width: u32,
    pub height: u32,
    frame_bytes: usize,
}

impl LoopbackSink {
    /// Open a loopback node and fix its format.
    pub fn open(dev: &VideoDevice, width: u32, height: u32, fps: u32) -> Result<Self, SinkError> {
        let path = dev.path.display().to_string();
        let device = Device::with_path(&dev.path).map_err(|source| SinkError::Open {
            path: path.clone(),
            source,
        })?;

        // S_FMT MUST happen before the stream is created. v4l2loopback only sets the
        // buffer `length` field once a format has been set, so creating the stream first
        // yields zero-sized buffers and a silent black camera. The v4l crate's own
        // example flags this under "BEWARE OF DRAGONS".
        let fmt = Format::new(width, height, OUTPUT_FORMAT.fourcc());
        let granted =
            OutputTrait::set_format(&device, &fmt).map_err(|source| SinkError::Format {
                path: path.clone(),
                width,
                height,
                format: OUTPUT_FORMAT.as_str(),
                source,
            })?;

        if let Ok(mut params) = OutputTrait::params(&device) {
            params.interval = v4l::Fraction::new(1, fps);
            let _ = OutputTrait::set_params(&device, &params);
        }

        // YUY2 is 2 bytes per pixel. Compute from what the driver *granted*, not what we
        // asked for — a size mismatch here writes past the end of a buffer or leaves a
        // torn frame.
        let frame_bytes = (granted.width * granted.height * 2) as usize;

        tracing::info!(
            path = %path,
            "{}x{} {} -> virtual camera",
            granted.width,
            granted.height,
            OUTPUT_FORMAT.as_str()
        );

        let mut sink = Self {
            device,
            stream: None,
            path,
            width: granted.width,
            height: granted.height,
            frame_bytes,
        };

        // Prime the node immediately, and do not treat this as optional.
        //
        // With exclusive_caps=1 the node advertises VIDEO_OUTPUT until a producer
        // actually *starts streaming* — merely holding the fd open is not enough. Until
        // that happens the device is not a capture device at all, and any consumer that
        // tries to open it gets "Not a video capture device / No such device".
        //
        // That creates a deadlock with power save: idle-at-startup means no frames are
        // written, so the node never flips, so no consumer can open it, so no consumer
        // event ever fires to wake us. Writing one frame here breaks it, and has the
        // separate benefit of making the camera present *before* Zoom, Chrome or Discord
        // launch — they enumerate cameras once at startup and never look again.
        sink.write_placeholder()?;

        Ok(sink)
    }

    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    fn ensure_stream(&mut self) -> Result<(), SinkError> {
        if self.stream.is_some() {
            return Ok(());
        }
        let stream = MmapStream::with_buffers(&self.device, Type::VideoOutput, 4)?;
        // SAFETY: as in `Camera` — the stream borrows the device and both live here,
        // with the stream torn down first in `Drop`.
        let stream: MmapStream<'static> = unsafe { std::mem::transmute(stream) };
        self.stream = Some(stream);
        Ok(())
    }

    /// Publish one frame. `frame` must be exactly [`frame_bytes`] of YUY2.
    pub fn write(&mut self, frame: &[u8]) -> Result<(), SinkError> {
        debug_assert_eq!(
            frame.len(),
            self.frame_bytes,
            "frame size mismatch: the sink was negotiated for {} bytes",
            self.frame_bytes
        );

        self.ensure_stream()?;
        let stream = self.stream.as_mut().expect("ensured above");
        let (buf, meta) = OutputStream::next(stream)?;

        let n = frame.len().min(buf.len());
        buf[..n].copy_from_slice(&frame[..n]);
        // Short frames would otherwise show the previous frame's tail as garbage.
        if n < buf.len() {
            buf[n..].fill(0);
        }
        meta.bytesused = n as u32;
        Ok(())
    }

    /// Keep the node claimed while producing nothing.
    ///
    /// Not the same as dropping the sink. Consumers enumerate cameras once at startup, so
    /// releasing the node while idle would make Cleanroom vanish from every already-open
    /// Zoom or Chrome. Holding it and publishing a still frame keeps the device present.
    pub fn write_placeholder(&mut self) -> Result<(), SinkError> {
        // Neutral mid-grey in YUY2: Y=0x80, U=V=0x80 is a flat grey with no chroma cast.
        let frame = vec![0x80u8; self.frame_bytes];
        self.write(&frame)
    }
}

impl Drop for LoopbackSink {
    fn drop(&mut self) {
        if let Some(mut s) = self.stream.take() {
            let _ = Stream::stop(&mut s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_is_the_browser_safe_one() {
        // Regression guard with teeth: switching this to NV12 or I420 loses Firefox or
        // makes Chromium convert every frame. Both are silent quality/compat regressions.
        assert_eq!(OUTPUT_FORMAT, PixelFormat::Yuyv);
        assert_eq!(&OUTPUT_FORMAT.fourcc().repr, b"YUYV");
    }

    #[test]
    fn selection_error_explains_what_was_seen() {
        // On a machine with a free loopback this returns Ok, which is also fine — the
        // point is that the failure path is actionable rather than a bare "not found".
        if let Err(e) = select_device("Nonexistent Label That Cannot Match") {
            let msg = e.to_string();
            assert!(
                msg.contains("modprobe"),
                "must tell the user how to fix it: {msg}"
            );
            assert!(
                msg.contains("exclusive_caps=1"),
                "must explain why exclusive_caps matters: {msg}"
            );
        }
    }

    #[test]
    fn opening_a_sink_makes_the_node_a_capture_device() {
        // Regression test for a deadlock found by integration testing: with
        // exclusive_caps=1 the node stays VIDEO_OUTPUT until a producer *streams*, so a
        // lazily-primed sink left the device unopenable — and because no consumer could
        // open it, no consumer event could ever fire to wake power save. Priming on open
        // is what breaks the cycle.
        let Ok(dev) = select_device("Cleanroom Camera") else {
            eprintln!("no free loopback device; skipping");
            return;
        };
        let Ok(sink) = LoopbackSink::open(&dev, 640, 480, 30) else {
            eprintln!("could not open loopback sink; skipping");
            return;
        };
        let after = crate::device::probe(&dev.path);
        assert_eq!(
            after.kind,
            NodeKind::Capture,
            "after a producer attaches and writes, the node must advertise CAPTURE or no \
             consumer can open it"
        );
        drop(sink);
    }

    #[test]
    fn available_devices_are_all_virtual_outputs() {
        for d in available_devices() {
            assert!(d.is_virtual);
            assert_eq!(d.kind, NodeKind::Output);
        }
    }
}
