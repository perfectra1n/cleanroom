//! Detecting whether anything is actually watching the virtual camera.
//!
//! This drives power save: with no consumer we can stop capture, turn the webcam LED
//! off and stop decoding. Getting it wrong in the *unsafe* direction blanks someone's
//! camera mid-call, so the contract here is deliberately paranoid.
//!
//! There are three ways to answer this question and only one of them works.
//!
//! 1. **`fuser` / scanning `/proc/*/fd`.** Fails silently under sandboxing: the
//!    `/proc/PID/fd` magic links of processes outside your user namespace fail the
//!    kernel's ptrace-mode check, so a Flatpak'd browser reading your camera looks
//!    exactly like nobody reading it. It reports a confident "no consumers" rather than
//!    an error, which is the worst possible failure shape.
//! 2. **inotify open/close counting.** Drifts low, because the kernel coalesces adjacent
//!    identical events — and browsers probe-open cameras while a capture fd is already
//!    open, so `IN_OPEN` events get merged and the running count sags below reality.
//! 3. **`V4L2_EVENT_PRI_CLIENT_USAGE`** (v4l2loopback ≥0.13). An absolute count from the
//!    kernel via the device fd, so no namespace can hide from it, delivered on
//!    `STREAMON`/`STREAMOFF` rather than `open()` — which means a browser probing the
//!    device without streaming correctly does not count.
//!
//! Only (3) is used. No Rust code anywhere else appears to use this event, so the ioctl
//! plumbing is hand-rolled below.

use std::os::fd::{AsRawFd, OwnedFd};
use std::path::Path;
use std::time::Duration;

/// `V4L2_EVENT_PRIVATE_START + V4L2LOOPBACK_EVENT_OFFSET + 1`.
///
/// Defined in v4l2loopback.c rather than in any UAPI header, so it must be hardcoded.
const V4L2_EVENT_PRI_CLIENT_USAGE: u32 = 0x0800_0000 + 0x08E0_0000 + 1;

/// Ubuntu shipped a downstream variant of the same event at the bare private-start
/// value. Try the modern number first and fall back, or the count silently never
/// arrives on those kernels.
const V4L2_EVENT_PRI_CLIENT_USAGE_LEGACY: u32 = 0x0800_0000;

/// Deliver the current state immediately on subscribe, not only on the next change.
/// Without this we would sit at "unknown" until a consumer happened to come or go.
const V4L2_EVENT_SUB_FL_SEND_INITIAL: u32 = 0x1;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct EventSubscription {
    type_: u32,
    id: u32,
    flags: u32,
    reserved: [u32; 5],
}

/// `struct v4l2_event` on 64-bit. Only the fields we read are named; the rest is padding
/// we must get exactly right or every offset after it shifts.
#[repr(C)]
#[derive(Clone, Copy)]
struct Event {
    type_: u32,
    _pad: u32,
    /// First `u32` of the event union is the client-usage count.
    count: u32,
    _union_rest: [u8; 60],
    /// How many more events are queued behind this one.
    pending: u32,
    sequence: u32,
    timestamp: [u64; 2],
    id: u32,
    reserved: [u32; 8],
}

impl Default for Event {
    fn default() -> Self {
        // SAFETY: all fields are integers or byte arrays; zero is a valid bit pattern.
        unsafe { std::mem::zeroed() }
    }
}

nix::ioctl_write_ptr!(vidioc_subscribe_event, b'V', 90, EventSubscription);
nix::ioctl_read!(vidioc_dqevent, b'V', 89, Event);

/// Watches one loopback device's consumer count.
pub struct ConsumerWatch {
    fd: OwnedFd,
    /// `None` means "we do not know", which callers must treat as in-use.
    count: Option<u32>,
}

impl ConsumerWatch {
    /// Subscribe to consumer-usage events on a loopback device.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(path)?;
        let fd = OwnedFd::from(file);

        let mut subscribed = false;
        for type_ in [
            V4L2_EVENT_PRI_CLIENT_USAGE,
            V4L2_EVENT_PRI_CLIENT_USAGE_LEGACY,
        ] {
            let sub = EventSubscription {
                type_,
                id: 0,
                flags: V4L2_EVENT_SUB_FL_SEND_INITIAL,
                reserved: [0; 5],
            };
            // SAFETY: `sub` is a correctly-shaped v4l2_event_subscription and the fd is
            // an open V4L2 device.
            if unsafe { vidioc_subscribe_event(fd.as_raw_fd(), &sub) }.is_ok() {
                subscribed = true;
                tracing::debug!(path = %path.display(), event_type = format!("{type_:#x}"),
                    "subscribed to client-usage events");
                break;
            }
        }

        if !subscribed {
            // Not fatal. The pipeline still runs; it just cannot power down, which is a
            // far better outcome than blanking the camera on a guess.
            tracing::warn!(
                path = %path.display(),
                "device does not support client-usage events (v4l2loopback < 0.13?); \
                 power save will stay off"
            );
        }

        Ok(Self {
            fd,
            count: if subscribed { None } else { Some(1) },
        })
    }

    /// Drain any queued events and return the current count.
    ///
    /// `None` means the count is not trustworthy. **Callers must treat `None` as
    /// in-use.** A detection failure must never be able to switch someone's camera off.
    pub fn poll(&mut self, timeout: Duration) -> Option<u32> {
        use nix::poll::{PollFd, PollFlags, PollTimeout};
        use std::os::fd::AsFd;

        let mut pfd = [PollFd::new(self.fd.as_fd(), PollFlags::POLLPRI)];
        let ms = timeout.as_millis().min(i32::MAX as u128) as u16;
        // V4L2 events arrive as *priority* readable, not ordinary readable.
        match nix::poll::poll(&mut pfd, PollTimeout::from(ms)) {
            Ok(0) => return self.count,
            Ok(_) => {}
            Err(_) => return self.count,
        }

        loop {
            let mut ev = Event::default();
            // SAFETY: `ev` is a correctly-sized v4l2_event and the fd is open.
            match unsafe { vidioc_dqevent(self.fd.as_raw_fd(), &mut ev) } {
                Ok(_) => {
                    if ev.type_ == V4L2_EVENT_PRI_CLIENT_USAGE
                        || ev.type_ == V4L2_EVENT_PRI_CLIENT_USAGE_LEGACY
                    {
                        self.count = Some(ev.count);
                    }
                    // Stop once the kernel says nothing is queued behind this one.
                    if ev.pending == 0 {
                        break;
                    }
                }
                // ENOENT is the documented "queue empty" answer, not a fault.
                Err(_) => break,
            }
        }

        self.count
    }

    /// Whether anything is streaming from the device.
    ///
    /// Unknown counts as in use, deliberately.
    pub fn in_use(&self) -> bool {
        self.count.is_none_or(|c| c > 0)
    }

    /// The raw count, or `None` when untrustworthy.
    pub fn count(&self) -> Option<u32> {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_structs_match_the_kernel_abi() {
        // If these drift, `count` and `pending` are read from the wrong offsets and the
        // consumer count becomes plausible nonsense rather than an obvious failure.
        assert_eq!(std::mem::size_of::<Event>(), 136, "struct v4l2_event");
        assert_eq!(
            std::mem::size_of::<EventSubscription>(),
            32,
            "struct v4l2_event_subscription"
        );
        // count is the first u32 of the union, at byte 8; pending sits at byte 72.
        assert_eq!(std::mem::offset_of!(Event, count), 8);
        assert_eq!(std::mem::offset_of!(Event, pending), 72);
    }

    #[test]
    fn the_event_number_matches_v4l2loopback() {
        // V4L2_EVENT_PRIVATE_START + V4L2LOOPBACK_EVENT_OFFSET + 1
        assert_eq!(V4L2_EVENT_PRI_CLIENT_USAGE, 0x10E0_0001);
    }

    #[test]
    fn unknown_is_treated_as_in_use() {
        // The single most important behaviour in this module. A watch that has not yet
        // learned the count must never report "nobody is watching" — that is how a
        // camera goes dark in the middle of a call.
        let w = ConsumerWatch {
            fd: std::fs::File::open("/dev/null").unwrap().into(),
            count: None,
        };
        assert!(w.in_use(), "an unknown count must be treated as in use");
        assert_eq!(w.count(), None);
    }

    #[test]
    fn a_known_zero_is_idle_and_a_known_positive_is_busy() {
        let mut w = ConsumerWatch {
            fd: std::fs::File::open("/dev/null").unwrap().into(),
            count: Some(0),
        };
        assert!(!w.in_use());
        w.count = Some(2);
        assert!(w.in_use());
    }

    #[test]
    fn watching_a_real_loopback_device_reports_something_sane() {
        let Some(dev) = crate::device::enumerate()
            .into_iter()
            .find(|d| d.is_virtual && d.accessible)
        else {
            eprintln!("no loopback device present; skipping");
            return;
        };
        let Ok(mut w) = ConsumerWatch::open(&dev.path) else {
            eprintln!("could not open loopback device; skipping");
            return;
        };
        // SEND_INITIAL means the count should arrive without waiting for a transition.
        let c = w.poll(Duration::from_millis(200));
        eprintln!("{} consumer count: {c:?}", dev.path.display());
    }
}
