//! V4L2 device enumeration.
//!
//! This looks like it should be three lines of `read_dir("/dev")` and is not, because
//! `/dev/video*` is a far messier namespace than it appears:
//!
//! * **A single UVC webcam exposes several nodes.** On the reference machine the C922 is
//!   `/dev/video0` *and* `/dev/video1`, and only the first can capture video — the second
//!   is a metadata node. Offering it in a camera picker produces a camera that "doesn't
//!   work" for reasons the user cannot possibly diagnose.
//! * **`capabilities` lies.** `struct v4l2_capability` has two fields. `capabilities` is
//!   the union across *every* node the driver owns, so a metadata node cheerfully reports
//!   `VIDEO_CAPTURE` because a sibling node has it. Only `device_caps` describes the node
//!   you actually opened, and it is only valid when `V4L2_CAP_DEVICE_CAPS` is set.
//! * **Virtual cameras must be excluded from *inputs*.** Listing v4l2loopback devices as
//!   capture sources invites someone to point Cleanroom at its own output, or at OBS's.
//!
//! We query with an ioctl rather than reading sysfs because sysfs does not expose
//! per-node capabilities at all — the earlier sysfs-based attempt classified the C922's
//! metadata node as a working camera.

use std::ffi::CStr;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};

// --- V4L2 ABI ----------------------------------------------------------------------
// From linux/videodev2.h. Declared here rather than pulled from a -sys crate so the
// layout assumptions are visible and checked by a test below.

const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
const V4L2_CAP_VIDEO_OUTPUT: u32 = 0x0000_0002;
const V4L2_CAP_META_CAPTURE: u32 = 0x0080_0000;
/// When set in `capabilities`, `device_caps` is populated and is the field to trust.
const V4L2_CAP_DEVICE_CAPS: u32 = 0x8000_0000;

#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

impl Default for V4l2Capability {
    fn default() -> Self {
        // SAFETY: every field is a plain integer or byte array; all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

nix::ioctl_read!(vidioc_querycap, b'V', 0, V4l2Capability);

// --- public API ---------------------------------------------------------------------

/// What a `/dev/video*` node actually is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// A real capture device we can use as an input.
    Capture,
    /// A metadata node belonging to a camera. Not capturable. `/dev/video1` on most
    /// UVC cameras.
    Metadata,
    /// An output-capable node: v4l2loopback, or a hardware encoder.
    Output,
    /// Present but not something we can classify usefully.
    Other,
}

#[derive(Debug, Clone)]
pub struct VideoDevice {
    pub path: PathBuf,
    /// The driver's card label, e.g. "C922 Pro Stream Webcam".
    pub card: String,
    /// Kernel driver, e.g. "uvcvideo" or "v4l2 loopback".
    pub driver: String,
    pub kind: NodeKind,
    /// True when this node is one of ours or another app's virtual camera.
    pub is_virtual: bool,
    /// False when the device exists but could not be opened — almost always because
    /// another process holds it. Reported rather than hidden: "in use" is far more
    /// useful to a user than a camera silently missing from the list.
    pub accessible: bool,
}

impl VideoDevice {
    /// Suitable as a capture source: a real camera, not a metadata node, not virtual.
    pub fn is_usable_input(&self) -> bool {
        self.kind == NodeKind::Capture && !self.is_virtual
    }
}

/// Drivers that indicate a virtual camera rather than real hardware.
///
/// Matching on the driver name is more reliable than the card label, which users can
/// and do rename — `card_label` is a v4l2loopback module parameter.
const VIRTUAL_DRIVERS: &[&str] = &["v4l2 loopback", "v4l2loopback", "akvcam", "vivid"];

/// Card-label fragments used by known virtual cameras. Belt and braces for the case
/// where a fork reports a different driver string.
const VIRTUAL_LABELS: &[&str] = &["v4l2loopback", "obs virtual", "virtual camera", "cleanroom"];

/// Enumerate every `/dev/video*` node, classified.
pub fn enumerate() -> Vec<VideoDevice> {
    let mut nodes: Vec<(u32, PathBuf)> = Vec::new();

    let Ok(entries) = std::fs::read_dir("/dev") else {
        return Vec::new();
    };
    for e in entries.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix("video") else {
            continue;
        };
        // "video" followed by digits only — skip things like "videoN-something".
        let Ok(idx) = rest.parse::<u32>() else {
            continue;
        };
        nodes.push((idx, e.path()));
    }
    // Numeric order, so /dev/video2 does not sort before /dev/video10.
    nodes.sort_by_key(|(i, _)| *i);

    nodes.into_iter().map(|(_, p)| probe(&p)).collect()
}

/// Cameras usable as inputs, in a stable order.
pub fn capture_devices() -> Vec<VideoDevice> {
    enumerate()
        .into_iter()
        .filter(|d| d.is_usable_input())
        .collect()
}

/// Classify one node.
pub fn probe(path: &Path) -> VideoDevice {
    let mut dev = VideoDevice {
        path: path.to_path_buf(),
        card: String::new(),
        driver: String::new(),
        kind: NodeKind::Other,
        is_virtual: false,
        accessible: false,
    };

    // O_NONBLOCK so a misbehaving driver cannot wedge enumeration, and no O_EXCL: we are
    // only asking what the node is, and must not disturb a device someone else is using.
    let fd = match open_nonblocking(path) {
        Ok(fd) => fd,
        Err(_) => {
            // Fall back to sysfs for the label so an in-use device still shows a name.
            dev.card = sysfs_name(path).unwrap_or_default();
            return dev;
        }
    };
    dev.accessible = true;

    let mut caps = V4l2Capability::default();
    // SAFETY: `caps` is a correctly-sized, correctly-aligned V4l2Capability, and the fd
    // is open for the duration of the call.
    if unsafe { vidioc_querycap(fd.as_raw_fd(), &mut caps) }.is_err() {
        dev.card = sysfs_name(path).unwrap_or_default();
        return dev;
    }

    dev.card = cstr_field(&caps.card);
    dev.driver = cstr_field(&caps.driver);

    // The whole reason this function exists. `capabilities` is the union across all of
    // the driver's nodes; only `device_caps` describes *this* one, and only when the
    // driver says it populated it.
    let effective = if caps.capabilities & V4L2_CAP_DEVICE_CAPS != 0 {
        caps.device_caps
    } else {
        caps.capabilities
    };

    dev.kind = if effective & V4L2_CAP_VIDEO_CAPTURE != 0 {
        NodeKind::Capture
    } else if effective & V4L2_CAP_META_CAPTURE != 0 {
        NodeKind::Metadata
    } else if effective & V4L2_CAP_VIDEO_OUTPUT != 0 {
        NodeKind::Output
    } else {
        NodeKind::Other
    };

    let driver_l = dev.driver.to_ascii_lowercase();
    let card_l = dev.card.to_ascii_lowercase();
    dev.is_virtual = VIRTUAL_DRIVERS.iter().any(|v| driver_l.contains(v))
        || VIRTUAL_LABELS.iter().any(|v| card_l.contains(v));

    dev
}

fn open_nonblocking(path: &Path) -> std::io::Result<OwnedFd> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)?;
    Ok(OwnedFd::from(file))
}

/// The driver's NUL-padded fixed-size strings are not guaranteed NUL-terminated when
/// full, so this handles both forms.
fn cstr_field(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    CStr::from_bytes_with_nul(&[&bytes[..end], &[0]].concat())
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn sysfs_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy();
    std::fs::read_to_string(format!("/sys/class/video4linux/{name}/name"))
        .ok()
        .map(|s| s.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_struct_matches_the_kernel_abi() {
        // If this drifts, every field after it is read from the wrong offset and the
        // classification silently becomes nonsense rather than failing.
        assert_eq!(std::mem::size_of::<V4l2Capability>(), 104);
        assert_eq!(std::mem::align_of::<V4l2Capability>(), 4);
    }

    #[test]
    fn parses_nul_padded_driver_strings() {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(b"uvcvideo");
        assert_eq!(cstr_field(&buf), "uvcvideo");
    }

    #[test]
    fn parses_a_completely_full_field_without_a_terminator() {
        // v4l2's fixed-size fields are not guaranteed NUL-terminated when full.
        let buf = [b'x'; 16];
        assert_eq!(cstr_field(&buf), "x".repeat(16));
    }

    #[test]
    fn enumeration_does_not_panic_and_is_ordered_numerically() {
        // Must be safe on any machine, including one with no video devices at all.
        let devs = enumerate();
        let indices: Vec<u32> = devs
            .iter()
            .filter_map(|d| {
                d.path
                    .file_name()?
                    .to_string_lossy()
                    .strip_prefix("video")?
                    .parse()
                    .ok()
            })
            .collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(
            indices, sorted,
            "/dev/video10 must not sort before /dev/video2"
        );
    }

    #[test]
    fn usable_inputs_exclude_metadata_and_virtual_nodes() {
        for d in capture_devices() {
            assert_eq!(
                d.kind,
                NodeKind::Capture,
                "{:?} is not a capture node",
                d.path
            );
            assert!(!d.is_virtual, "{:?} is a virtual camera", d.path);
        }
    }
}
