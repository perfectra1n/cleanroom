//! Pixel formats and the capture-mode ladder.
//!
//! Choosing a capture mode is not "ask for 1920x1080@30 and hope". USB bandwidth makes
//! the *format* the dominant variable: on the reference C922, 1080p is 30fps over MJPG
//! and **5fps** over YUYV, because raw 4:2:2 at that size saturates USB 2. So the ladder
//! prefers compressed capture first and only then falls back to raw, and it retries down
//! through resolutions rather than failing outright — cameras routinely advertise modes
//! they will not actually grant.

use v4l::FourCC;

/// A capture pixel format we know how to handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Motion JPEG. Needs a decode step, but it is the only way to get 1080p30 out of a
    /// USB 2 camera, so it is the *preferred* format rather than a fallback.
    Mjpeg,
    /// Packed 4:2:2, two pixels per four bytes. Universally supported, bandwidth-hungry.
    Yuyv,
    /// Semi-planar 4:2:0. Cheaper than YUYV and directly uploadable, but rare on UVC.
    Nv12,
    /// Planar 4:2:0.
    Yu12,
}

impl PixelFormat {
    pub fn fourcc(self) -> FourCC {
        match self {
            PixelFormat::Mjpeg => FourCC::new(b"MJPG"),
            PixelFormat::Yuyv => FourCC::new(b"YUYV"),
            PixelFormat::Nv12 => FourCC::new(b"NV12"),
            PixelFormat::Yu12 => FourCC::new(b"YU12"),
        }
    }

    pub fn from_fourcc(f: FourCC) -> Option<Self> {
        match &f.repr {
            b"MJPG" | b"JPEG" => Some(PixelFormat::Mjpeg),
            b"YUYV" | b"YUY2" => Some(PixelFormat::Yuyv),
            b"NV12" => Some(PixelFormat::Nv12),
            b"YU12" | b"I420" => Some(PixelFormat::Yu12),
            _ => None,
        }
    }

    /// Lower sorts first. MJPEG wins because it is the only format that reaches 1080p30
    /// over USB 2; NV12 beats YUYV because it is 12bpp rather than 16.
    pub fn preference(self) -> u8 {
        match self {
            PixelFormat::Mjpeg => 0,
            PixelFormat::Nv12 => 1,
            PixelFormat::Yu12 => 2,
            PixelFormat::Yuyv => 3,
        }
    }

    pub fn needs_decode(self) -> bool {
        matches!(self, PixelFormat::Mjpeg)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PixelFormat::Mjpeg => "MJPG",
            PixelFormat::Yuyv => "YUYV",
            PixelFormat::Nv12 => "NV12",
            PixelFormat::Yu12 => "YU12",
        }
    }
}

/// One capture mode: a format at a size and rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureMode {
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl std::fmt::Display for CaptureMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@{}x{}/{}fps",
            self.format.as_str(),
            self.width,
            self.height,
            self.fps
        )
    }
}

/// Build the ordered list of modes to try for a requested size and rate.
///
/// Every candidate is tried in turn because a camera advertising a mode is not a promise
/// it will grant one — phone-as-webcam apps and cheap UVC devices routinely list modes
/// that fail at `STREAMON`. Giving up after the first refusal is how you get "the camera
/// doesn't work" on hardware that works fine one rung down.
pub fn mode_ladder(
    available: &[(PixelFormat, u32, u32)],
    want_w: u32,
    want_h: u32,
    want_fps: u32,
) -> Vec<CaptureMode> {
    let mut out: Vec<CaptureMode> = Vec::new();

    let push = |m: CaptureMode, out: &mut Vec<CaptureMode>| {
        if !out.contains(&m) {
            out.push(m);
        }
    };

    // 1. Exactly what was asked for, best format first.
    let mut exact: Vec<_> = available
        .iter()
        .filter(|(_, w, h)| *w == want_w && *h == want_h)
        .collect();
    exact.sort_by_key(|(f, _, _)| f.preference());
    for (f, w, h) in exact {
        push(
            CaptureMode {
                format: *f,
                width: *w,
                height: *h,
                fps: want_fps,
            },
            &mut out,
        );
    }

    // 2. Same aspect ratio, smaller — a 16:9 request should not silently become 4:3 and
    // hand the user a cropped or letterboxed picture.
    let want_ar = want_w as f32 / want_h.max(1) as f32;
    let mut same_ar: Vec<_> = available
        .iter()
        .filter(|(_, w, h)| {
            let ar = *w as f32 / (*h).max(1) as f32;
            (ar - want_ar).abs() < 0.02 && *w <= want_w
        })
        .collect();
    // Largest first, best format first.
    same_ar.sort_by(|a, b| {
        (b.1 * b.2)
            .cmp(&(a.1 * a.2))
            .then(a.0.preference().cmp(&b.0.preference()))
    });
    for (f, w, h) in same_ar {
        push(
            CaptureMode {
                format: *f,
                width: *w,
                height: *h,
                fps: want_fps,
            },
            &mut out,
        );
    }

    // 3. Known-safe sizes almost every camera supports.
    for (w, h) in [(1280, 720), (640, 480)] {
        let mut safe: Vec<_> = available
            .iter()
            .filter(|(_, aw, ah)| *aw == w && *ah == h)
            .collect();
        safe.sort_by_key(|(f, _, _)| f.preference());
        for (f, aw, ah) in safe {
            push(
                CaptureMode {
                    format: *f,
                    width: *aw,
                    height: *ah,
                    fps: want_fps.min(30),
                },
                &mut out,
            );
        }
    }

    // 4. Anything at all, largest first, rather than failing.
    let mut rest: Vec<_> = available.iter().collect();
    rest.sort_by(|a, b| {
        (b.1 * b.2)
            .cmp(&(a.1 * a.2))
            .then(a.0.preference().cmp(&b.0.preference()))
    });
    for (f, w, h) in rest {
        push(
            CaptureMode {
                format: *f,
                width: *w,
                height: *h,
                fps: want_fps.min(30),
            },
            &mut out,
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C922's real mode list, trimmed.
    fn c922() -> Vec<(PixelFormat, u32, u32)> {
        vec![
            (PixelFormat::Mjpeg, 1920, 1080),
            (PixelFormat::Mjpeg, 1280, 720),
            (PixelFormat::Mjpeg, 640, 480),
            (PixelFormat::Yuyv, 1920, 1080),
            (PixelFormat::Yuyv, 1280, 720),
            (PixelFormat::Yuyv, 640, 480),
        ]
    }

    #[test]
    fn mjpeg_is_preferred_over_yuyv_at_the_same_size() {
        // The single most consequential ordering decision in this file: YUYV at 1080p is
        // 5fps over USB 2, MJPG is 30.
        let ladder = mode_ladder(&c922(), 1920, 1080, 30);
        let first = ladder[0];
        assert_eq!(first.format, PixelFormat::Mjpeg);
        assert_eq!((first.width, first.height), (1920, 1080));
    }

    #[test]
    fn the_exact_request_comes_first() {
        let ladder = mode_ladder(&c922(), 1280, 720, 30);
        assert_eq!((ladder[0].width, ladder[0].height), (1280, 720));
    }

    #[test]
    fn falls_back_through_smaller_sizes_rather_than_failing() {
        // A camera that cannot do 4K must still end up streaming.
        let ladder = mode_ladder(&c922(), 3840, 2160, 30);
        assert!(!ladder.is_empty(), "must always offer something to try");
        assert!(ladder.iter().any(|m| m.width == 1920 && m.height == 1080));
    }

    #[test]
    fn aspect_ratio_is_preserved_before_falling_back() {
        // 16:9 requested: the 16:9 options must be tried before any 4:3 one, or the
        // user silently gets a differently-framed picture.
        let avail = vec![
            (PixelFormat::Mjpeg, 1280, 720),
            (PixelFormat::Mjpeg, 640, 480),
        ];
        let ladder = mode_ladder(&avail, 1920, 1080, 30);
        let first_169 = ladder.iter().position(|m| m.width == 1280).unwrap();
        let first_43 = ladder.iter().position(|m| m.width == 640).unwrap();
        assert!(first_169 < first_43, "16:9 must be preferred over 4:3");
    }

    #[test]
    fn ladder_has_no_duplicates() {
        // Retrying the identical mode wastes a STREAMON round-trip per duplicate, and
        // the fallback path is already the slow path.
        let ladder = mode_ladder(&c922(), 1920, 1080, 30);
        let mut seen = ladder.clone();
        seen.dedup();
        assert_eq!(seen.len(), ladder.len());
    }

    #[test]
    fn an_empty_camera_yields_an_empty_ladder_rather_than_panicking() {
        assert!(mode_ladder(&[], 1920, 1080, 30).is_empty());
    }

    #[test]
    fn fourcc_round_trips_and_accepts_aliases() {
        for f in [
            PixelFormat::Mjpeg,
            PixelFormat::Yuyv,
            PixelFormat::Nv12,
            PixelFormat::Yu12,
        ] {
            assert_eq!(PixelFormat::from_fourcc(f.fourcc()), Some(f));
        }
        // Drivers disagree on spelling; both must map to the same thing.
        assert_eq!(
            PixelFormat::from_fourcc(FourCC::new(b"YUY2")),
            Some(PixelFormat::Yuyv)
        );
        assert_eq!(
            PixelFormat::from_fourcc(FourCC::new(b"I420")),
            Some(PixelFormat::Yu12)
        );
    }

    #[test]
    fn only_mjpeg_needs_decoding() {
        assert!(PixelFormat::Mjpeg.needs_decode());
        assert!(!PixelFormat::Yuyv.needs_decode());
        assert!(!PixelFormat::Nv12.needs_decode());
    }
}
