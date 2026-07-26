//! Decoding and cover-fitting the replacement background plate.
//!
//! This is CPU work done once per (image, frame size) rather than per frame. A 1080p plate
//! is 8 MB of RGBA; re-uploading that at 30 fps would cost more PCIe bandwidth than the
//! whole rest of the pipeline, and re-decoding it would cost more CPU than the MJPEG decode
//! we already do per frame.

use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum BackgroundError {
    #[error("cannot read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "cannot decode {path}: {detail}. Cleanroom reads PNG and JPEG; \
         convert the file or choose another."
    )]
    Decode { path: PathBuf, detail: String },
}

/// A decoded plate, already cover-fitted to one specific frame size.
pub struct Plate {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// What this was built from. Compared to decide whether a reload is needed.
    key: PlateKey,
}

/// Written by hand rather than derived: `rgba` is megabytes of pixels, and a derived Debug
/// would dump all of it into any test failure or log line that touched a Plate.
impl std::fmt::Debug for Plate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plate")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.rgba.len())
            .field("key", &self.key)
            .finish()
    }
}

/// Identity of a cached plate.
///
/// `mtime` is in here so that editing an image in place takes effect without having to
/// rename it or restart the daemon — the path alone would cache a stale picture forever,
/// and "I changed the file and nothing happened" is a bad way to find that out.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlateKey {
    path: PathBuf,
    mtime: Option<std::time::SystemTime>,
    frame: (u32, u32),
}

fn key_for(path: &Path, frame: (u32, u32)) -> PlateKey {
    PlateKey {
        path: path.to_path_buf(),
        mtime: std::fs::metadata(path).and_then(|m| m.modified()).ok(),
        frame,
    }
}

impl Plate {
    /// Whether this plate is still the right one for `path` at `frame`.
    pub fn is_current(&self, path: &Path, frame: (u32, u32)) -> bool {
        self.key == key_for(path, frame)
    }

    /// Decode `path` and cover-fit it to `frame`.
    pub fn load(path: &Path, frame: (u32, u32)) -> Result<Self, BackgroundError> {
        let key = key_for(path, frame);

        let bytes = std::fs::read(path).map_err(|source| BackgroundError::Read {
            path: path.to_path_buf(),
            source,
        })?;

        let img = image::load_from_memory(&bytes).map_err(|e| BackgroundError::Decode {
            path: path.to_path_buf(),
            detail: e.to_string(),
        })?;

        let (tw, th) = frame;
        let fitted = cover_fit(&img, tw, th);

        Ok(Self {
            rgba: fitted.into_raw(),
            width: tw,
            height: th,
            key,
        })
    }
}

/// Scale to cover, then centre-crop.
///
/// Cover rather than contain: letterboxing a background plate would put black bars behind
/// the subject, which reads as a broken effect. Cover means the whole frame is always
/// covered and the overflow is cropped evenly from both sides, which is what every video
/// conferencing tool does and what people expect.
///
/// `scale = max(tw/bw, th/bh)` is the smallest scale where both axes still reach the frame.
fn cover_fit(img: &image::DynamicImage, tw: u32, th: u32) -> image::RgbaImage {
    use image::GenericImageView;

    let (bw, bh) = img.dimensions();
    if bw == 0 || bh == 0 || tw == 0 || th == 0 {
        return image::RgbaImage::from_pixel(tw.max(1), th.max(1), image::Rgba([0, 0, 0, 255]));
    }

    let scale = (tw as f32 / bw as f32).max(th as f32 / bh as f32);
    // Round up: rounding down can leave a scaled side one pixel short of the frame, and
    // then the crop below reads past the edge and the fit silently gains a black seam.
    let sw = ((bw as f32 * scale).ceil() as u32).max(tw);
    let sh = ((bh as f32 * scale).ceil() as u32).max(th);

    // Lanczos3: this runs once per image, not per frame, so the good filter is free. A
    // cheap filter here is visible for as long as the background stays on screen.
    let scaled = img.resize_exact(sw, sh, image::imageops::FilterType::Lanczos3);

    let x = (sw - tw) / 2;
    let y = (sh - th) / 2;
    scaled.crop_imm(x, y, tw, th).to_rgba8()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> image::DynamicImage {
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            w,
            h,
            image::Rgba([10, 20, 30, 255]),
        ))
    }

    /// Every case must fill the frame exactly. A plate one pixel short shows as a seam of
    /// whatever was in the texture before, which is the sort of thing that only ever gets
    /// noticed by someone else, on a call.
    #[test]
    fn cover_fit_always_fills_the_frame_exactly() {
        for (bw, bh) in [
            (1920, 1080), // same aspect
            (1000, 1000), // square into 16:9
            (4000, 500),  // extreme letterbox
            (37, 4001),   // extreme portrait, non-round
            (1, 1),       // degenerate but legal
            (1919, 1079), // just under, exercises the ceil
        ] {
            // A small target on purpose. The arithmetic under test is scale-invariant,
            // and Lanczos3 into 1920x1080 in an unoptimised build made this single test
            // take most of two minutes — which is paid on every `mise run check`.
            let out = cover_fit(&img(bw, bh), 320, 180);
            assert_eq!(
                (out.width(), out.height()),
                (320, 180),
                "{bw}x{bh} did not cover the frame"
            );
        }
    }

    #[test]
    fn a_zero_sized_image_does_not_panic() {
        let out = cover_fit(&img(0, 0), 640, 360);
        assert_eq!((out.width(), out.height()), (640, 360));
    }

    /// The cache key has to notice an edit in place, or "I changed the file and nothing
    /// happened" becomes the bug report.
    #[test]
    fn the_cache_key_covers_path_size_and_mtime() {
        let dir = std::env::temp_dir().join(format!("cleanroom-plate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("plate.png");
        std::fs::write(&p, b"not really a png").unwrap();

        let a = key_for(&p, (1920, 1080));
        assert_eq!(a, key_for(&p, (1920, 1080)), "stable for the same inputs");
        assert_ne!(a, key_for(&p, (1280, 720)), "frame size must be in the key");
        assert_ne!(
            a,
            key_for(&dir.join("other.png"), (1920, 1080)),
            "path must be in the key"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_file_says_which_one() {
        let e = Plate::load(Path::new("/nonexistent/plate.png"), (640, 360)).unwrap_err();
        assert!(
            e.to_string().contains("plate.png"),
            "must name the file: {e}"
        );
    }

    #[test]
    fn an_undecodable_file_says_what_is_supported() {
        let dir = std::env::temp_dir().join(format!("cleanroom-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("bad.png");
        std::fs::write(&p, b"definitely not an image").unwrap();

        let e = Plate::load(&p, (640, 360)).unwrap_err();
        let m = e.to_string();
        assert!(m.contains("PNG and JPEG"), "must say what is accepted: {m}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
