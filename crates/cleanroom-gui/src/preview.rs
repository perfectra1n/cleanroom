//! Live preview, by consuming the virtual camera like any other app.
//!
//! The GUI opens `/dev/video10` exactly the way Zoom or Chrome would, which makes the
//! preview WYSIWYG-correct by construction rather than by careful reimplementation: what
//! is on screen is literally what a meeting app receives, including the composite, the
//! matte and the mirror. A separate preview path fed from inside the daemon would be
//! lower-latency and would also be a second renderer that could disagree with the real one.
//!
//! Two consequences worth being explicit about, because both are visible to the user:
//!
//! * An open preview is a **real consumer**, so it wakes the camera out of power save. That
//!   is arguably correct — you are looking at the picture — but it does mean the LED comes
//!   on when the window opens.
//! * The daemon's consumer count includes us, so it will read one higher than the number of
//!   meeting apps.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Preview width. The window is ~560 px wide, so anything beyond this is thrown away by
/// the scaler — and the conversion below is per-pixel CPU work, which at 1920x1080 would
/// be 2 Mpx a frame for a picture displayed at a fraction of that.
const PREVIEW_W: u32 = 480;
const PREVIEW_H: u32 = 270;

pub struct Preview {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Preview {
    /// Start reading `path`, delivering frames through `on_frame` on the Slint event loop.
    pub fn start(
        path: String,
        on_frame: impl Fn(Vec<u8>, u32, u32) + Send + Sync + 'static,
    ) -> Self {
        // Arc rather than a borrow: each frame hands the callback to the Slint event loop,
        // which runs it on the UI thread at a time of its choosing, so it has to be owned
        // and shareable rather than referenced from this thread's stack.
        let on_frame = Arc::new(on_frame);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let handle = std::thread::Builder::new()
            .name("cleanroom-preview".into())
            .spawn(move || run(path, stop_thread, on_frame))
            .expect("spawning the preview thread");
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Raw RGB8 plus dimensions, not a `slint::Image`.
///
/// `slint::Image` is neither Send nor Sync — it is reference-counted against the UI thread
/// — so it cannot cross this boundary. The pixels do, and the image is constructed inside
/// the event-loop closure where it belongs.
type FrameSink = Arc<dyn Fn(Vec<u8>, u32, u32) + Send + Sync>;

fn run(path: String, stop: Arc<AtomicBool>, on_frame: FrameSink) {
    // Reconnect rather than give up. The device disappears whenever the daemon restarts
    // its pipeline — a resolution change, a resume from suspend — and a preview that goes
    // permanently black after an unrelated setting change would look like a bug in the
    // setting.
    while !stop.load(Ordering::Relaxed) {
        match stream_once(&path, &stop, &on_frame) {
            Ok(()) => return,
            Err(e) => {
                tracing::debug!(error = %e, "preview stream ended; retrying");
                for _ in 0..10 {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            }
        }
    }
}

fn stream_once(
    path: &str,
    stop: &AtomicBool,
    on_frame: &FrameSink,
) -> Result<(), Box<dyn std::error::Error>> {
    // Ask for the preview size and take whatever the driver grants. v4l2loopback serves
    // the producer's geometry regardless of what a consumer requests, and `Camera::open`
    // already reads back what it actually got rather than trusting the request.
    let mut cam = cleanroom_video::Camera::open(path, PREVIEW_W, PREVIEW_H, 30)?;
    let mode = cam.mode();
    cam.start()?;

    let mut rgb = vec![0u8; (PREVIEW_W * PREVIEW_H * 3) as usize];

    while !stop.load(Ordering::Relaxed) {
        let frame = cam.next_frame()?;
        if frame.format != cleanroom_video::PixelFormat::Yuyv {
            // The daemon publishes YUY2 and nothing else does. Anything here means we are
            // reading a device we did not expect, and guessing at the layout would draw
            // garbage rather than fail.
            return Err(format!("preview expected YUY2, got {:?}", frame.format).into());
        }

        yuy2_to_rgb_scaled(
            frame.data,
            mode.width,
            mode.height,
            &mut rgb,
            PREVIEW_W,
            PREVIEW_H,
        );

        // The only safe way to reach UI state from here. If the event loop has already
        // gone, this returns an error and the frame is simply dropped.
        let cb = on_frame.clone();
        let pixels = rgb.clone();
        let _ = slint::invoke_from_event_loop(move || cb(pixels, PREVIEW_W, PREVIEW_H));
    }
    Ok(())
}

/// YUY2 to RGB with nearest-neighbour downscale, in one pass.
///
/// Deliberately not two passes. Converting 1920x1080 and *then* scaling would touch 2 Mpx
/// per frame for a picture shown at 480x270; sampling only the pixels that survive means
/// 130k. Nearest-neighbour is fine at this size — the alternative is visible only if you
/// are looking for it, and this runs 30 times a second on the UI process.
fn yuy2_to_rgb_scaled(src: &[u8], sw: u32, sh: u32, dst: &mut [u8], dw: u32, dh: u32) {
    if sw == 0 || sh == 0 {
        return;
    }
    let src_stride = sw as usize * 2;

    for y in 0..dh as usize {
        let sy = y * sh as usize / dh as usize;
        let row = sy * src_stride;
        for x in 0..dw as usize {
            let sx = x * sw as usize / dw as usize;
            // YUY2 packs two pixels per four bytes as Y0 U Y1 V, so the chroma pair is
            // shared and the luma index depends on which half of the macropixel we are in.
            let pair = (sx & !1) * 2;
            let i = row + pair;
            if i + 3 >= src.len() {
                continue;
            }
            let y_val = if sx & 1 == 0 { src[i] } else { src[i + 2] } as f32;
            let u = src[i + 1] as f32 - 128.0;
            let v = src[i + 3] as f32 - 128.0;

            // BT.601 limited range, matching what the pipeline encodes.
            let c = y_val - 16.0;
            let r = 1.164 * c + 1.596 * v;
            let g = 1.164 * c - 0.813 * v - 0.391 * u;
            let b = 1.164 * c + 2.017 * u;

            let o = (y * dw as usize + x) * 3;
            dst[o] = r.clamp(0.0, 255.0) as u8;
            dst[o + 1] = g.clamp(0.0, 255.0) as u8;
            dst[o + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grey in, grey out. This catches the classic YUY2 mistake of reading the chroma
    /// bytes as luma, which produces a picture that is recognisable but wrongly coloured —
    /// exactly the kind of bug that survives a glance at a preview.
    #[test]
    fn a_neutral_grey_frame_converts_to_neutral_grey() {
        let (sw, sh) = (16u32, 8u32);
        // Y=128, U=V=128 is mid grey with no chroma cast.
        let src = vec![128u8; (sw * sh * 2) as usize];
        let (dw, dh) = (8u32, 4u32);
        let mut dst = vec![0u8; (dw * dh * 3) as usize];
        yuy2_to_rgb_scaled(&src, sw, sh, &mut dst, dw, dh);

        for px in dst.chunks_exact(3) {
            let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
            assert!(
                (r - g).abs() <= 2 && (g - b).abs() <= 2,
                "expected neutral grey, got {r},{g},{b}"
            );
            assert!((100..=160).contains(&r), "expected mid grey, got {r}");
        }
    }

    /// All-zero YUY2 is mid-green through BT.601, not black. Asserted here so the preview
    /// agrees with the rest of the pipeline about what a zeroed buffer looks like, rather
    /// than quietly clamping it into something else.
    #[test]
    fn an_all_zero_frame_reads_as_green_like_everywhere_else() {
        let src = vec![0u8; 16 * 8 * 2];
        let mut dst = vec![0u8; 8 * 4 * 3];
        yuy2_to_rgb_scaled(&src, 16, 8, &mut dst, 8, 4);
        assert_eq!(dst[0], 0, "red clamps to 0");
        assert!(dst[1] > 100 && dst[1] < 180, "green ~135, got {}", dst[1]);
        assert_eq!(dst[2], 0, "blue clamps to 0");
    }

    /// A short or mis-sized buffer must not panic: the source is a device that can hand
    /// back a partial frame while a format change is in flight.
    #[test]
    fn a_truncated_frame_does_not_panic() {
        let src = vec![128u8; 10];
        let mut dst = vec![0u8; 8 * 4 * 3];
        yuy2_to_rgb_scaled(&src, 16, 8, &mut dst, 8, 4);
    }

    #[test]
    fn a_zero_sized_source_does_not_panic() {
        let mut dst = vec![0u8; 8 * 4 * 3];
        yuy2_to_rgb_scaled(&[], 0, 0, &mut dst, 8, 4);
    }
}
