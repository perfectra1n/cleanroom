//! Live preview, by consuming the daemon's `cleanroom_cam` PipeWire node.
//!
//! The bytes on this node are the *same composited frames* the virtual camera carries — one
//! pipeline, one composite, one matte, one mirror — so the preview stays WYSIWYG by
//! construction rather than by carefully reimplementing the renderer. What changed is only
//! the transport it arrives over.
//!
//! That choice is not cosmetic. v4l2loopback admits exactly **one** streaming capture
//! consumer, so a preview reading the loopback device was a preview that made Chrome, Zoom
//! or anything else find the camera busy. PipeWire imposes no such limit: any number of
//! consumers can link to the same node. The loopback device and its single capture slot are
//! now left entirely to real apps, and this window is not competing for them.
//!
//! Two consequences worth being explicit about, because both are visible to the user:
//!
//! * The preview follows the window. A hidden or closed window is not a consumer of
//!   anything; there is no background stream left running behind it.
//! * Opening the window does wake the camera, through the PipeWire consumer path rather
//!   than through the loopback device.

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
    /// Start reading the PipeWire node `node`, delivering frames through `on_frame` on the
    /// Slint event loop.
    pub fn start(
        node: String,
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
            .spawn(move || run(node, stop_thread, on_frame))
            .expect("spawning the preview thread");
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        // Signal, then join, so the PipeWire stream is disconnected before this returns and
        // the caller can honestly say the preview has stopped. Bounded at roughly 50 ms:
        // that is how often `PwCapture::run` polls the stop flag from inside its main loop.
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

fn run(node: String, stop: Arc<AtomicBool>, on_frame: FrameSink) {
    // Reconnect rather than give up. The node disappears whenever the daemon restarts its
    // pipeline — a resolution change, a resume from suspend — and it may not exist at all
    // yet when the window opens first. A preview that went permanently black after an
    // unrelated setting change would look like a bug in the setting.
    while !stop.load(Ordering::Relaxed) {
        match capture_once(&node, &stop, &on_frame) {
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

/// One capture session, from link-up to the node going away or the stop flag being set.
///
/// `PwCapture::run` blocks on its own PipeWire main loop and owns the thread until then, so
/// everything this function does happens inside the frame callback. The geometry is not
/// requested: it is whatever the daemon negotiated, handed in with every frame, and it can
/// change under us when the daemon is reconfigured — which is exactly why the scaler is
/// given `w` and `h` per frame rather than a size read once at startup.
fn capture_once(
    node: &str,
    stop: &Arc<AtomicBool>,
    on_frame: &FrameSink,
) -> Result<(), cleanroom_video::PwCaptureError> {
    let should_stop = stop.clone();
    let sink = on_frame.clone();
    let mut rgb = vec![0u8; (PREVIEW_W * PREVIEW_H * 3) as usize];

    cleanroom_video::PwCapture::run(
        node,
        move || should_stop.load(Ordering::Relaxed),
        move |yuy2, w, h| {
            yuy2_to_rgb_scaled(yuy2, w, h, &mut rgb, PREVIEW_W, PREVIEW_H);

            // The only safe way to reach UI state from here. If the event loop has already
            // gone, this returns an error and the frame is simply dropped.
            let cb = sink.clone();
            let pixels = rgb.clone();
            let _ = slint::invoke_from_event_loop(move || cb(pixels, PREVIEW_W, PREVIEW_H));
        },
    )
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
