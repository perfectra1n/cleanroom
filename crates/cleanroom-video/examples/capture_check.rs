//! Capture a burst of frames from the first usable camera and report what actually
//! happened — negotiated mode, real frame rate, and whether the driver dropped any.
//!
//! Run: nix develop -c cargo run -p cleanroom-video --example capture_check

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(dev) = cleanroom_video::capture_devices().into_iter().next() else {
        eprintln!("no usable camera found");
        return Ok(());
    };
    println!("opening {} ({})", dev.path.display(), dev.card);

    let mut cam = cleanroom_video::Camera::open(&dev.path.display().to_string(), 1920, 1080, 30)?;
    println!("negotiated: {}", cam.mode());

    const N: usize = 90;
    let mut bytes = 0usize;
    let mut first_seq = None;
    let mut last_seq = 0u32;

    let start = Instant::now();
    for i in 0..N {
        let f = cam.next_frame()?;
        bytes += f.data.len();
        if first_seq.is_none() {
            first_seq = Some(f.sequence);
            println!(
                "first frame: {} bytes, {}x{} {}",
                f.data.len(),
                f.width,
                f.height,
                f.format.as_str()
            );
            // A JPEG must start with SOI. If this is wrong we are handing garbage to the
            // decoder, most likely because bytesused was ignored.
            if f.format.needs_decode() {
                println!(
                    "  JPEG SOI marker: {}",
                    if f.data.starts_with(&[0xFF, 0xD8]) {
                        "present"
                    } else {
                        "MISSING — frame is not a valid JPEG"
                    }
                );
            }
        }
        last_seq = f.sequence;
        if i == N / 2 {
            // Exercise the power-save path mid-stream: stop and resume must not need a
            // renegotiation, and must not lose the device.
            cam.stop();
            assert!(!cam.is_streaming());
            cam.start()?;
        }
    }
    let elapsed = start.elapsed();

    let fps = N as f64 / elapsed.as_secs_f64();
    let delivered = last_seq.saturating_sub(first_seq.unwrap_or(0)) + 1;
    println!(
        "\n{N} frames in {:.2}s = {fps:.1} fps ({:.1} MB, {:.0} KB/frame avg)",
        elapsed.as_secs_f64(),
        bytes as f64 / 1e6,
        bytes as f64 / N as f64 / 1024.0
    );
    println!(
        "driver sequence spanned {delivered} frames -> {} dropped by the driver",
        delivered.saturating_sub(N as u32)
    );
    Ok(())
}
