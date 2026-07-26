//! Camera -> decode -> virtual camera, with no effects.
//!
//! This is the plan's gate-after-B4: it proves every hard Linux-ism in the video path
//! (format negotiation, MJPEG decode, exclusive_caps, the S_FMT-before-stream ordering)
//! before any GPU work depends on it. Everything after this gate is shaders.
//!
//!     nix develop -c cargo run --release -p cleanroom-video --example passthrough
//!     ffplay /dev/video10          # in another terminal
//!
//! Ctrl-C to stop.

use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let Some(cam_dev) = cleanroom_video::capture_devices().into_iter().next() else {
        eprintln!("no usable camera found");
        return Ok(());
    };

    let mut cam =
        cleanroom_video::Camera::open(&cam_dev.path.display().to_string(), 1920, 1080, 30)?;
    let mode = cam.mode();

    let sink_dev = cleanroom_video::select_device("Cleanroom Camera")?;
    let mut sink =
        cleanroom_video::LoopbackSink::open(&sink_dev, mode.width, mode.height, mode.fps)?;

    let mut decoder = cleanroom_video::FrameDecoder::new(mode.width, mode.height)?;
    let mut yuy2 = cleanroom_video::Yuy2Frame::new(mode.width, mode.height);

    println!("\n  {} -> {}", cam_dev.path.display(), sink.path);
    println!("  {}\n", mode);
    println!("  watch it with:  ffplay {}\n", sink.path);

    let mut frames = 0u64;
    let mut decode_total = Duration::ZERO;
    let mut report = Instant::now();

    loop {
        let raw = cam.next_frame()?;
        let t0 = Instant::now();
        decoder.to_yuy2(raw.data, raw.format, raw.width, raw.height, &mut yuy2)?;
        decode_total += t0.elapsed();

        sink.write(&yuy2.data)?;
        frames += 1;

        if report.elapsed() >= Duration::from_secs(2) {
            let secs = report.elapsed().as_secs_f64();
            println!(
                "{:.1} fps   decode {:.2} ms/frame",
                frames as f64 / secs,
                decode_total.as_secs_f64() * 1000.0 / frames as f64
            );
            frames = 0;
            decode_total = Duration::ZERO;
            report = Instant::now();
        }
    }
}
