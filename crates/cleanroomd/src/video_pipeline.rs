//! The video pipeline thread.
//!
//! Runs on a dedicated OS thread rather than a tokio task because V4L2 is blocking I/O
//! with a hard cadence: `DQBUF` parks until the next frame arrives, and doing that on an
//! async runtime's worker would stall every other task on it for up to a frame interval.
//!
//! It communicates with the async side only through [`Shared`] and an atomic stop flag,
//! so there is no channel to back up and no way for a slow consumer to apply
//! backpressure to the camera.

use crate::state::{HealthState, Shared};
use cleanroom_gpu::{FramePipeline, Gpu};
use cleanroom_ipc::PipelineStats;
use cleanroom_matting::{INFER_H, INFER_W, Matter};
use cleanroom_video::{Camera, ConsumerWatch, FrameDecoder, LoopbackSink, Yuy2Frame};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How long to wait on consumer events per iteration while idle.
///
/// Long enough that an idle daemon is not spinning, short enough that a meeting app
/// opening the camera sees video within a frame or two rather than after a visible pause.
const IDLE_POLL: Duration = Duration::from_millis(250);

pub struct VideoPipeline {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VideoPipeline {
    /// Start the pipeline. Returns immediately; failures are reported through
    /// [`Shared`]'s health rather than by refusing to start, so the daemon stays up and
    /// can explain itself over D-Bus.
    pub fn start(shared: Arc<Shared>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let handle = std::thread::Builder::new()
            .name("cleanroom-video".into())
            .spawn(move || run(shared, stop_thread))
            .expect("spawning the video thread");
        Self {
            stop,
            handle: Some(handle),
        }
    }

    /// Ask the thread to stop and wait for it.
    ///
    /// Joining rather than detaching matters: the thread owns the camera and the
    /// loopback fd, and letting the process exit while it still holds them is what turns
    /// a clean shutdown into a device left in a half-streaming state.
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for VideoPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run(shared: Arc<Shared>, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Relaxed) {
        match run_once(&shared, &stop) {
            Ok(()) => return, // asked to stop
            Err(e) => {
                shared.set_video_health(HealthState::failed(e.to_string()));
                // Back off before retrying. A camera that was unplugged, or a loopback
                // device another app grabbed first, will usually come back — retrying
                // forever at full speed would just spam the log.
                for _ in 0..20 {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(250));
                }
            }
        }
    }
}

fn run_once(shared: &Arc<Shared>, stop: &AtomicBool) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = shared.config();

    if !cfg.video.enabled {
        shared.set_video_health(HealthState::idle("video disabled in config"));
        while !stop.load(Ordering::Relaxed) && !shared.config().video.enabled {
            std::thread::sleep(IDLE_POLL);
        }
        return Ok(());
    }

    // --- open the camera -----------------------------------------------------------
    let cam_path = match cfg.video.device.clone() {
        Some(p) => p,
        None => cleanroom_video::capture_devices()
            .first()
            .map(|d| d.path.display().to_string())
            .ok_or("no usable camera found")?,
    };

    let mut cam = Camera::open(&cam_path, cfg.video.width, cfg.video.height, cfg.video.fps)?;
    let mode = cam.mode();

    // --- open the virtual camera ----------------------------------------------------
    let sink_dev = cleanroom_video::select_device(&cfg.video.card_label)?;
    let mut sink = LoopbackSink::open(&sink_dev, mode.width, mode.height, mode.fps)?;
    shared.set_vcam_path(sink.path.clone());

    let mut watch = ConsumerWatch::open(&sink_dev.path)?;
    let mut decoder = FrameDecoder::new(mode.width, mode.height)?;
    let mut frame = Yuy2Frame::new(mode.width, mode.height);

    // The GPU. A failure here is reported and the pipeline continues as a CPU
    // passthrough — the *only* place a CPU path is acceptable, because a camera that
    // still works without effects beats a camera that does not work. It is reported as
    // Degraded rather than Nominal so it can never be mistaken for the real thing.
    let gpu = match Gpu::new(cfg.gpu.render_node.as_deref()) {
        Ok(g) => {
            let name = g.choice.to_string();
            shared.set_gpu_adapter(name);
            Some(FramePipeline::new(g, mode.width, mode.height))
        }
        Err(e) => {
            shared.set_gpu_adapter(format!("unavailable: {e}"));
            shared.set_video_health(HealthState::degraded(format!(
                "no GPU — passing the camera through unmodified: {e}"
            )));
            None
        }
    };
    let mut gpu = gpu;
    let mut processed = vec![0u8; (mode.width * mode.height * 2) as usize];

    // Matting. Only loaded when a background effect actually needs a matte — with the mode
    // Off there is nothing to segment, and loading a model to compute an unused alpha would
    // be pure waste.
    //
    // A missing model is reported as Degraded rather than failing: blur without a matte is
    // still a working camera, it just has nothing to separate foreground from background.
    // What it must never be is *silently* the same as having one.
    let mut matter = None;
    let mut matte_rgba = Vec::new();
    if gpu.is_some() && cfg.video.background != cleanroom_core::BackgroundMode::Off {
        match cleanroom_matting::find_model().and_then(|m| Matter::new(&m)) {
            Ok(m) => {
                if let Some(pipe) = gpu.as_mut() {
                    pipe.enable_matte_input(INFER_W, INFER_H);
                }
                matte_rgba = vec![0u8; (INFER_W * INFER_H * 4) as usize];
                matter = Some(m);
            }
            Err(e) => {
                shared.set_video_health(HealthState::degraded(format!(
                    "background effects have no subject to separate — matting model \
                     unavailable: {e}"
                )));
            }
        }
    }

    // Report the mode we actually got, not the one that was asked for.
    shared.set_video_health(HealthState::nominal(format!(
        "{} -> {} ({})",
        cam_path, sink.path, mode
    )));

    let mut stats = StatsAccumulator::new();
    let mut capturing = true;

    while !stop.load(Ordering::Relaxed) {
        // A config change that alters the negotiated mode needs a full restart. Cheap to
        // check, and it keeps `set video.width` working without a daemon restart.
        let now_cfg = shared.config();
        if !now_cfg.video.enabled
            || now_cfg.video.device.as_deref().unwrap_or(&cam_path) != cam_path
            || (now_cfg.video.width, now_cfg.video.height, now_cfg.video.fps)
                != (cfg.video.width, cfg.video.height, cfg.video.fps)
            // Switching to or from Off changes whether a matting model is needed at all.
            || (now_cfg.video.background == cleanroom_core::BackgroundMode::Off)
                != (cfg.video.background == cleanroom_core::BackgroundMode::Off)
        {
            tracing::info!("video config changed; restarting pipeline");
            return Ok(());
        }

        let consumers = watch.poll(Duration::from_millis(0));
        let wanted = !now_cfg.video.power_save || watch.in_use();

        if wanted && !capturing {
            // Someone opened the camera. Resume before they notice.
            cam.start()?;
            capturing = true;
            shared.set_video_health(HealthState::nominal(format!(
                "{} -> {} ({})",
                cam_path, sink.path, mode
            )));
        } else if !wanted && capturing {
            // Nothing is watching. Stop capture: LED off, no USB traffic, no decode.
            // The loopback fd stays open, so the device does not disappear from any app
            // that already enumerated it — they only look once, at startup.
            cam.stop();
            capturing = false;
            shared.set_video_health(HealthState::idle(
                "no consumers; camera released (virtual camera still present)",
            ));
        }

        if !capturing {
            watch.poll(IDLE_POLL);
            stats.publish_if_due(shared, consumers, true);
            continue;
        }

        let raw = match cam.next_frame() {
            Ok(f) => f,
            Err(e) => {
                // A read failure mid-stream is usually an unplug. Surface it and let the
                // outer loop reopen rather than spinning on a dead fd.
                return Err(Box::new(e));
            }
        };

        let t0 = Instant::now();
        decoder.to_yuy2(raw.data, raw.format, raw.width, raw.height, &mut frame)?;
        let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Effects. Read the live config each frame so a blur-strength change takes effect
        // on the very next frame rather than at the next restart.
        let mut matting_ms = 0.0;
        let gpu_ms = match gpu.as_mut() {
            Some(pipe) => {
                let t1 = Instant::now();
                pipe.process(
                    &frame.data,
                    &mut processed,
                    now_cfg.video.background,
                    now_cfg.video.blur_strength,
                    now_cfg.video.mirror,
                );
                // Matting runs *after* the composite, so its matte applies to the next
                // frame rather than this one. One frame of latency is invisible, and it
                // avoids a pipeline stall: the alternative is to downscale, read back, run
                // the network and only then composite, serialising the GPU behind the CPU
                // on every single frame.
                if let Some(m) = matter.as_mut() {
                    let t2 = Instant::now();
                    if pipe.read_matte_input(&mut matte_rgba) {
                        match m.infer(&matte_rgba) {
                            Ok(alpha) => pipe.set_matte(alpha, INFER_W, INFER_H),
                            Err(e) => {
                                tracing::warn!(error = %e, "matting failed for one frame");
                            }
                        }
                    }
                    matting_ms = t2.elapsed().as_secs_f64() * 1000.0;
                }

                sink.write(&processed)?;
                t1.elapsed().as_secs_f64() * 1000.0
            }
            None => {
                sink.write(&frame.data)?;
                0.0
            }
        };

        stats.record(decode_ms, gpu_ms, matting_ms);
        stats.publish_if_due(shared, consumers, false);
    }

    Ok(())
}

/// Accumulates per-second telemetry.
struct StatsAccumulator {
    frames: u64,
    decode_ms_total: f64,
    gpu_ms_total: f64,
    matting_ms_total: f64,
    since: Instant,
}

impl StatsAccumulator {
    fn new() -> Self {
        Self {
            frames: 0,
            decode_ms_total: 0.0,
            gpu_ms_total: 0.0,
            matting_ms_total: 0.0,
            since: Instant::now(),
        }
    }

    fn record(&mut self, decode_ms: f64, gpu_ms: f64, matting_ms: f64) {
        self.frames += 1;
        self.decode_ms_total += decode_ms;
        self.gpu_ms_total += gpu_ms;
        self.matting_ms_total += matting_ms;
    }

    fn publish_if_due(&mut self, shared: &Arc<Shared>, consumers: Option<u32>, idle: bool) {
        let elapsed = self.since.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }

        let mut s = PipelineStats {
            // An unknown consumer count is reported as 0 here only for display; the
            // *decision* to keep capturing uses `in_use()`, which treats unknown as busy.
            vcam_consumers: consumers.unwrap_or(0),
            ..Default::default()
        };
        if !idle && self.frames > 0 {
            s.fps = self.frames as f64 / elapsed.as_secs_f64();
            // Decode is CPU work and reported as such. gpu_ms and matting_ms stay zero
            // until there is actually a GPU stage — reporting a number there before then
            // would be inventing data.
            s.decode_ms = self.decode_ms_total / self.frames as f64;
            s.gpu_ms = self.gpu_ms_total / self.frames as f64;
            s.matting_ms = self.matting_ms_total / self.frames as f64;
        }
        shared.set_stats(s);

        self.frames = 0;
        self.decode_ms_total = 0.0;
        self.gpu_ms_total = 0.0;
        self.matting_ms_total = 0.0;
        self.since = Instant::now();
    }
}
