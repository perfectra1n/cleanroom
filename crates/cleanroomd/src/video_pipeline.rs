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
use cleanroom_core::Config;
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

/// Consecutive matting failures before the recurrent state is cleared.
///
/// Small on purpose. The state is cheap to rebuild — RVM re-shapes it on the next frame —
/// so there is little reason to persevere with one that is producing errors.
const MATTE_ERROR_RESET: u32 = 30;

pub struct VideoPipeline {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// Why [`run_once`] returned.
///
/// This exists because the two reasons are indistinguishable as `Ok(())`, and getting them
/// the wrong way round is silent and fatal: `run_once` returned `Ok(())` both when asked to
/// stop *and* when a config change needed the devices reopened, and the caller treated
/// every `Ok` as "stop". So a single `cleanroom-ctl set video.width 1280` ended the video
/// thread for good — the camera and loopback fds were dropped, the v4l2loopback node
/// reverted to output-only ("Not a video capture device"), and every app lost the camera.
///
/// Nothing reported it, either: health is only written *by* that thread, so the daemon went
/// on serving the last value it had published, "no consumers; camera released (virtual
/// camera still present)". Plausible, reassuring and false.
///
/// Making the two cases separate variants means the compiler now asks the question.
#[must_use]
enum Outcome {
    /// The stop flag was set. The thread should end.
    Stopped,
    /// Something changed that needs the devices reopened. Re-enter `run_once`.
    Restart,
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
    // The Matter outlives every restart of the pipeline below, and that is deliberate.
    //
    // `impl Drop for Matter` leaks its ONNX session on purpose, because dropping one that
    // owns a Dawn context segfaults. That is a fine trade for one session per *process* —
    // the OS reclaims it at exit — but it was previously a local inside `run_once`, so
    // every restart leaked a whole Dawn context plus RVM's weights. `set video.width`
    // twice and you have leaked twice. Restarts here are routine: a config change, a
    // camera unplug, an Off/Blur toggle.
    //
    // Keeping it out here makes the leak per-process again, and makes `Matter::reset()`
    // load-bearing rather than decorative: the recurrent state now genuinely survives a
    // restart, so somebody has to say when it is no longer meaningful.
    let mut matter: Option<Matter> = None;

    while !stop.load(Ordering::Relaxed) {
        match run_once(&shared, &stop, &mut matter) {
            Ok(Outcome::Stopped) => return,
            Ok(Outcome::Restart) => continue,
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

fn run_once(
    shared: &Arc<Shared>,
    stop: &AtomicBool,
    matter: &mut Option<Matter>,
) -> Result<Outcome, Box<dyn std::error::Error>> {
    let cfg = shared.config();

    if !cfg.video.enabled {
        shared.set_video_health(HealthState::idle("video disabled in config"));
        while !stop.load(Ordering::Relaxed) && !shared.config().video.enabled {
            std::thread::sleep(IDLE_POLL);
        }
        // Either video was re-enabled — in which case the devices need opening — or we are
        // stopping, which the caller's own loop condition catches.
        return Ok(Outcome::Restart);
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

    // --- the second transport -------------------------------------------------------
    //
    // A separate thread with its own PipeWire main loop, because this one is about to
    // block in V4L2 DQBUF and cannot host one. The frame slot is the whole interface
    // between them: newest-wins, so a slow consumer costs latency on its own side only.
    //
    // A failure here is Degraded, never fatal. The loopback device is what most apps use;
    // losing the portal transport is a real reduction in coverage and must be reported,
    // but it is not a reason to have no camera at all.
    let pw_source = if cfg.video.pipewire_source {
        let s = PwSourceThread::start(mode.width, mode.height, mode.fps);
        shared.set_pw_node(cleanroom_core::node::VIRTUAL_CAM_NODE);
        Some(s)
    } else {
        shared.set_pw_node("");
        None
    };

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

    // Matting. Only *run* when a background effect actually needs a matte — with the mode
    // Off there is nothing to segment, and computing an unused alpha would be pure waste.
    //
    // A missing model is reported as Degraded rather than failing: blur without a matte is
    // still a working camera, it just has nothing to separate foreground from background.
    // What it must never be is *silently* the same as having one.
    let mut matte_rgba = Vec::new();
    let want_matte = gpu.is_some() && cfg.video.background != cleanroom_core::BackgroundMode::Off;

    // Note what does *not* happen when a matte is not wanted: the Matter is not dropped.
    //
    // Dropping one reclaims nothing — `Drop for Matter` forgets its session on purpose, so
    // the ONNX and Dawn resources stay allocated either way — while costing a full model
    // load and a second Dawn context to come back. Holding it is therefore both cheaper and
    // simpler, and makes an Off -> Blur toggle instant rather than a reload.
    if want_matte {
        if matter.is_none() {
            match cleanroom_matting::find_model().and_then(|m| Matter::new(&m)) {
                Ok(m) => *matter = Some(m),
                Err(e) => {
                    shared.set_video_health(HealthState::degraded(format!(
                        "background effects have no subject to separate — matting model \
                         unavailable: {e}"
                    )));
                }
            }
        }

        if let Some(m) = matter.as_mut() {
            // Unconditional, including on a freshly constructed Matter. Reaching here means
            // the pipeline is (re)starting, so whatever the recurrent state describes — a
            // different resolution, a different camera, or a scene from before an unplug —
            // is not the scene about to arrive. Feeding stale state across a geometry change
            // is a hard shape error; across everything else it is a soft one, a few frames
            // of matte that belong to the previous scene.
            m.reset();
            if let Some(pipe) = gpu.as_mut() {
                pipe.enable_matte_input(INFER_W, INFER_H);
            }
            matte_rgba = vec![0u8; (INFER_W * INFER_H * 4) as usize];
        }
    }

    // Whether to run inference in the loop below. Deliberately not `matter.is_some()`: the
    // Matter is retained across a switch to Off, so its mere presence no longer means a
    // matte is wanted. Gating on the wrong one would spend ~9 ms a frame computing an alpha
    // the composite discards, and would read into a `matte_rgba` that was never sized.
    let matting_active = want_matte && matter.is_some();

    // Report the mode we actually got, not the one that was asked for.
    shared.set_video_health(HealthState::nominal(format!(
        "{} -> {} ({})",
        cam_path, sink.path, mode
    )));

    let mut stats = StatsAccumulator::new();
    let mut capturing = true;
    // Last driver sequence number, for detecting frames the camera produced that we never
    // collected. `None` until the first frame, and reset across a power-save gap, where a
    // jump is expected rather than a drop.
    let mut last_sequence: Option<u32> = None;
    // Consecutive `infer` failures. A single one is a frame not worth crashing over; a run
    // of them means the recurrent state is wedged and needs clearing.
    let mut infer_errors: u32 = 0;
    // The decoded replacement plate, and whether its current failure has been reported.
    // The flag stops a broken path re-logging thirty times a second.
    let mut plate: Option<crate::background::Plate> = None;
    let mut plate_reported: Option<String> = None;

    while !stop.load(Ordering::Relaxed) {
        // A config change that alters the negotiated mode needs a full restart. Cheap to
        // check, and it keeps `set video.width` working without a daemon restart.
        let now_cfg = shared.config();
        if needs_restart(&cfg, &now_cfg, &cam_path) {
            tracing::info!("video config changed; restarting pipeline");
            return Ok(Outcome::Restart);
        }

        // The replacement plate is reloaded, never restarted. Changing the picture does not
        // change the negotiated mode, and a restart would drop the loopback fd and blank
        // the camera for every app in the call — a heavy price for swapping a JPEG.
        let effective_background = match gpu.as_mut() {
            Some(pipe) => sync_background_plate(
                pipe,
                &now_cfg,
                (mode.width, mode.height),
                &mut plate,
                shared,
                &mut plate_reported,
            ),
            None => now_cfg.video.background,
        };

        let consumers = watch.poll(Duration::from_millis(0));
        let wanted = !now_cfg.video.power_save || watch.in_use();

        if wanted && !capturing {
            // Someone opened the camera. Resume before they notice.
            cam.start()?;
            capturing = true;
            // The gap could have been seconds or hours. RVM's recurrent state is a claim
            // about what the previous frame looked like, and across a gap that long it is a
            // claim about a scene that has since been left, relit, or walked out of — so it
            // is worse than no information. The sequence counter jumps across the gap too,
            // and that is not a dropped frame.
            if let Some(m) = matter.as_mut() {
                m.reset();
            }
            last_sequence = None;
            infer_errors = 0;
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

        // Frames the driver produced that we never collected. V4L2's sequence number is
        // the driver's own counter, so a gap is unambiguous — it is the camera outrunning
        // us, not us miscounting. Wrapping is handled by `wrapping_sub`; a genuine u32 wrap
        // at 30 fps takes about four and a half years.
        let dropped_now = u64::from(match last_sequence {
            Some(prev) => raw.sequence.wrapping_sub(prev).saturating_sub(1),
            None => 0,
        });
        last_sequence = Some(raw.sequence);

        let t0 = Instant::now();
        decoder.to_yuy2(raw.data, raw.format, raw.width, raw.height, &mut frame)?;
        let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Effects. Read the live config each frame so a blur-strength change takes effect
        // on the very next frame rather than at the next restart.
        let mut matting_ms = 0.0;
        let mut matte_rejected = 0u64;
        let gpu_ms = match gpu.as_mut() {
            Some(pipe) => {
                let t1 = Instant::now();
                pipe.process(
                    &frame.data,
                    &mut processed,
                    effective_background,
                    now_cfg.video.blur_strength,
                    now_cfg.video.mirror,
                );
                // Matting runs *after* the composite, so its matte applies to the next
                // frame rather than this one. One frame of latency is invisible, and it
                // avoids a pipeline stall: the alternative is to downscale, read back, run
                // the network and only then composite, serialising the GPU behind the CPU
                // on every single frame.
                if let Some(m) = matter.as_mut().filter(|_| matting_active) {
                    let t2 = Instant::now();
                    if pipe.read_matte_input(&mut matte_rgba) {
                        match m.infer(&matte_rgba) {
                            Ok(alpha) => {
                                pipe.set_matte(alpha, INFER_W, INFER_H);
                                infer_errors = 0;
                            }
                            Err(e) => {
                                // One failure is a frame, not an outage — the previous
                                // matte stays up and nobody notices. A *run* of them is
                                // different: the most likely cause is recurrent state the
                                // session will keep choking on, and re-feeding it every
                                // frame forever is how a transient becomes permanent.
                                infer_errors += 1;
                                tracing::warn!(
                                    error = %e,
                                    consecutive = infer_errors,
                                    "matting failed for one frame"
                                );
                                if infer_errors >= MATTE_ERROR_RESET {
                                    tracing::warn!(
                                        "resetting matting state after {infer_errors} \
                                         consecutive failures"
                                    );
                                    m.reset();
                                    infer_errors = 0;
                                }
                            }
                        }
                    }
                    matte_rejected = m.rejected;
                    matting_ms = t2.elapsed().as_secs_f64() * 1000.0;
                }

                sink.write(&processed)?;
                // Same buffer to both transports. Advertising YUY2 on the PipeWire node is
                // what makes that possible — an I420 node would need a conversion here.
                if let Some(pw) = pw_source.as_ref() {
                    pw.slot.put(&processed);
                }
                t1.elapsed().as_secs_f64() * 1000.0
            }
            None => {
                sink.write(&frame.data)?;
                if let Some(pw) = pw_source.as_ref() {
                    pw.slot.put(&frame.data);
                }
                0.0
            }
        };

        stats.record(decode_ms, gpu_ms, matting_ms, dropped_now, matte_rejected);
        stats.publish_if_due(shared, consumers, false);
    }

    // The only way out of that loop is the stop flag.
    Ok(Outcome::Stopped)
}

/// The PipeWire `Video/Source` publisher, on its own thread.
///
/// Owns its stop flag and join handle so `Drop` tears the thread down on every exit path
/// from `run_once`, including the `?` returns. Leaking it across a pipeline restart would
/// leave a second node publishing the old geometry, and PipeWire would happily show a user
/// two cameras with the same name.
struct PwSourceThread {
    slot: Arc<cleanroom_video::FrameSlot>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PwSourceThread {
    fn start(width: u32, height: u32, fps: u32) -> Self {
        let slot = cleanroom_video::FrameSlot::new(width, height, fps);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let slot_thread = slot.clone();
        let handle = std::thread::Builder::new()
            .name("cleanroom-pwcam".into())
            .spawn(move || {
                // Degraded, never fatal: the loopback device is what most apps use, so
                // losing the portal transport reduces coverage but is not a reason to have
                // no camera at all.
                if let Err(e) =
                    cleanroom_video::PwSource::run(slot_thread, "Cleanroom Camera", move || {
                        stop_thread.load(Ordering::Relaxed)
                    })
                {
                    tracing::warn!(error = %e, "PipeWire Video/Source failed");
                }
            })
            .expect("spawning the PipeWire camera thread");
        Self {
            slot,
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for PwSourceThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Keep the GPU's background plate in step with config, and say which mode is really usable.
///
/// Returns the mode to actually composite with. That is not always the configured one:
/// `Replace` with no usable image would key the subject onto flat black, which looks like a
/// broken effect rather than an unset setting. Falling back to `Blur` and reporting Degraded
/// is the project's rule — a fallback nobody is told about is the one failure mode this
/// codebase is most determined not to have.
///
/// Reload, never restart. Swapping the picture does not change the negotiated mode, and a
/// restart would drop the loopback fd and blank the camera for everyone in the call.
fn sync_background_plate(
    pipe: &mut FramePipeline,
    cfg: &Config,
    frame: (u32, u32),
    plate: &mut Option<crate::background::Plate>,
    shared: &Arc<Shared>,
    reported: &mut Option<String>,
) -> cleanroom_core::BackgroundMode {
    use cleanroom_core::BackgroundMode;

    let wanted = cfg.video.background;
    if wanted != BackgroundMode::Replace {
        return wanted;
    }

    let Some(path) = cfg.video.background_image.as_deref() else {
        return degrade(
            shared,
            reported,
            "background replace has no image — set video.background_image to a PNG or JPEG"
                .to_string(),
        );
    };

    // Reload only when the path, its mtime or the frame size has actually moved.
    let stale = plate.as_ref().is_none_or(|p| !p.is_current(path, frame));
    if stale {
        match crate::background::Plate::load(path, frame) {
            Ok(p) => {
                pipe.set_background_image(&p.rgba, p.width, p.height);
                *plate = Some(p);
                if reported.take().is_some() {
                    // Recovered: say so, or the UI keeps showing a stale complaint.
                    shared.set_video_health(HealthState::nominal(format!(
                        "background image loaded: {}",
                        path.display()
                    )));
                }
                tracing::info!(path = %path.display(), "background plate loaded");
            }
            Err(e) => {
                *plate = None;
                pipe.clear_background_image();
                return degrade(shared, reported, e.to_string());
            }
        }
    }

    if pipe.has_background_image() {
        BackgroundMode::Replace
    } else {
        degrade(
            shared,
            reported,
            "background replace has no usable image".to_string(),
        )
    }
}

/// Report a background-plate problem once, then fall back to blur.
///
/// Once, because this is called per frame: without the latch a bad path would write health
/// and log thirty times a second, which buries every other message in the journal.
fn degrade(
    shared: &Arc<Shared>,
    reported: &mut Option<String>,
    detail: String,
) -> cleanroom_core::BackgroundMode {
    if reported.as_deref() != Some(detail.as_str()) {
        tracing::warn!(%detail, "falling back to blur");
        shared.set_video_health(HealthState::degraded(format!(
            "{detail} — blurring instead"
        )));
        *reported = Some(detail);
    }
    cleanroom_core::BackgroundMode::Blur
}

/// Whether a config change needs the devices reopened.
///
/// Split out of the frame loop so the policy can be tested without a camera. The
/// distinction it draws is the whole point: anything affecting the *negotiated mode* — the
/// device, its geometry, whether a matting model is needed at all — cannot be changed on a
/// running stream, while everything else (blur strength, mirror, which non-Off background)
/// is read fresh each frame and must NOT restart, because a restart drops the loopback fd
/// and every consuming app sees the camera vanish.
///
/// `started_with` is the config the current run opened its devices against; `now` is live.
fn needs_restart(started_with: &Config, now: &Config, cam_path: &str) -> bool {
    use cleanroom_core::BackgroundMode::Off;

    // `None` means "first usable camera", which is the one already open, so an unset device
    // is not a change away from it.
    let device_changed = now.video.device.as_deref().unwrap_or(cam_path) != cam_path;

    !now.video.enabled
        || device_changed
        || (now.video.width, now.video.height, now.video.fps)
            != (
                started_with.video.width,
                started_with.video.height,
                started_with.video.fps,
            )
        // Switching to or from Off changes whether a matting model is needed at all.
        || (now.video.background == Off) != (started_with.video.background == Off)
        // Publishing or withdrawing a PipeWire node means starting or stopping a thread
        // that was handed the negotiated geometry at construction.
        || now.video.pipewire_source != started_with.video.pipewire_source
}

/// Accumulates per-second telemetry.
struct StatsAccumulator {
    frames: u64,
    decode_ms_total: f64,
    gpu_ms_total: f64,
    matting_ms_total: f64,
    /// Summed per interval: "frames lost in the last second" is the actionable number.
    dropped: u64,
    /// Carried, not summed. `Matter::rejected` is already cumulative since startup, and
    /// what matters is whether it is climbing, so re-publishing the latest reading keeps
    /// it monotonic across intervals instead of resetting it every second.
    matte_rejected: u64,
    since: Instant,
}

impl StatsAccumulator {
    fn new() -> Self {
        Self {
            frames: 0,
            decode_ms_total: 0.0,
            gpu_ms_total: 0.0,
            matting_ms_total: 0.0,
            dropped: 0,
            matte_rejected: 0,
            since: Instant::now(),
        }
    }

    fn record(
        &mut self,
        decode_ms: f64,
        gpu_ms: f64,
        matting_ms: f64,
        dropped: u64,
        matte_rejected: u64,
    ) {
        self.frames += 1;
        self.decode_ms_total += decode_ms;
        self.gpu_ms_total += gpu_ms;
        self.matting_ms_total += matting_ms;
        self.dropped += dropped;
        self.matte_rejected = matte_rejected;
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
            // Survives an idle interval. This is a since-startup total, so blanking it
            // whenever the camera is released would make a real problem disappear the
            // moment the meeting ended.
            matte_rejected: self.matte_rejected,
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
            s.dropped = self.dropped;
        }
        shared.set_stats(s);

        self.frames = 0;
        self.decode_ms_total = 0.0;
        self.gpu_ms_total = 0.0;
        self.matting_ms_total = 0.0;
        self.dropped = 0;
        self.since = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_core::BackgroundMode;

    const CAM: &str = "/dev/video0";

    /// Guards the class of bug that cost the most time here: a restart that should not
    /// have happened. A restart drops the loopback fd, and because `exclusive_caps=1` makes
    /// the node revert to output-only the moment the producer stops, every app that had the
    /// camera open loses it — mid-call, with no error anywhere.
    #[test]
    fn live_adjustable_settings_never_restart_the_pipeline() {
        let base = Config::default();

        let mut blur = base.clone();
        blur.video.blur_strength = 0.95;
        assert!(!needs_restart(&base, &blur, CAM), "blur strength is live");

        let mut mirror = base.clone();
        mirror.video.mirror = !base.video.mirror;
        assert!(!needs_restart(&base, &mirror, CAM), "mirror is live");

        let mut power = base.clone();
        power.video.power_save = !base.video.power_save;
        assert!(!needs_restart(&base, &power, CAM), "power save is live");

        // Blur -> Replace -> Remove all need the same matting model, so switching between
        // them is a uniform change, not a pipeline change.
        for mode in [BackgroundMode::Replace, BackgroundMode::Remove] {
            let mut m = base.clone();
            m.video.background = mode;
            assert!(
                !needs_restart(&base, &m, CAM),
                "{mode:?} needs no restart when already non-Off"
            );
        }
    }

    #[test]
    fn changes_to_the_negotiated_mode_do_restart() {
        let base = Config::default();

        for (name, mutate) in [
            (
                "width",
                (|c: &mut Config| c.video.width = 1280) as fn(&mut Config),
            ),
            ("height", |c: &mut Config| c.video.height = 720),
            ("fps", |c: &mut Config| c.video.fps = 60),
            ("device", |c: &mut Config| {
                c.video.device = Some("/dev/video9".into())
            }),
            ("enabled", |c: &mut Config| c.video.enabled = false),
            ("background off", |c: &mut Config| {
                c.video.background = BackgroundMode::Off
            }),
        ] {
            let mut changed = base.clone();
            mutate(&mut changed);
            assert!(
                needs_restart(&base, &changed, CAM),
                "changing {name} must reopen the devices"
            );
        }
    }

    /// `video.device = None` means "first usable camera". That is the camera already open,
    /// so it must not read as a change — otherwise an unrelated `set` on any other key
    /// would restart the pipeline forever on a default config.
    #[test]
    fn an_unset_device_is_not_a_change_away_from_the_open_one() {
        let base = Config::default();
        assert_eq!(base.video.device, None, "precondition");
        assert!(!needs_restart(&base, &base, CAM));
    }

    /// The bug this whole `Outcome` type exists to prevent: `run_once` returned `Ok(())`
    /// both for "stop" and for "restart me", the caller read every `Ok` as "stop", and a
    /// single `cleanroom-ctl set video.width` ended the video thread permanently — camera
    /// and loopback fds dropped, `/dev/video10` reverting to "Not a video capture device",
    /// and the daemon still reporting the last health it had published.
    #[test]
    fn restart_and_stop_are_distinguishable() {
        assert!(matches!(Outcome::Restart, Outcome::Restart));
        assert!(matches!(Outcome::Stopped, Outcome::Stopped));
        assert!(
            !matches!(Outcome::Restart, Outcome::Stopped),
            "a restart must never be readable as a stop"
        );
    }
}
