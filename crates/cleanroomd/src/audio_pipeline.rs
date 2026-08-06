//! The audio pipeline thread.
//!
//! Like the video pipeline, this owns an OS thread rather than a tokio task: PipeWire's
//! main loop is not async and expects to drive itself.
//!
//! Audio stays on the CPU deliberately. The prior art measured DeepFilterNet at ~0.6 ms
//! per hop on CPU against ~1.2 ms on CUDA — kernel-launch overhead exceeds the work at a
//! 480-sample hop. That is also what lets the audio and video tracks run concurrently
//! without contending for anything.

use crate::state::{HealthState, Shared};
use cleanroom_audio::{Denoiser, SharedAudio, VirtualMic, to_dbfs};
use cleanroom_core::VIRTUAL_MIC_NODE;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Read-only view of the current mic levels.
pub struct LevelHandle(Arc<SharedAudio>);

impl LevelHandle {
    /// Input and output peak levels in dBFS. The gap between them is the suppression
    /// the user is actually getting, which is the number worth showing.
    pub fn dbfs(&self) -> (f32, f32) {
        let i = self.0.level_in.lock().map(|g| *g).unwrap_or(0.0);
        let o = self.0.level_out.lock().map(|g| *g).unwrap_or(0.0);
        (to_dbfs(i), to_dbfs(o))
    }
}

pub struct AudioPipeline {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    shared_audio: Arc<SharedAudio>,
}

impl AudioPipeline {
    pub fn start(shared: Arc<Shared>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let shared_audio = SharedAudio::new();

        let stop_thread = stop.clone();
        let audio_thread = shared_audio.clone();
        let handle = std::thread::Builder::new()
            .name("cleanroom-audio".into())
            .spawn(move || run(shared, audio_thread, stop_thread))
            .expect("spawning the audio thread");

        Self {
            stop,
            handle: Some(handle),
            shared_audio,
        }
    }

    /// A cheap handle the async side can poll for metering without touching the thread.
    pub fn level_handle(&self) -> LevelHandle {
        LevelHandle(self.shared_audio.clone())
    }

    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for AudioPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Why `run_once` returned without an error.
///
/// The same distinction the video pipeline had to learn the hard way: "asked to stop" and
/// "config changed, come straight back" are different answers, and collapsing them into
/// one return value is how a routine `set audio.device` ends the audio thread for good.
/// Worse, the old shape here reported a device *change* as `Err("config changed")`, so
/// switching microphones published a Failed health state and sat in the failure backoff
/// for five seconds — a dead mic and a red banner for doing exactly what the user asked.
#[must_use]
enum Outcome {
    /// The stop flag was set. The thread should end.
    Stopped,
    /// A setting changed that cannot be applied in place. Re-enter `run_once` now.
    Restart,
}

fn run(shared: Arc<Shared>, audio: Arc<SharedAudio>, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Relaxed) {
        match run_once(&shared, &audio, &stop) {
            Ok(Outcome::Stopped) => return,
            Ok(Outcome::Restart) => continue,
            Err(e) => {
                shared.set_audio_health(HealthState::failed(e.to_string()));
                // Back off before retrying: a PipeWire daemon that is restarting will
                // come back, and hammering it at full speed would just spam the log.
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
    audio: &Arc<SharedAudio>,
    stop: &Arc<AtomicBool>,
) -> Result<Outcome, Box<dyn std::error::Error>> {
    // On a restart this thread is still SCHED_RR from the previous run, and the model
    // load below is exactly the sustained CPU burn RLIMIT_RTTIME exists to kill. Drop
    // back to normal scheduling first; the promotion is re-requested below once the
    // heavy lifting is done. See `demote_current_thread` for the observed failure.
    crate::realtime::demote_current_thread();

    let cfg = shared.config();

    if !cfg.audio.enabled {
        shared.set_audio_health(HealthState::idle("audio disabled in config"));
        while !stop.load(Ordering::Relaxed) && !shared.config().audio.enabled {
            std::thread::sleep(Duration::from_millis(250));
        }
        // Two ways out of that wait, and they must not read the same: leaving because the
        // user re-enabled audio has to restart the pipeline, not end the thread.
        return Ok(if stop.load(Ordering::Relaxed) {
            Outcome::Stopped
        } else {
            Outcome::Restart
        });
    }

    // Load the denoiser up front so a missing model is reported as a clear health state
    // rather than as a mic that mysteriously does nothing. Running without it is a
    // legitimate mode — a passthrough virtual mic is still useful — but it is *reported*
    // rather than silently substituted.
    let denoiser = if cfg.audio.denoise.enabled {
        match cleanroom_audio::find_model().and_then(|m| {
            Denoiser::new(
                &m,
                cfg.audio.denoise.attenuation_db,
                cfg.audio.denoise.post_filter_beta,
            )
        }) {
            Ok(d) => Some(d),
            Err(e) => {
                shared.set_audio_health(HealthState::degraded(format!(
                    "running as passthrough — denoiser unavailable: {e}"
                )));
                None
            }
        }
    } else {
        None
    };

    let denoise_active = denoiser.is_some();
    let target = cfg.audio.device.clone();

    // Nominal both when the denoiser is running and when it is off *by choice* — a
    // passthrough the user asked for is healthy, and leaving the previous health in place
    // here meant a device switch could leave "failed: config changed" on screen while the
    // mic worked fine. The one case that keeps its earlier detail is denoise requested but
    // unavailable, which set Degraded above.
    //
    // Deliberately does NOT include the attenuation value. That is applied live via
    // set_atten_lim without restarting the node, so embedding it here would leave a
    // stale number on screen the moment the user moves the slider — the UI shows the
    // live value next to the slider instead.
    if denoise_active == cfg.audio.denoise.enabled {
        shared.set_audio_health(HealthState::nominal(format!(
            "{} -> {} ({})",
            target
                .as_ref()
                .map(|t| t.as_str().to_string())
                .unwrap_or_else(|| "system default".into()),
            VIRTUAL_MIC_NODE,
            if denoise_active {
                "DeepFilterNet"
            } else {
                "passthrough"
            },
        )));
    }

    // Moved into the process closure. `DfTract`'s thread-safety is not documented, and
    // upstream's own LADSPA plugin deliberately never moves one across a thread boundary
    // — so it is constructed here and used only on the PipeWire thread that owns it.
    let mut denoiser = denoiser;

    // Restart when a setting changes that we cannot apply in place.
    //
    // The attenuation limit deliberately is NOT in this list: it is applied live through
    // `Denoiser::set_attenuation`, so dragging a slider must not interrupt audio. The
    // prior art respawned an entire helper process per slider drag and dropped ~200 ms
    // of microphone audio each time.
    // Apply live-tunable parameters inside the process callback, so moving a slider takes
    // effect on the next hop with no restart and no dropped audio.
    let live = shared.clone();
    let mut applied_atten = cfg.audio.denoise.attenuation_db;
    let mut applied_pf = cfg.audio.denoise.post_filter_beta;

    let stop_check = stop.clone();
    let watch = shared.clone();
    let started_with = (
        cfg.audio.enabled,
        cfg.audio.denoise.enabled,
        cfg.audio.device.clone(),
    );

    // Ask for real-time scheduling before the loop starts, so a slow or absent rtkit costs
    // startup latency rather than a glitch mid-stream. This runs on the audio thread
    // because rtkit promotes a *thread*, not a process.
    crate::realtime::publish(shared, crate::realtime::request_for_current_thread());

    VirtualMic::run(
        audio.clone(),
        target,
        VIRTUAL_MIC_NODE,
        "Cleanroom Microphone",
        move |inp, outp| match denoiser.as_mut() {
            Some(d) => {
                let c = live.config();
                if c.audio.denoise.attenuation_db != applied_atten {
                    applied_atten = c.audio.denoise.attenuation_db;
                    d.set_attenuation(applied_atten);
                }
                if c.audio.denoise.post_filter_beta != applied_pf {
                    applied_pf = c.audio.denoise.post_filter_beta;
                    d.set_post_filter(applied_pf);
                }
                d.process(inp, outp)
            }
            None => outp.copy_from_slice(inp),
        },
        move || {
            if stop_check.load(Ordering::Relaxed) {
                return true;
            }
            let now = watch.config();
            let changed = (
                now.audio.enabled,
                now.audio.denoise.enabled,
                now.audio.device.clone(),
            ) != started_with;
            if changed {
                tracing::info!("audio config changed; restarting the virtual microphone");
            }
            changed
        },
        shared.audio_registry.clone(),
    )?;

    // The main loop only quits for the stop flag or for a config change the watch closure
    // noticed; which one it was decides whether the thread ends or comes straight back.
    Ok(if stop.load(Ordering::Relaxed) {
        Outcome::Stopped
    } else {
        Outcome::Restart
    })
}
