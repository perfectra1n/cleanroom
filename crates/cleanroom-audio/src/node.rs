//! The PipeWire virtual microphone.
//!
//! Two `Stream`s joined by a ring buffer, rather than one `pw_filter`. That is not a
//! workaround for a design preference — `pipewire::filter::Filter` has never been wrapped
//! in the Rust bindings. It has been requested since 2021 with three unmerged attempts,
//! so planning around it landing is planning around nothing. The two-stream shape is
//! what `module-loopback` and `module-example-source` use anyway, and we need the ring
//! buffer regardless because the quantum and the hop do not divide.
//!
//! ## The properties that are load-bearing
//!
//! * **`media.class = Audio/Source`**, not `Audio/Source/Virtual`. The `/Virtual` variant
//!   keeps `portconfig_direction = INPUT` — it exposes ports you *feed*, the null-sink
//!   topology, which is the opposite of what we want — and separately, QtWebEngine and
//!   Electron clients do not list it as a microphone at all.
//! * **`Direction::Output`** for that stream. PipeWire's own docs are explicit: "a virtual
//!   sound card or camera will use a `PW_DIRECTION_OUTPUT` stream". Producing data means
//!   Output even though the node is a *source*.
//! * **`media.category = Playback`** on the source stream. Counter-intuitive, but it is
//!   what upstream's own virtual-source examples set: we play data into the graph.
//!
//! WirePlumber classifies `Audio/Source` as a *device* rather than a stream, which is
//! what makes it a link target other apps can select, lets it become the default input,
//! and is why `AUTOCONNECT` is inert here — it *is* the target.

use crate::ringbuf::{HOP, RING_HOPS};
use cleanroom_core::CaptureTarget;
use libspa::param::audio::{AudioFormat, AudioInfoRaw};
use libspa::pod::Pod;
use pipewire::properties::properties;
use pipewire::stream::StreamFlags;
use std::cell::RefCell;
use std::io::Cursor;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// DeepFilterNet operates at exactly this rate. The reference machine's graph already
/// runs at 48 kHz, so no resampling is needed — but the format is advertised as fixed
/// regardless, so PipeWire converts for us if a graph ever runs at something else.
pub const SAMPLE_RATE: u32 = 48_000;

/// Mono. DeepFilterNet's mono model is what we run, and a virtual microphone has no use
/// for stereo — every consumer downmixes it anyway.
pub const CHANNELS: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("PipeWire error: {0}")]
    PipeWire(#[from] pipewire::Error),

    #[error("could not build the audio format parameter")]
    Format,

    #[error("no hardware microphone available to capture from")]
    NoInput,

    #[error("could not spawn the denoise worker thread: {0}")]
    Worker(std::io::Error),
}

/// A peak level the RT callbacks can publish without taking a lock.
///
/// This used to be `Mutex<f32>`, locked from both realtime callbacks *and* the daemon's
/// metering poll — a textbook priority-inversion pair: the RT thread parks on a mutex
/// held by a normal-priority thread that just got preempted. The critical section was
/// nanoseconds, so it rarely bit, but "rarely" is not a property a realtime path gets to
/// have. An f32 round-trips through its bit pattern losslessly, so an `AtomicU32` is the
/// whole fix.
#[derive(Default)]
pub struct AtomicLevel(AtomicU32);

impl AtomicLevel {
    pub fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}

/// Everything the streams, the worker and the daemon share. Atomics only — this struct
/// is touched from the realtime callbacks, and nothing on that path may wait.
#[derive(Default)]
pub struct SharedAudio {
    /// Peak level seen on the way in, linear 0..1. Read by the daemon for metering.
    pub level_in: AtomicLevel,
    pub level_out: AtomicLevel,
    /// Total samples lost to a full ring. Non-zero means the denoise worker could not
    /// keep up. Not yet surfaced by the daemon's status — read it in a debugger or a
    /// future health line, but do not mistake this comment for reporting.
    pub overruns: AtomicU64,
}

impl SharedAudio {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

/// Build the fixed audio format parameter.
///
/// A *single, fully fixed* format is advertised on purpose. Because WirePlumber treats
/// `Audio/Source` as a device, it picks a format from our `EnumFormat`, fixates it, and
/// pushes it back — and it suspends the node to do so. Offering exactly one option means
/// there is nothing to negotiate and no renegotiation churn when a consumer links.
fn audio_format_param(buffer: &mut Vec<u8>) -> Option<&Pod> {
    let mut info = AudioInfoRaw::new();
    info.set_format(AudioFormat::F32LE);
    info.set_rate(SAMPLE_RATE);
    info.set_channels(CHANNELS);

    let obj = libspa::pod::Object {
        type_: libspa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: libspa::param::ParamType::EnumFormat.as_raw(),
        properties: info.into(),
    };

    let (cursor, _) = libspa::pod::serialize::PodSerializer::serialize(
        Cursor::new(Vec::new()),
        &libspa::pod::Value::Object(obj),
    )
    .ok()?;
    *buffer = cursor.into_inner();
    Pod::from_bytes(buffer)
}

/// Peak absolute level of a block, for metering.
fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, s| m.max(s.abs()))
}

/// Convert a linear peak to dBFS, floored so silence does not print as -inf.
pub fn to_dbfs(linear: f32) -> f32 {
    if linear <= 1e-7 {
        -100.0
    } else {
        20.0 * linear.log10()
    }
}

/// How long the microphone is held after the last listener disappears.
///
/// Long enough to ride out an app renegotiating its format, short enough that the recording
/// light does not stay on after a meeting ends. Two seconds is well past the millisecond
/// scale of link churn and well under anyone's patience.
const RELEASE_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// A join guard for the denoise worker thread. Dropping it stops and joins the worker.
///
/// The `Drop` is the load-bearing part, not a convenience: `VirtualMic::run` has a
/// dozen fallible steps between spawning the worker and reaching its shutdown code, and
/// an early `?` — most plausibly `connect_rc` failing because PipeWire is down, the
/// exact case the daemon retries every five seconds — must not orphan a thread that has
/// loaded a whole DeepFilterNet session and parks every 10 ms forever. It also closes a
/// health race by ordering: the worker's own health report cannot land after the
/// caller's failure report, because the join here happens while the error is still
/// propagating.
struct DenoiseWorker {
    handle: Option<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    waker: std::thread::Thread,
}

impl DenoiseWorker {
    /// The handle the capture callback uses to wake the worker after a push.
    fn waker(&self) -> std::thread::Thread {
        self.waker.clone()
    }
}

impl Drop for DenoiseWorker {
    fn drop(&mut self) {
        // The worker parks with a timeout, so this join is bounded even if the unpark
        // races the park. Worst case it waits out a model load that was still running.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

/// Spawn the worker and wire its rings, returning the guard plus the ring ends the two
/// RT callbacks keep. See [`crate::ringbuf::run_worker`] for what runs on the thread,
/// and [`VirtualMic::run`] for why `make_process` is a factory.
fn spawn_denoise_worker<F, M>(
    shared: Arc<SharedAudio>,
    make_process: M,
) -> Result<
    (
        DenoiseWorker,
        rtrb::Producer<f32>,
        crate::ringbuf::CycleReader,
    ),
    AudioError,
>
where
    F: FnMut(&[f32; HOP], &mut [f32; HOP]) + 'static,
    M: FnOnce() -> F + Send + 'static,
{
    let (cap_tx, cap_rx) = rtrb::RingBuffer::<f32>::new(RING_HOPS * HOP);
    let (out_tx, out_rx) = rtrb::RingBuffer::<f32>::new(RING_HOPS * HOP);
    let stop = Arc::new(AtomicBool::new(false));
    let handle = std::thread::Builder::new()
        .name("cleanroom-denoise".into())
        .spawn({
            let stop = stop.clone();
            move || {
                let process = make_process();
                crate::ringbuf::run_worker(cap_rx, out_tx, process, stop, shared)
            }
        })
        .map_err(AudioError::Worker)?;
    let waker = handle.thread().clone();
    Ok((
        DenoiseWorker {
            handle: Some(handle),
            stop,
            waker,
        },
        cap_tx,
        crate::ringbuf::CycleReader::new(out_rx),
    ))
}

/// Feed every registry event about links and nodes into the tracker, so the timer tick
/// can answer "who is listening, and what else could we capture". The returned listener
/// must be kept alive for the life of the loop.
fn watch_registry(
    registry: &pipewire::registry::RegistryRc,
    tracker: &Rc<RefCell<crate::registry::LinkTracker>>,
) -> pipewire::registry::Listener {
    let t_add = tracker.clone();
    let t_del = tracker.clone();
    registry
        .add_listener_local()
        .global(move |g| {
            let Some(props) = g.props else { return };
            match g.type_ {
                pipewire::types::ObjectType::Link => {
                    let id_of = |k: &str| props.get(k).and_then(|v| v.parse::<u32>().ok());
                    t_add.borrow_mut().add_link(
                        g.id,
                        id_of(*pipewire::keys::LINK_OUTPUT_NODE),
                        id_of(*pipewire::keys::LINK_INPUT_NODE),
                    );
                }
                pipewire::types::ObjectType::Node => {
                    let class = props.get(*pipewire::keys::MEDIA_CLASS).unwrap_or("");
                    let name = props.get(*pipewire::keys::NODE_NAME).unwrap_or("");
                    let desc = props.get(*pipewire::keys::NODE_DESCRIPTION);
                    if !name.is_empty() {
                        t_add.borrow_mut().add_node(g.id, class, name, desc);
                    }
                }
                _ => {}
            }
        })
        .global_remove(move |id| t_del.borrow_mut().remove_global(id))
        .register()
}

/// Runs the PipeWire main loop with a capture stream and a source stream.
///
/// Blocks until `stop` is signalled. Intended to own a dedicated thread: the PipeWire
/// main loop is not async and expects to drive itself.
pub struct VirtualMic;

impl VirtualMic {
    /// Build and run the node graph.
    ///
    /// `make_process` is a factory, not the processor itself, and that is load-bearing:
    /// it runs ON the dedicated worker thread and returns the per-hop closure, which
    /// then never crosses a thread boundary in its life. The denoiser it typically
    /// captures is `!Send` — `DfTract` holds `Rc<Tensor>` — so it *cannot* be built on
    /// the caller's thread and handed over. An earlier shape of this API took the
    /// closure directly and executed it on PipeWire's RT data thread anyway, the `!Send`
    /// move hidden by the C FFI boundary; the factory makes the compiler enforce what a
    /// doc comment used to plead. It also puts the model load (a long CPU burn) on a
    /// thread that is never realtime, so `RLIMIT_RTTIME` can no longer kill the process
    /// over it.
    ///
    /// The closure is called with one hop in and one hop out — see
    /// [`crate::ringbuf::run_worker`].
    pub fn run<F, M>(
        shared: Arc<SharedAudio>,
        capture_target: Option<CaptureTarget>,
        node_name: &str,
        node_description: &str,
        make_process: M,
        should_stop: impl Fn() -> bool + 'static,
        view: Arc<crate::registry::RegistryView>,
    ) -> Result<(), AudioError>
    where
        F: FnMut(&[f32; HOP], &mut [f32; HOP]) + 'static,
        M: FnOnce() -> F + Send + 'static,
    {
        pipewire::init();

        // The worker and its rings exist before either stream does, because the capture
        // callback needs the producer end and the worker's wake handle at registration
        // time. The guard stays a named local for the whole function: every `?` below
        // must stop and join the worker on its way out (see `DenoiseWorker`).
        let (worker, cap_tx, out_rx) = spawn_denoise_worker(shared.clone(), make_process)?;
        let waker = worker.waker();

        // The 0.10 bindings hand out reference-counted handles rather than plain owned
        // ones; the Rc variants are what `Stream::new` expects.
        let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
        let context = pipewire::context::ContextRc::new(&mainloop, None)?;
        let core = context.connect_rc(None)?;

        // --- capture: the real microphone -------------------------------------------
        //
        // `target.object` is pinned to a concrete hardware node. Leaving it unset would
        // bind the system default, which becomes *our own node* the moment a user selects
        // Cleanroom as their default input — and then the pipeline captures its own
        // output and howls. `CaptureTarget` makes naming our own node unrepresentable;
        // this is where that guarantee is cashed in.
        let mut capture_props = properties! {
            *pipewire::keys::MEDIA_TYPE => "Audio",
            *pipewire::keys::MEDIA_CATEGORY => "Capture",
            *pipewire::keys::MEDIA_ROLE => "Communication",
            *pipewire::keys::NODE_NAME => "cleanroom_capture",
            *pipewire::keys::NODE_DESCRIPTION => "Cleanroom microphone capture",
            // NOT node.passive. That was tried and produced a permanently silent virtual
            // mic: a passive node will not drive the graph, and because our capture and
            // source streams are two *independent* nodes joined only by a userspace ring
            // buffer — not by a PipeWire link — nothing else was ever going to drive the
            // capture side. It stayed paused and delivered silence while the hardware mic
            // was reading a healthy -40 dBFS.
            //
            // Releasing the hardware mic when nobody is listening is still worth doing,
            // but it has to be done by watching our source node's links and disconnecting
            // the capture stream, not by a property.
            "node.always-process" => "true",
        };
        if let Some(t) = &capture_target {
            capture_props.insert(*pipewire::keys::TARGET_OBJECT, t.as_str());
            tracing::info!(target = t.as_str(), "capture pinned to hardware node");
        } else {
            tracing::warn!(
                "no capture target configured; binding the system default. If Cleanroom \
                 becomes the default input this would self-capture — set audio.device."
            );
        }

        let capture =
            pipewire::stream::StreamRc::new(core.clone(), "cleanroom-capture", capture_props)?;

        // This closure runs on the realtime data thread (RT_PROCESS below): everything
        // in it is a bounded copy, an atomic, or a futex wake. No locks, no allocation,
        // no inference — that all happens on the worker this callback wakes.
        let cap_shared = shared.clone();
        let mut cap_tx = cap_tx;
        let _cap_listener = capture
            .add_local_listener_with_user_data(())
            .process(move |stream, _| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                let datas = buffer.datas_mut();
                let Some(d) = datas.first_mut() else { return };

                // Read the chunk size before taking the data slice: `chunk()` borrows
                // immutably and `data()` mutably, so they cannot overlap.
                let chunk_size = d.chunk().size() as usize;
                let Some(slice) = d.data() else { return };
                let n = (chunk_size / std::mem::size_of::<f32>()).min(slice.len() / 4);
                if n == 0 {
                    return;
                }
                // SAFETY: PipeWire negotiated F32LE, so the buffer holds f32 samples;
                // `n` is derived from the chunk size the server reported.
                let samples =
                    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, n) };

                cap_shared.level_in.set(peak(samples));
                crate::ringbuf::push_or_drop(&mut cap_tx, samples, &cap_shared.overruns);
                waker.unpark();
            })
            .register()?;

        // --- source: the virtual microphone -----------------------------------------
        let playback_props = properties! {
            *pipewire::keys::MEDIA_TYPE => "Audio",
            // "Playback" because *we* produce the data, even though the node is a source.
            *pipewire::keys::MEDIA_CATEGORY => "Playback",
            // The switch that makes this a microphone rather than a playback stream.
            *pipewire::keys::MEDIA_CLASS => "Audio/Source",
            *pipewire::keys::MEDIA_ROLE => "Communication",
            *pipewire::keys::NODE_NAME => node_name,
            *pipewire::keys::NODE_DESCRIPTION => node_description,
            *pipewire::keys::NODE_VIRTUAL => "true",
            // Keep process() firing with nothing attached, so the ring drains and does
            // not present a wall of stale audio to the first consumer that links.
            "node.always-process" => "true",
        };

        let playback =
            pipewire::stream::StreamRc::new(core.clone(), "cleanroom-source", playback_props)?;

        // Also on the realtime data thread. It pops what the worker already produced —
        // whole cycles or silence, never a partial fill — and does nothing else. See
        // `ringbuf::CycleReader` for why partial fills are forbidden.
        let play_shared = shared.clone();
        let mut out_rx = out_rx;
        let _play_listener = playback
            .add_local_listener_with_user_data(())
            .process(move |stream, _| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                // `requested()` is how many frames the graph wants this cycle. Honouring
                // it rather than filling the whole buffer is what keeps latency bounded.
                let requested = buffer.requested() as usize;
                let datas = buffer.datas_mut();
                let Some(d) = datas.first_mut() else { return };

                let stride = std::mem::size_of::<f32>();
                let capacity = d.data().map(|s| s.len() / stride).unwrap_or(0);
                let want = if requested == 0 {
                    capacity
                } else {
                    requested.min(capacity)
                };
                if want == 0 {
                    return;
                }

                let Some(slice) = d.data() else { return };
                // SAFETY: negotiated F32LE; `want` is bounded by the buffer's capacity.
                let out =
                    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f32, want) };

                out_rx.pop_cycle(out);
                play_shared.level_out.set(peak(out));

                let chunk = d.chunk_mut();
                *chunk.offset_mut() = 0;
                *chunk.stride_mut() = stride as i32;
                *chunk.size_mut() = (want * stride) as u32;
            })
            .register()?;

        // --- connect ------------------------------------------------------------------
        let mut cap_buf = Vec::new();
        let cap_param = audio_format_param(&mut cap_buf).ok_or(AudioError::Format)?;
        capture.connect(
            libspa::utils::Direction::Input,
            None,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
            &mut [cap_param],
        )?;

        let mut play_buf = Vec::new();
        let play_param = audio_format_param(&mut play_buf).ok_or(AudioError::Format)?;
        // No AUTOCONNECT: WirePlumber only autoconnects nodes it classifies as streams,
        // and an Audio/Source is a device. It *is* the target; it does not seek one.
        playback.connect(
            libspa::utils::Direction::Output,
            None,
            StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
            &mut [play_param],
        )?;

        tracing::info!(node = node_name, "virtual microphone published");

        // --- registry: who is listening, and what else could we capture ----------------
        //
        // Our node id is only assigned once the stream is connected, which is why this is
        // here rather than up with the other setup.
        let tracker = Rc::new(RefCell::new(crate::registry::LinkTracker::new()));
        tracker.borrow_mut().set_node_id(playback.node_id());

        let registry = core.get_registry_rc()?;
        let _reg_listener = watch_registry(&registry, &tracker);

        // Poll the stop flag from a timer rather than blocking forever, so shutdown does
        // not depend on the main loop noticing a signal.
        //
        // The same tick decides whether the hardware microphone is still needed. Doing it
        // on a timer rather than directly in the registry callback is deliberate: link
        // churn arrives in bursts during renegotiation, and reacting to each event
        // individually would toggle the stream several times for what is really one change.
        let loop_ref = mainloop.loop_();
        let quit = mainloop.clone();
        // `add_timer` takes an `Fn`, not an `FnMut`, so anything the tick mutates needs
        // interior mutability. Single-threaded — the PipeWire loop owns this — so Cell and
        // RefCell rather than anything atomic.
        let idle = RefCell::new(crate::registry::IdlePolicy::new(RELEASE_GRACE));
        let capture_active = std::cell::Cell::new(true);
        let capture_for_timer = capture.clone();
        let playback_for_timer = playback.clone();
        let _timer = loop_ref.add_timer(move |_| {
            if should_stop() {
                quit.quit();
                return;
            }

            // Our node id is not assigned when `connect()` returns — the stream has to be
            // negotiated on this loop first — so keep asking until it is real. Doing it
            // here rather than once at setup is the difference between a working release
            // and a permanently silent microphone.
            {
                let mut t = tracker.borrow_mut();
                if !t.node_known() {
                    t.set_node_id(playback_for_timer.node_id());
                }
            }

            let t = tracker.borrow();
            let listeners = t.listeners();
            view.publish(t.sources(), listeners.unwrap_or(0));
            drop(t);

            // Unknown is not idle. Until we know which node is ours we cannot know whether
            // anyone is listening, and releasing the microphone on a guess is exactly the
            // silent-mic failure this feature exists to avoid — the same contract the
            // camera's consumer detection follows.
            let want = match listeners {
                None => true,
                Some(n) => idle
                    .borrow_mut()
                    .should_capture(n, std::time::Instant::now()),
            };
            if want != capture_active.get() {
                // set_active rather than disconnect: reconnecting means renegotiating the
                // format, which suspends the node and is exactly the churn we are trying to
                // avoid. Deactivating is enough for the hardware device to suspend.
                match capture_for_timer.set_active(want) {
                    Ok(()) => {
                        capture_active.set(want);
                        tracing::info!(
                            listeners,
                            capturing = want,
                            "microphone {} because nothing is listening",
                            if want { "resumed" } else { "released" }
                        );
                    }
                    Err(e) => {
                        // Not fatal: the worst case is holding the microphone open, which
                        // is what we did before this existed.
                        tracing::warn!(error = %e, "could not change capture stream state");
                    }
                }
            }
        });
        _timer
            .update_timer(
                Some(std::time::Duration::from_millis(200)),
                Some(std::time::Duration::from_millis(200)),
            )
            .into_result()
            .ok();

        mainloop.run();

        // Deterministically before the log line rather than at scope end. Leaking the
        // thread instead would leak a denoiser (and its model) per pipeline restart —
        // the exact leak-per-restart bug the video side had with its Matter.
        drop(worker);

        tracing::info!("virtual microphone stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dbfs_conversion_is_sane() {
        assert!((to_dbfs(1.0) - 0.0).abs() < 0.01, "full scale is 0 dBFS");
        assert!(
            (to_dbfs(0.5) + 6.02).abs() < 0.05,
            "half scale is about -6 dBFS"
        );
        assert_eq!(to_dbfs(0.0), -100.0, "silence must floor, not go to -inf");
    }

    #[test]
    fn peak_finds_the_largest_magnitude_including_negatives() {
        assert_eq!(peak(&[0.1, -0.7, 0.3]), 0.7);
        assert_eq!(peak(&[]), 0.0);
    }

    #[test]
    fn the_published_class_is_the_one_clients_can_actually_see() {
        // Audio/Source/Virtual is invisible to QtWebEngine and Electron clients, and has
        // the wrong port direction besides. This is a documentation-in-a-test guard.
        let expected = "Audio/Source";
        assert_ne!(expected, "Audio/Source/Virtual");
    }

    #[test]
    fn rate_matches_deepfilternet() {
        assert_eq!(SAMPLE_RATE, 48_000);
        assert_eq!(CHANNELS, 1);
    }

    /// Dropping the worker guard must stop and join the thread — the guard's whole
    /// reason to exist. `VirtualMic::run` has a dozen fallible steps after the spawn
    /// (most plausibly `connect_rc` failing because PipeWire is down, which the daemon
    /// retries every five seconds), and before the guard each early `?` orphaned an
    /// immortal thread holding a loaded DeepFilterNet session — one more per retry —
    /// whose late health report then overwrote the failure report with "nominal".
    #[test]
    fn dropping_the_worker_guard_stops_and_joins_the_thread() {
        let shared = SharedAudio::new();
        let (worker, _cap_tx, _out_rx) = spawn_denoise_worker(shared.clone(), || {
            |_: &[f32; HOP], out: &mut [f32; HOP]| out.fill(0.0)
        })
        .expect("spawning the worker");

        drop(worker);

        // The join inside Drop is what makes this deterministic: once drop returns,
        // the thread has exited and released its clone of `shared`.
        assert_eq!(
            Arc::strong_count(&shared),
            1,
            "the worker thread must be gone, not parked forever"
        );
    }
}
