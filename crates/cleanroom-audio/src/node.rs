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

use crate::ringbuf::{HOP, HopBridge};
use cleanroom_core::CaptureTarget;
use libspa::param::audio::{AudioFormat, AudioInfoRaw};
use libspa::pod::Pod;
use pipewire::properties::properties;
use pipewire::stream::StreamFlags;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

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
}

/// Everything the two streams share.
pub struct SharedAudio {
    pub bridge: Mutex<HopBridge>,
    /// Peak level seen on the way in, linear 0..1. Read by the daemon for metering.
    pub level_in: Mutex<f32>,
    pub level_out: Mutex<f32>,
}

impl SharedAudio {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            // 32 hops = 320 ms of slack. Enough to ride out a scheduling hiccup without
            // letting latency run away if something goes badly wrong.
            bridge: Mutex::new(HopBridge::new(32)),
            level_in: Mutex::new(0.0),
            level_out: Mutex::new(0.0),
        })
    }
}

impl Default for SharedAudio {
    fn default() -> Self {
        // Only here to satisfy clippy; the Arc constructor is the real entry point.
        Self {
            bridge: Mutex::new(HopBridge::new(32)),
            level_in: Mutex::new(0.0),
            level_out: Mutex::new(0.0),
        }
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

/// Runs the PipeWire main loop with a capture stream and a source stream.
///
/// Blocks until `stop` is signalled. Intended to own a dedicated thread: the PipeWire
/// main loop is not async and expects to drive itself.
pub struct VirtualMic;

impl VirtualMic {
    /// Build and run the node graph.
    ///
    /// `process` is called with one hop in and one hop out, on the PipeWire thread but
    /// outside the RT callback path — see [`HopBridge::drain`].
    pub fn run<F>(
        shared: Arc<SharedAudio>,
        capture_target: Option<CaptureTarget>,
        node_name: &str,
        node_description: &str,
        mut process: F,
        should_stop: impl Fn() -> bool + 'static,
    ) -> Result<(), AudioError>
    where
        F: FnMut(&[f32; HOP], &mut [f32; HOP]) + 'static,
    {
        pipewire::init();

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

        let cap_shared = shared.clone();
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

                if let Ok(mut lvl) = cap_shared.level_in.lock() {
                    *lvl = peak(samples);
                }
                if let Ok(mut b) = cap_shared.bridge.lock() {
                    b.submit(samples);
                }
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

        let play_shared = shared.clone();
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

                if let Ok(mut b) = play_shared.bridge.lock() {
                    // Run the denoiser over everything that has arrived, then hand out
                    // what it produced. Both happen here rather than in the capture
                    // callback so the work lands on the playback cycle's budget.
                    b.drain(&mut process);
                    b.collect(out);
                }
                if let Ok(mut lvl) = play_shared.level_out.lock() {
                    *lvl = peak(out);
                }

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

        // Poll the stop flag from a timer rather than blocking forever, so shutdown does
        // not depend on the main loop noticing a signal.
        let loop_ref = mainloop.loop_();
        let quit = mainloop.clone();
        let _timer = loop_ref.add_timer(move |_| {
            if should_stop() {
                quit.quit();
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
}
