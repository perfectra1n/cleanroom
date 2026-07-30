//! The PipeWire `Video/Source` node: the second virtual-camera transport.
//!
//! v4l2loopback reaches Chrome, Electron, Zoom, Discord and OBS, which is most users, and
//! is why it was built first. It does not reach everybody:
//!
//! * Flatpak and portal-aware apps can only see a PipeWire node.
//! * Fedora patches Firefox's `media.webrtc.camera.allow-pipewire` on, and
//!   [PipeWire #3659] means loopback source nodes are not created when device capabilities
//!   change — which is exactly what `exclusive_caps=1` does when a producer attaches. So on
//!   Fedora Firefox the loopback device can simply never appear.
//!
//! Neither transport alone reaches everyone, so both are published.
//!
//! ## Two properties, both mandatory
//!
//! `media.class = Video/Source` **and** `media.role = Camera`. xdg-desktop-portal's
//! `camera.c` and WirePlumber's `find-portal-access.lua` check them *independently*, so a
//! node with only the class is invisible to the portal — upstream's own `video-src.c` sets
//! only the class and would not be seen.
//!
//! ## Why YUY2 rather than I420
//!
//! The prior art (`funinkina/openeffects`) advertises I420. Our composite already produces
//! YUY2 for the loopback sink, so advertising YUY2 means the same buffer feeds both
//! transports with no conversion. It is also the format chosen precisely because Firefox
//! and Chromium both accept it without complaint.
//!
//! [PipeWire #3659]: https://gitlab.freedesktop.org/pipewire/pipewire/-/issues/3659

use libspa::param::video::VideoFormat;
use libspa::pod::Pod;
use pipewire::properties::properties;
use pipewire::stream::{StreamFlags, StreamState};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum PwSourceError {
    #[error("PipeWire error: {0}")]
    PipeWire(#[from] pipewire::Error),

    #[error("could not build the video format parameter")]
    Format,
}

/// The most recent processed frame, handed from the video thread to the PipeWire thread.
///
/// A single slot rather than a queue, deliberately. A camera has no use for stale frames:
/// if the consumer is behind, the right thing is to give it the newest frame and drop the
/// rest, which is what overwriting a slot does for free. A queue would add latency and then
/// need a policy to throw it away again.
pub struct FrameSlot {
    inner: Mutex<Slot>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

struct Slot {
    data: Vec<u8>,
    /// Bumped on every write, so the consumer can tell a fresh frame from a repeat.
    seq: u64,
}

impl FrameSlot {
    pub fn new(width: u32, height: u32, fps: u32) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Slot {
                data: Vec::new(),
                seq: 0,
            }),
            width,
            height,
            fps,
        })
    }

    /// Bytes in one YUY2 frame at this geometry.
    pub fn frame_bytes(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 2
    }

    /// Publish a frame. Called from the video thread once per processed frame.
    pub fn put(&self, frame: &[u8]) {
        if let Ok(mut g) = self.inner.lock() {
            g.data.clear();
            g.data.extend_from_slice(frame);
            g.seq = g.seq.wrapping_add(1);
        }
    }

    /// Copy the latest frame into `out`, returning how many bytes were written.
    fn copy_latest(&self, out: &mut [u8]) -> Option<usize> {
        let g = self.inner.lock().ok()?;
        if g.data.is_empty() {
            return None;
        }
        let n = g.data.len().min(out.len());
        out[..n].copy_from_slice(&g.data[..n]);
        Some(n)
    }
}

/// Fill a buffer with black in YUY2.
///
/// Not zeroes. All-zero YUY2 decodes to mid-green through BT.601, because Y=0 clamps to
/// black luma while U=V=0 puts both chroma channels at -128. Y=16, U=V=128 is actual
/// limited-range black. A consumer that connects before the first frame arrives should see
/// black, not a green flash that looks like a broken effect.
fn fill_black_yuy2(buf: &mut [u8]) {
    for px in buf.chunks_mut(2) {
        px[0] = 16;
        if px.len() > 1 {
            px[1] = 128;
        }
    }
}

/// Build the fixed video format parameter.
///
/// One fully-fixed option, for the same reason the audio node advertises one: there is
/// nothing to negotiate, so no renegotiation churn when a consumer links.
fn video_format_param(buffer: &mut Vec<u8>, w: u32, h: u32, fps: u32) -> Option<&Pod> {
    use libspa::param::format::{FormatProperties, MediaSubtype, MediaType};
    use libspa::pod::{Property, PropertyFlags, Value};

    // Built by hand rather than from `VideoInfoRaw`. libspa implements
    // `From<AudioInfoRaw> for Vec<Property>` but has no video equivalent, so the audio
    // node's `info.into()` shortcut does not exist here — the properties have to be listed.
    let props = vec![
        Property {
            key: FormatProperties::MediaType.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(libspa::utils::Id(MediaType::Video.as_raw())),
        },
        Property {
            key: FormatProperties::MediaSubtype.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(libspa::utils::Id(MediaSubtype::Raw.as_raw())),
        },
        Property {
            key: FormatProperties::VideoFormat.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(libspa::utils::Id(VideoFormat::YUY2.as_raw())),
        },
        Property {
            key: FormatProperties::VideoSize.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Rectangle(libspa::utils::Rectangle {
                width: w,
                height: h,
            }),
        },
        Property {
            key: FormatProperties::VideoFramerate.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Fraction(libspa::utils::Fraction { num: fps, denom: 1 }),
        },
    ];

    let obj = libspa::pod::Object {
        type_: libspa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: libspa::param::ParamType::EnumFormat.as_raw(),
        properties: props,
    };

    let (cursor, _) = libspa::pod::serialize::PodSerializer::serialize(
        Cursor::new(Vec::new()),
        &libspa::pod::Value::Object(obj),
    )
    .ok()?;
    *buffer = cursor.into_inner();
    Pod::from_bytes(buffer)
}

pub struct PwSource;

impl PwSource {
    /// Publish the node and run the loop until `should_stop`.
    ///
    /// Owns a dedicated thread: the PipeWire main loop drives itself, and the video thread
    /// is already blocked in V4L2 `DQBUF` and cannot host it.
    ///
    /// `active` is this node's answer to "is anybody watching?", and it is what lets power
    /// save see a PipeWire consumer at all — `ConsumerWatch` counts v4l2loopback STREAMONs
    /// and is blind to this transport entirely. It is cleared before returning, so a
    /// caller that keeps the flag after the loop ends reads "nobody", never a stale yes.
    pub fn run(
        slot: Arc<FrameSlot>,
        node_description: &str,
        should_stop: impl Fn() -> bool + 'static,
        active: Arc<AtomicBool>,
    ) -> Result<(), PwSourceError> {
        pipewire::init();

        let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
        let context = pipewire::context::ContextRc::new(&mainloop, None)?;
        let core = context.connect_rc(None)?;

        let props = properties! {
            *pipewire::keys::MEDIA_TYPE => "Video",
            *pipewire::keys::MEDIA_CATEGORY => "Source",
            // Both of these are required, and each is checked by a different consumer.
            *pipewire::keys::MEDIA_CLASS => "Video/Source",
            *pipewire::keys::MEDIA_ROLE => "Camera",
            *pipewire::keys::NODE_NAME => cleanroom_core::node::VIRTUAL_CAM_NODE,
            *pipewire::keys::NODE_DESCRIPTION => node_description,
        };

        let stream = pipewire::stream::StreamRc::new(core.clone(), "cleanroom-camera", props)?;

        let (w, h, fps) = (slot.width, slot.height, slot.fps);
        let stride = (w as usize) * 2;
        let frame_bytes = slot.frame_bytes();

        let slot_cb = slot.clone();
        let active_cb = active.clone();
        let _listener = stream
            .add_local_listener_with_user_data(())
            // Streaming means at least one consumer, and it means it exactly. This node is
            // a DRIVER, and WirePlumber only runs a DRIVER source node while something is
            // linked to it — so the graph reaches Streaming when the first consumer links
            // and leaves it when the last one goes. Paused, Connecting, Unconnected and
            // Error all mean nobody is watching, which is why this is a `matches!` on the
            // one state rather than a list of the ones to exclude.
            .state_changed(move |_stream, _ud, _old, new| {
                active_cb.store(matches!(new, StreamState::Streaming), Ordering::Relaxed);
            })
            .param_changed(move |stream, _, id, param| {
                if param.is_none() || id != libspa::param::ParamType::Format.as_raw() {
                    return;
                }
                // Answer with our buffer requirements and the header metadata. The header
                // is not optional in practice: WebRTC consumers expect it, and its absence
                // shows up as a consumer that connects and then sees nothing.
                let mut b_buf = Vec::new();
                let mut m_buf = Vec::new();
                let (Some(buffers), Some(meta)) = (
                    buffers_param(&mut b_buf, frame_bytes, stride),
                    meta_header_param(&mut m_buf),
                ) else {
                    tracing::warn!("could not build buffer/meta params");
                    return;
                };
                if let Err(e) = stream.update_params(&mut [buffers, meta]) {
                    tracing::warn!(error = %e, "update_params failed");
                }
            })
            .process(move |stream, _| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }
                let data = &mut datas[0];
                let written = match data.data() {
                    Some(dst) => match slot_cb.copy_latest(dst) {
                        Some(n) => n,
                        None => {
                            // Nothing produced yet. Black, so a consumer that gets here
                            // first sees a valid picture rather than uninitialised memory.
                            let n = frame_bytes.min(dst.len());
                            fill_black_yuy2(&mut dst[..n]);
                            n
                        }
                    },
                    None => 0,
                };
                let chunk = data.chunk_mut();
                *chunk.offset_mut() = 0;
                *chunk.stride_mut() = stride as i32;
                *chunk.size_mut() = written as u32;
            })
            .register()?;

        let mut fmt_buf = Vec::new();
        let fmt = video_format_param(&mut fmt_buf, w, h, fps).ok_or(PwSourceError::Format)?;

        // DRIVER because a virtual camera has no hardware clock: nothing else in the graph
        // can pace us, so we pace the graph. RT_PROCESS because the callback is a memcpy
        // from a slot, which is short and bounded.
        stream.connect(
            libspa::utils::Direction::Output,
            None,
            StreamFlags::DRIVER | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
            &mut [fmt],
        )?;

        tracing::info!(
            node = cleanroom_core::node::VIRTUAL_CAM_NODE,
            "{}x{}@{} published as a PipeWire Video/Source",
            w,
            h,
            fps
        );

        // Bound to locals, not dropped: a `TimerSource` removes its source from the loop
        // when it falls, so `let _ = install_timers(..)` would arm both timers and disarm
        // them again on the same line — no cadence, and nothing to notice the stop flag.
        let (_frame_timer, _stop_timer) = install_timers(
            mainloop.loop_(),
            mainloop.clone(),
            stream.clone(),
            fps,
            should_stop,
        );

        mainloop.run();
        // The loop is over, so no consumer can be attached to it any more. Say so before
        // returning: the daemon keeps reading this flag, and a stale `true` left behind by
        // a dead thread would hold the camera awake forever.
        active.store(false, Ordering::Relaxed);
        tracing::info!("PipeWire Video/Source stopped");
        Ok(())
    }
}

/// Arm the frame tick and the stop poll on `loop_`, returning both guards.
///
/// Split out of [`PwSource::run`] purely to keep that function inside its length budget;
/// the two timers have nothing else to do with each other. The guards are returned rather
/// than kept here because they **must outlive `mainloop.run()`** — see the call site.
fn install_timers<'l>(
    loop_: &'l pipewire::loop_::Loop,
    mainloop: pipewire::main_loop::MainLoopRc,
    stream: pipewire::stream::StreamRc,
    fps: u32,
    should_stop: impl Fn() -> bool + 'static,
) -> (
    pipewire::loop_::TimerSource<'l>,
    pipewire::loop_::TimerSource<'l>,
) {
    // Drive the cadence ourselves. `is_driving()` is false while the stream is paused, so
    // the tick becomes a cheap no-op rather than something to guard separately.
    let interval = std::time::Duration::from_nanos(1_000_000_000 / fps.max(1) as u64);
    let frame_timer = loop_.add_timer(move |_| {
        if stream.is_driving() {
            let _ = stream.trigger_process();
        }
    });
    frame_timer
        .update_timer(Some(interval), Some(interval))
        .into_result()
        .ok();

    let poll = std::time::Duration::from_millis(200);
    let stop_timer = loop_.add_timer(move |_| {
        if should_stop() {
            mainloop.quit();
        }
    });
    stop_timer
        .update_timer(Some(poll), Some(poll))
        .into_result()
        .ok();

    (frame_timer, stop_timer)
}

/// `SPA_PARAM_Buffers`: how many buffers, how big, and how they are laid out.
fn buffers_param(buffer: &mut Vec<u8>, size: usize, stride: usize) -> Option<&Pod> {
    use libspa::pod::{ChoiceValue, Property, PropertyFlags, Value};
    use libspa::utils::{Choice, ChoiceEnum, ChoiceFlags};

    let obj = libspa::pod::Object {
        type_: libspa::utils::SpaTypes::ObjectParamBuffers.as_raw(),
        id: libspa::param::ParamType::Buffers.as_raw(),
        properties: vec![
            Property {
                key: libspa_sys::SPA_PARAM_BUFFERS_buffers,
                flags: PropertyFlags::empty(),
                value: Value::Choice(ChoiceValue::Int(Choice(
                    ChoiceFlags::empty(),
                    // A range rather than a fixed count: the consumer knows better than we
                    // do how much buffering it wants, and 8 is PipeWire's usual ceiling.
                    ChoiceEnum::Range {
                        default: 4,
                        min: 2,
                        max: 8,
                    },
                ))),
            },
            Property {
                key: libspa_sys::SPA_PARAM_BUFFERS_blocks,
                flags: PropertyFlags::empty(),
                value: Value::Int(1),
            },
            Property {
                key: libspa_sys::SPA_PARAM_BUFFERS_size,
                flags: PropertyFlags::empty(),
                value: Value::Int(size as i32),
            },
            Property {
                key: libspa_sys::SPA_PARAM_BUFFERS_stride,
                flags: PropertyFlags::empty(),
                value: Value::Int(stride as i32),
            },
        ],
    };

    let (cursor, _) = libspa::pod::serialize::PodSerializer::serialize(
        Cursor::new(Vec::new()),
        &Value::Object(obj),
    )
    .ok()?;
    *buffer = cursor.into_inner();
    Pod::from_bytes(buffer)
}

/// `SPA_PARAM_Meta` for the buffer header.
///
/// Mandatory in practice. WebRTC consumers expect a header on every buffer, and without it
/// they connect successfully and then display nothing — a failure with no error anywhere.
fn meta_header_param(buffer: &mut Vec<u8>) -> Option<&Pod> {
    use libspa::pod::{Property, PropertyFlags, Value};

    let obj = libspa::pod::Object {
        type_: libspa::utils::SpaTypes::ObjectParamMeta.as_raw(),
        id: libspa::param::ParamType::Meta.as_raw(),
        properties: vec![
            Property {
                key: libspa_sys::SPA_PARAM_META_type,
                flags: PropertyFlags::empty(),
                value: Value::Id(libspa::utils::Id(libspa_sys::SPA_META_Header)),
            },
            Property {
                key: libspa_sys::SPA_PARAM_META_size,
                flags: PropertyFlags::empty(),
                value: Value::Int(std::mem::size_of::<libspa_sys::spa_meta_header>() as i32),
            },
        ],
    };

    let (cursor, _) = libspa::pod::serialize::PodSerializer::serialize(
        Cursor::new(Vec::new()),
        &Value::Object(obj),
    )
    .ok()?;
    *buffer = cursor.into_inner();
    Pod::from_bytes(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-zero YUY2 is mid-green, not black, so "just memset it" is wrong here. This is
    /// the same confusion that made a zeroed loopback buffer look like green-screen mode.
    #[test]
    fn the_placeholder_frame_is_black_rather_than_zeroed() {
        let mut buf = vec![0u8; 16];
        fill_black_yuy2(&mut buf);
        for (i, &b) in buf.iter().enumerate() {
            let want = if i % 2 == 0 { 16 } else { 128 };
            assert_eq!(b, want, "byte {i} should be limited-range black");
        }
    }

    #[test]
    fn an_odd_length_buffer_does_not_panic() {
        let mut buf = vec![0u8; 7];
        fill_black_yuy2(&mut buf);
        assert_eq!(buf[6], 16);
    }

    #[test]
    fn frame_bytes_matches_yuy2_at_the_declared_geometry() {
        let s = FrameSlot::new(1920, 1080, 30);
        assert_eq!(s.frame_bytes(), 1920 * 1080 * 2);
    }

    /// The slot must report "nothing yet" rather than a stale or empty frame, so the
    /// process callback can substitute black instead of publishing garbage.
    #[test]
    fn an_empty_slot_reports_nothing_and_a_written_one_reports_bytes() {
        let s = FrameSlot::new(4, 2, 30);
        let mut out = vec![0u8; s.frame_bytes()];
        assert_eq!(s.copy_latest(&mut out), None);

        s.put(&vec![0xABu8; s.frame_bytes()]);
        assert_eq!(s.copy_latest(&mut out), Some(s.frame_bytes()));
        assert!(out.iter().all(|&b| b == 0xAB));
    }

    /// Newest-wins is the point: a camera consumer wants the current frame, not a backlog.
    #[test]
    fn a_second_frame_replaces_the_first() {
        let s = FrameSlot::new(4, 2, 30);
        s.put(&vec![1u8; s.frame_bytes()]);
        s.put(&vec![2u8; s.frame_bytes()]);
        let mut out = vec![0u8; s.frame_bytes()];
        s.copy_latest(&mut out);
        assert!(out.iter().all(|&b| b == 2), "the newer frame must win");
    }
}
