//! Capturing the daemon's own PipeWire node, so the GUI preview is not a second v4l2 client.
//!
//! The preview used to open the v4l2loopback device like any other consumer. That works
//! exactly once: a loopback node accepts **one** streaming capture consumer at a time, so
//! with the preview open the user's actual meeting app found the camera busy — and with the
//! meeting app open the preview showed nothing. The two features were mutually exclusive.
//!
//! PipeWire has no such limit. The daemon already publishes `cleanroom_cam` as a
//! `Video/Source` (see [`crate::pw_source`]), and a PipeWire node can be linked by any
//! number of consumers at once. So the preview captures from *that* node and leaves the
//! loopback device entirely to the outside world.
//!
//! ## The format is negotiated, not declared
//!
//! Unlike the source side — which knows its geometry because it chose it — a consumer knows
//! only that it wants YUY2. The daemon's resolution is whatever the user's camera and
//! settings produced. So the `EnumFormat` offered here leaves size and framerate *unfixed*
//! (a `Choice::Range` rather than a fixed value) and the real geometry is read out of the
//! `Format` param that PipeWire hands back once both ends agree. Every frame is delivered
//! with the negotiated width and height alongside it, because the preview has no other way
//! to learn them and they can change under it when the daemon is reconfigured.

use libspa::param::video::VideoFormat;
use libspa::pod::Pod;
use pipewire::properties::PropertiesBox;
use pipewire::stream::{StreamFlags, StreamListener, StreamState};
use std::cell::Cell;
use std::io::Cursor;
use std::rc::Rc;

#[derive(Debug, thiserror::Error)]
pub enum PwCaptureError {
    #[error("PipeWire error: {0}")]
    PipeWire(#[from] pipewire::Error),

    #[error("could not build the video format parameter")]
    Format,

    #[error("the capture stream lost its link to `{0}`")]
    Disconnected(String),
}

/// What the three stream callbacks share, and the one bit `run` needs back afterwards.
struct CaptureState {
    /// The negotiated geometry, written by `param_changed` and read by `process`.
    ///
    /// Zero until negotiation completes. `process` cannot fire before `param_changed` in
    /// practice, but a frame delivered with a 0x0 geometry would be a lie, so it is
    /// treated as "not ready" rather than trusted.
    size: (u32, u32),

    /// Whether the stream ever got as far as connecting.
    ///
    /// PipeWire reports `Unconnected` on the way *in* as well as on the way out, so the
    /// state alone cannot distinguish "not started yet" from "lost the node".
    reached_connection: bool,

    /// Shared with `run`, which outlives the callbacks and cannot otherwise see this.
    failed: Rc<Cell<bool>>,
}

pub struct PwCapture;

impl PwCapture {
    /// Capture `node_name` until `should_stop`, handing every frame to `on_frame`.
    ///
    /// Blocking: the PipeWire main loop drives itself and wants its own thread. Returns
    /// `Ok(())` when `should_stop` asked it to stop, and `Err` when the stream failed —
    /// which is the caller's cue to retry, because the daemon may simply not be running
    /// yet. The GUI treats `Err` as "try again in a couple of seconds".
    ///
    /// `on_frame` receives raw YUY2 bytes plus the negotiated width and height. It runs on
    /// the PipeWire thread inside the process callback, so it must be quick.
    pub fn run(
        node_name: &str,
        should_stop: impl Fn() -> bool + 'static,
        on_frame: impl FnMut(&[u8], u32, u32) + 'static,
    ) -> Result<(), PwCaptureError> {
        pipewire::init();

        let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
        let context = pipewire::context::ContextRc::new(&mainloop, None)?;
        let core = context.connect_rc(None)?;

        let stream = pipewire::stream::StreamRc::new(
            core.clone(),
            "cleanroom-preview",
            capture_props(node_name),
        )?;

        let failed = Rc::new(Cell::new(false));
        let _listener = register_capture_listener(&stream, &mainloop, failed.clone(), on_frame)?;

        let mut fmt_buf = Vec::new();
        let fmt = capture_format_param(&mut fmt_buf).ok_or(PwCaptureError::Format)?;

        // AUTOCONNECT is what makes the link happen at all; `node.dont-reconnect` in the
        // properties is what keeps it honest. See `capture_props`.
        stream.connect(
            libspa::utils::Direction::Input,
            None,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
            &mut [fmt],
        )?;

        // 50 ms, not the 200 ms the source node uses. This one is joined from the UI
        // thread when the preview window closes, and a fifth of a second of frozen window
        // is visible; a twentieth is not.
        let quit = mainloop.clone();
        let stop_timer = mainloop.loop_().add_timer(move |_| {
            if should_stop() {
                quit.quit();
            }
        });
        let tick = std::time::Duration::from_millis(50);
        stop_timer
            .update_timer(Some(tick), Some(tick))
            .into_result()
            .ok();

        tracing::info!(
            node = node_name,
            "capturing the PipeWire node for the preview"
        );
        mainloop.run();

        if failed.get() {
            return Err(PwCaptureError::Disconnected(node_name.to_string()));
        }
        tracing::info!(node = node_name, "PipeWire capture stopped");
        Ok(())
    }
}

/// The stream properties, of which the last two are the whole safety story.
///
/// `target.object` names the node to link to and `node.dont-reconnect` says what to do when
/// it is absent. Without them, PipeWire autoconnects the capture stream to the default
/// *physical* webcam when the target is missing — the preview would silently show, and
/// seize, the real camera. With them, an absent target is an error we can report and retry
/// instead of a wrong camera nobody asked for.
///
/// Built by hand rather than with the `properties!` macro, because syn cannot see into a
/// macro body and the complexity ratchet counts every one of them as code it could not
/// measure. This dictionary is the security-relevant part of the module; it should be the
/// last thing hidden from a tool.
fn capture_props(node_name: &str) -> PropertiesBox {
    let mut props = PropertiesBox::new();
    props.insert(*pipewire::keys::MEDIA_TYPE, "Video");
    props.insert(*pipewire::keys::MEDIA_CATEGORY, "Capture");
    props.insert(*pipewire::keys::MEDIA_ROLE, "Camera");
    props.insert(*pipewire::keys::TARGET_OBJECT, node_name);
    props.insert(*pipewire::keys::NODE_DONT_RECONNECT, "true");
    props
}

/// The `EnumFormat` offer: YUY2, at whatever size and rate the other end likes.
///
/// The mirror image of `pw_source::video_format_param`, which fixes everything
/// because it is the producer and knows. Here the size and framerate are `Choice::Range`
/// with deliberately wide bounds, so the daemon's geometry — which this process does not
/// know — is never the reason negotiation fails. The defaults inside the ranges are only a
/// hint for the case where the other end has no preference either.
fn capture_format_param(buffer: &mut Vec<u8>) -> Option<&Pod> {
    use libspa::param::format::{FormatProperties, MediaSubtype, MediaType};
    use libspa::pod::{ChoiceValue, Property, PropertyFlags, Value};
    use libspa::utils::{Choice, ChoiceEnum, ChoiceFlags, Fraction, Id, Rectangle};

    // Hand-built for the same reason the source side is: libspa has no
    // `From<VideoInfoRaw> for Vec<Property>`, and the `object!` macro cannot express a
    // property list assembled in Rust.
    let props = vec![
        Property {
            key: FormatProperties::MediaType.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(Id(MediaType::Video.as_raw())),
        },
        Property {
            key: FormatProperties::MediaSubtype.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(Id(MediaSubtype::Raw.as_raw())),
        },
        Property {
            key: FormatProperties::VideoFormat.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(Id(VideoFormat::YUY2.as_raw())),
        },
        Property {
            key: FormatProperties::VideoSize.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Choice(ChoiceValue::Rectangle(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Range {
                    default: Rectangle {
                        width: 1280,
                        height: 720,
                    },
                    min: Rectangle {
                        width: 1,
                        height: 1,
                    },
                    max: Rectangle {
                        width: 8192,
                        height: 8192,
                    },
                },
            ))),
        },
        Property {
            key: FormatProperties::VideoFramerate.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Choice(ChoiceValue::Fraction(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Range {
                    default: Fraction { num: 30, denom: 1 },
                    min: Fraction { num: 0, denom: 1 },
                    max: Fraction {
                        num: 1000,
                        denom: 1,
                    },
                },
            ))),
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

/// Wire up the three callbacks that matter: negotiation, frames, and death.
fn register_capture_listener(
    stream: &pipewire::stream::Stream,
    mainloop: &pipewire::main_loop::MainLoopRc,
    failed: Rc<Cell<bool>>,
    mut on_frame: impl FnMut(&[u8], u32, u32) + 'static,
) -> Result<StreamListener<CaptureState>, PwCaptureError> {
    let state = CaptureState {
        size: (0, 0),
        reached_connection: false,
        failed,
    };

    let quit = mainloop.clone();
    let listener = stream
        .add_local_listener_with_user_data(state)
        .state_changed(move |_, state, _old, new| {
            if !note_state_change(state, &new) {
                return;
            }
            tracing::warn!(state = ?new, "PipeWire capture stream dropped; asking for a retry");
            quit.quit();
        })
        .param_changed(|_, state, id, param| {
            if let Some(size) = negotiated_size(id, param) {
                tracing::info!("preview negotiated {}x{}", size.0, size.1);
                state.size = size;
            }
        })
        .process(move |stream, state| {
            let (w, h) = state.size;
            if w == 0 || h == 0 {
                return;
            }
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let Some(data) = buffer.datas_mut().first_mut() else {
                return;
            };
            let Some(bytes) = plane_bytes(data) else {
                return;
            };
            on_frame(bytes, w, h);
        })
        .register()?;
    Ok(listener)
}

/// Record a state transition, returning true when it means the stream is finished.
///
/// Split out of the callback so the "have we ever been connected?" rule can be tested
/// without a PipeWire daemon. `Unconnected` is reported both before the first connect and
/// after the target disappears, and only the second one is a failure — telling them apart
/// is the entire job here. Once the target node is gone, `node.dont-reconnect` guarantees
/// nothing else will be linked in its place, so ending the loop is the only correct move.
fn note_state_change(state: &mut CaptureState, new: &StreamState) -> bool {
    match new {
        StreamState::Connecting | StreamState::Paused | StreamState::Streaming => {
            state.reached_connection = true;
            false
        }
        StreamState::Error(_) | StreamState::Unconnected => {
            if !state.reached_connection {
                return false;
            }
            state.failed.set(true);
            true
        }
    }
}

/// The geometry out of a `Format` param, or `None` if this is not one we can use.
fn negotiated_size(id: u32, param: Option<&Pod>) -> Option<(u32, u32)> {
    use libspa::param::format::{MediaSubtype, MediaType};

    let param = param?;
    if id != libspa::param::ParamType::Format.as_raw() {
        return None;
    }
    let (media_type, media_subtype) = libspa::param::format_utils::parse_format(param).ok()?;
    if media_type != MediaType::Video || media_subtype != MediaSubtype::Raw {
        return None;
    }
    let mut info = libspa::param::video::VideoInfoRaw::default();
    info.parse(param).ok()?;
    let size = info.size();
    Some((size.width, size.height))
}

/// The valid bytes of a capture buffer's first plane.
///
/// The mapped region is `maxsize` long but only `[offset, offset + size)` of it was
/// written, and a producer is free to report either as zero. Every step is checked rather
/// than asserted: this runs in the realtime process callback, where a panic takes out the
/// PipeWire thread and the preview with it.
fn plane_bytes(data: &mut libspa::buffer::Data) -> Option<&[u8]> {
    let offset = data.chunk().offset() as usize;
    let size = data.chunk().size() as usize;
    let end = offset.checked_add(size)?;
    let mapped = data.data()?;
    mapped.get(offset..end).filter(|bytes| !bytes.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The consumer does not know the daemon's resolution, so a format that fixed one would
    /// negotiate successfully only by luck.
    #[test]
    fn the_capture_format_offers_yuy2_without_fixing_a_size() {
        use libspa::param::format::FormatProperties;
        use libspa::pod::{ChoiceValue, Value};
        use libspa::utils::{Choice, ChoiceEnum};

        let mut buf = Vec::new();
        let pod = capture_format_param(&mut buf);
        assert!(pod.is_some(), "the EnumFormat pod must serialize");
        assert!(!buf.is_empty(), "a serialized pod is never zero bytes");

        // Read the pod back rather than trusting the builder: the point of this format is
        // the shape of two of its properties, and "it serialized" does not check that.
        let Ok((_, Value::Object(obj))) =
            libspa::pod::deserialize::PodDeserializer::deserialize_any_from(&buf)
        else {
            panic!("the EnumFormat pod must deserialize back into an object");
        };
        let find = |key: FormatProperties| {
            obj.properties
                .iter()
                .find(|p| p.key == key.as_raw())
                .map(|p| p.value.clone())
        };

        assert_eq!(
            find(FormatProperties::VideoFormat),
            Some(Value::Id(libspa::utils::Id(VideoFormat::YUY2.as_raw()))),
            "the preview consumes the same YUY2 the daemon produces"
        );
        assert!(
            matches!(
                find(FormatProperties::VideoSize),
                Some(Value::Choice(ChoiceValue::Rectangle(Choice(
                    _,
                    ChoiceEnum::Range { .. }
                ))))
            ),
            "the size must stay a range: this end does not know the daemon's geometry"
        );
        assert!(
            matches!(
                find(FormatProperties::VideoFramerate),
                Some(Value::Choice(ChoiceValue::Fraction(Choice(
                    _,
                    ChoiceEnum::Range { .. }
                ))))
            ),
            "the framerate is the daemon's to choose too"
        );
    }

    /// Regression test for the physical-webcam hazard: without both of these properties,
    /// PipeWire happily autoconnects the preview to the user's real camera when the daemon
    /// node is missing, showing and seizing it with no error anywhere.
    #[test]
    fn capture_props_target_the_named_node_and_forbid_reconnecting_elsewhere() {
        let props = capture_props("cleanroom_cam");
        assert_eq!(
            props.get(*pipewire::keys::TARGET_OBJECT),
            Some("cleanroom_cam")
        );
        assert_eq!(
            props.get(*pipewire::keys::NODE_DONT_RECONNECT),
            Some("true")
        );
        assert_eq!(props.get(*pipewire::keys::MEDIA_CATEGORY), Some("Capture"));
        assert_eq!(props.get(*pipewire::keys::MEDIA_ROLE), Some("Camera"));
    }

    /// `Unconnected` before the first connect is normal; after one it means the node went
    /// away. Confusing the two either ends the loop before it starts or hangs forever.
    #[test]
    fn only_a_disconnect_after_connecting_counts_as_a_failure() {
        let failed = Rc::new(Cell::new(false));
        let mut state = CaptureState {
            size: (0, 0),
            reached_connection: false,
            failed: failed.clone(),
        };

        assert!(!note_state_change(&mut state, &StreamState::Unconnected));
        assert!(!failed.get(), "the initial Unconnected is not a failure");

        assert!(!note_state_change(&mut state, &StreamState::Connecting));
        assert!(!note_state_change(&mut state, &StreamState::Streaming));

        assert!(note_state_change(&mut state, &StreamState::Unconnected));
        assert!(
            failed.get(),
            "losing the node after connecting must fail the run"
        );
    }

    /// The contract `run` owes the GUI is that it *returns*: the retry loop cannot ask
    /// again if the previous attempt is still blocked in `mainloop.run()`. Whether the
    /// daemon node exists on this machine is not something a test can arrange, so both
    /// outcomes pass — hanging or panicking is the failure.
    #[test]
    fn a_capture_stream_reaches_a_running_pipewire_daemon() {
        let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR") else {
            eprintln!("no PipeWire socket; skipping");
            return;
        };
        if !std::path::Path::new(&runtime).join("pipewire-0").exists() {
            eprintln!("no PipeWire socket; skipping");
            return;
        }

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let frames = Rc::new(Cell::new(0u32));
        let counted = frames.clone();
        let result = PwCapture::run(
            cleanroom_core::node::VIRTUAL_CAM_NODE,
            move || std::time::Instant::now() >= deadline,
            move |bytes, w, h| {
                assert!(!bytes.is_empty() && w > 0 && h > 0);
                counted.set(counted.get() + 1);
            },
        );
        match result {
            Ok(()) => eprintln!("captured {} frames", frames.get()),
            Err(e) => eprintln!("capture ended with an error, which is also a return: {e}"),
        }
    }
}
