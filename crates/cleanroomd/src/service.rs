//! The D-Bus interface implementation.

use crate::doctor;
use crate::settings;
use crate::state::Shared;
use cleanroom_ipc::{DeviceInfo, Status};
use std::sync::Arc;
use zbus::interface;

pub struct Service {
    pub shared: Arc<Shared>,
}

#[interface(name = "io.github.perfectra1n.Cleanroom1")]
impl Service {
    async fn status(&self) -> Status {
        self.shared.status()
    }

    async fn list_cameras(&self) -> Vec<DeviceInfo> {
        // Only nodes that can actually capture, and never a virtual camera — offering
        // one would let a user point Cleanroom at its own output, or at OBS's.
        cleanroom_video::capture_devices()
            .into_iter()
            .map(|d| DeviceInfo {
                id: d.path.display().to_string(),
                description: d.card,
                available: d.accessible,
            })
            .collect()
    }

    async fn list_microphones(&self) -> Vec<DeviceInfo> {
        // Read from the PipeWire registry watcher rather than enumerating here: the
        // registry lives on the audio thread's main loop, and this is the async side.
        //
        // `available` is true for every entry. A PipeWire source is shareable — unlike a
        // V4L2 camera, several clients can read one microphone at once — so there is no
        // "in use by something else" state to report.
        self.shared
            .audio_registry
            .sources()
            .into_iter()
            .map(|s| DeviceInfo {
                id: s.name,
                description: s.description,
                available: true,
            })
            .collect()
    }

    /// Which autostart mechanism applies here, whether it is on, and anything the user
    /// has to do by hand. Returns (mechanism, instruction, enabled).
    async fn autostart(&self) -> (String, String, bool) {
        match zbus::Connection::session().await {
            Ok(c) => {
                let r = crate::autostart::status(&c).await;
                (r.mechanism.as_str().to_string(), r.instruction, r.enabled)
            }
            Err(e) => ("unknown".into(), e.to_string(), false),
        }
    }

    /// Turn autostart on or off. Returns (mechanism, instruction).
    ///
    /// The instruction is non-empty exactly when this session supports neither standard
    /// mechanism, in which case it holds the compositor line to paste. Returning it rather
    /// than silently succeeding is the point: a checkbox that claims to have done something
    /// it could not do is worse than one that explains itself.
    async fn set_autostart(&self, on: bool) -> zbus::fdo::Result<(String, String)> {
        let connection = zbus::Connection::session()
            .await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        let r = crate::autostart::set(&connection, on)
            .await
            .map_err(zbus::fdo::Error::Failed)?;
        Ok((r.mechanism.as_str().to_string(), r.instruction))
    }

    async fn get(&self, key: &str) -> zbus::fdo::Result<String> {
        settings::get(&self.shared.config(), key)
            .map_err(|e| zbus::fdo::Error::InvalidArgs(e.to_string()))
    }

    async fn set(&self, key: &str, value: &str) -> zbus::fdo::Result<()> {
        // A virtual camera stored as the *input* is a config the pipeline can never
        // satisfy: `Camera::open` fails, the daemon crash-loops on its 5-second backoff,
        // and nothing along that path names the actual mistake. Refuse here, where the
        // error reaches the person who typed the value — `ctl set` prints it and the GUI
        // surfaces it — rather than store a trap whose failure appears later and
        // elsewhere. The probe is a cheap non-blocking ioctl; see
        // [`refuse_virtual_camera`] for what passes through untouched.
        if key == "video.device"
            && !settings::is_clearing_word(value)
            && let Some(reason) =
                refuse_virtual_camera(&cleanroom_video::probe(std::path::Path::new(value)))
        {
            return Err(zbus::fdo::Error::InvalidArgs(reason));
        }

        // The config stores a PipeWire `node.name`, but what people see — in the GUI
        // picker, in `list-microphones` output — is the `node.description`. Accept either
        // here and store the name, because a description written into the config is a trap
        // that fails *silently*: `target.object` only matches names, so PipeWire falls
        // back to some other source while the status line goes on echoing the configured
        // string as if it were live.
        let value = if key == "audio.device" && !settings::is_clearing_word(value) {
            match resolve_microphone(value, &self.shared.audio_registry.sources()) {
                Some(name) => {
                    tracing::info!(
                        description = value,
                        node = %name,
                        "microphone chosen by description; storing its node name"
                    );
                    name
                }
                None => value.to_string(),
            }
        } else {
            value.to_string()
        };
        let value = value.as_str();

        let current = self.shared.config();
        let updated = settings::set(&current, key, value)
            .map_err(|e| zbus::fdo::Error::InvalidArgs(e.to_string()))?;

        // Persist failure is surfaced rather than swallowed, but the in-memory change
        // still stands: a read-only config directory should degrade to "works until
        // restart", not to "settings silently do nothing".
        if let Err(e) = self.shared.replace_config(updated) {
            tracing::error!(error = %e, "config change applied in memory but could not be saved");
            return Err(zbus::fdo::Error::IOError(format!(
                "setting applied to the running pipeline but NOT saved: {e}"
            )));
        }

        tracing::info!(key, value, "setting changed");
        Ok(())
    }

    async fn keys(&self) -> zbus::fdo::Result<Vec<(String, String)>> {
        settings::keys(&self.shared.config()).map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }

    async fn reload(&self) -> zbus::fdo::Result<()> {
        let (cfg, outcome) = cleanroom_core::persist::load(self.shared.paths())
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        tracing::info!(?outcome, "config reloaded from disk");
        self.shared
            .replace_config(cfg)
            .map_err(|e| zbus::fdo::Error::IOError(e.to_string()))?;
        Ok(())
    }

    async fn doctor(&self) -> Vec<String> {
        doctor::run(&self.shared.config(), &self.shared.rt_status())
            .into_iter()
            .map(|c| c.to_string())
            .collect()
    }

    async fn shutdown(&self) {
        tracing::info!("shutdown requested over D-Bus");
        self.shared.request_shutdown();
    }

    #[zbus(property)]
    async fn interface_version(&self) -> u32 {
        cleanroom_ipc::INTERFACE_VERSION
    }
}

/// Map a microphone named by its human-readable description to the `node.name` the
/// config must store. `None` means "store the value as given".
///
/// A value that already names a known node passes through untouched — even if it also
/// happens to equal some other node's description. A description shared by two devices
/// (two identical webcams) is left alone rather than guessed at: storing the wrong
/// sibling would be the same silent-wrong-device failure this exists to prevent, and the
/// unresolved string at least fails visibly.
fn resolve_microphone(value: &str, sources: &[cleanroom_audio::Source]) -> Option<String> {
    if sources.iter().any(|s| s.name == value) {
        return None;
    }
    let mut described = sources.iter().filter(|s| s.description == value);
    match (described.next(), described.next()) {
        (Some(only), None) => Some(only.name.clone()),
        _ => None,
    }
}

/// Why a `video.device` value must be refused outright, or `None` to store it as given.
///
/// Deliberately as restrained as [`resolve_microphone`]: only a node that *positively*
/// classifies as a virtual camera is refused. An unknown, unplugged or unopenable path
/// probes as not-virtual and passes through — rewriting or refusing it would destroy a
/// setting that becomes valid the moment the camera is plugged back in, the same
/// reasoning as storing an unknown microphone as given. Enumeration already refuses to
/// *offer* virtual nodes ([`Service::list_cameras`]); this closes the other door, a path
/// typed by hand — which is exactly how a v4l2loopback node at `/dev/video0` got stored
/// and cost a debugging session that ended at a healthy GPU.
fn refuse_virtual_camera(dev: &cleanroom_video::VideoDevice) -> Option<String> {
    if !dev.is_virtual {
        return None;
    }
    // Name the mechanism (driver, e.g. "v4l2 loopback") when the kernel reports one;
    // the card label is the fallback because the classification can come from either.
    let label = if dev.driver.is_empty() {
        dev.card.as_str()
    } else {
        dev.driver.as_str()
    };
    Some(format!(
        "{} is a virtual camera ({label}), not a capture device — Cleanroom would be \
         capturing its own (or another app's) output. Pick a real camera from \
         `cleanroom-ctl devices`.",
        dev.path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_audio::Source;

    fn sources() -> Vec<Source> {
        vec![
            Source {
                name: "alsa_input.usb-Logitech_A50-00.mono-fallback".into(),
                description: "A50 Mono".into(),
            },
            Source {
                name: "alsa_input.usb-Focusrite_Scarlett-00.HiFi__Mic1__source".into(),
                description: "Scarlett Solo (3rd Gen.) Input 1 Mic".into(),
            },
        ]
    }

    /// The bug this exists for: the GUI stored "Scarlett Solo (3rd Gen.) Input 1 Mic" in
    /// the config, `target.object` matched nothing, and the capture stream silently bound
    /// a webcam microphone while the status line went on claiming the Scarlett.
    #[test]
    fn a_description_resolves_to_its_node_name() {
        assert_eq!(
            resolve_microphone("Scarlett Solo (3rd Gen.) Input 1 Mic", &sources()),
            Some("alsa_input.usb-Focusrite_Scarlett-00.HiFi__Mic1__source".into())
        );
    }

    #[test]
    fn a_node_name_passes_through_untouched() {
        assert_eq!(
            resolve_microphone("alsa_input.usb-Logitech_A50-00.mono-fallback", &sources()),
            None
        );
    }

    /// A string matching nothing may be a device that is currently unplugged; rewriting
    /// it would destroy a setting that becomes valid again the moment it is plugged in.
    #[test]
    fn an_unknown_value_is_stored_as_given() {
        assert_eq!(
            resolve_microphone("alsa_input.not-plugged-in", &sources()),
            None
        );
        assert_eq!(resolve_microphone("", &[]), None);
    }

    /// Two identical devices share a description. Guessing between them silently is the
    /// same wrong-device failure this resolution exists to prevent, so leave it alone.
    #[test]
    fn an_ambiguous_description_is_not_guessed_at() {
        let mut s = sources();
        s.push(Source {
            name: "alsa_input.usb-Logitech_A50-01.mono-fallback".into(),
            description: "A50 Mono".into(),
        });
        assert_eq!(resolve_microphone("A50 Mono", &s), None);
    }

    use cleanroom_video::{NodeKind, VideoDevice};

    fn loopback() -> VideoDevice {
        VideoDevice {
            path: "/dev/video0".into(),
            card: "Cleanroom Camera".into(),
            driver: "v4l2 loopback".into(),
            kind: NodeKind::Output,
            is_virtual: true,
            accessible: true,
        }
    }

    /// The bug this exists for: video.device = /dev/video0 was a v4l2loopback node, the
    /// daemon crash-looped every 5 seconds on "reports no pixel format we can use", and
    /// the trail led to a healthy RTX 5090 instead of the config. The refusal must name
    /// what the device actually is, so the fix is obvious from the error alone.
    #[test]
    fn a_virtual_camera_is_refused_with_its_identity_named() {
        let reason = refuse_virtual_camera(&loopback()).expect("a loopback node must be refused");
        assert!(reason.contains("virtual camera"), "got: {reason}");
        assert!(reason.contains("v4l2 loopback"), "got: {reason}");
        assert!(reason.contains("/dev/video0"), "got: {reason}");
    }

    /// Our own sink is the worst value of all — storing it points the pipeline at its
    /// own output — and it must be refused even when a fork's driver string is
    /// unrecognised and only the card label gives it away.
    #[test]
    fn our_own_sink_is_refused_even_when_only_the_label_identifies_it() {
        let mut dev = loopback();
        dev.driver = String::new();
        let reason = refuse_virtual_camera(&dev).expect("our own sink must be refused");
        assert!(reason.contains("Cleanroom Camera"), "got: {reason}");
    }

    /// Same reasoning as `an_unknown_value_is_stored_as_given` for microphones: an
    /// unplugged camera probes as nothing in particular, and refusing it would destroy a
    /// setting that becomes valid again the moment it is plugged back in.
    #[test]
    fn an_unplugged_or_unknown_video_device_is_stored_as_given() {
        let dev = VideoDevice {
            path: "/dev/video-not-plugged-in".into(),
            card: String::new(),
            driver: String::new(),
            kind: NodeKind::Other,
            is_virtual: false,
            accessible: false,
        };
        assert_eq!(refuse_virtual_camera(&dev), None);
    }

    /// A real capture node is the value this setting exists for.
    #[test]
    fn a_real_capture_device_passes_the_virtual_guard() {
        let dev = VideoDevice {
            path: "/dev/video2".into(),
            card: "C922 Pro Stream Webcam".into(),
            driver: "uvcvideo".into(),
            kind: NodeKind::Capture,
            is_virtual: false,
            accessible: true,
        };
        assert_eq!(refuse_virtual_camera(&dev), None);
    }
}
