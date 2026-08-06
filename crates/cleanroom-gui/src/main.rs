//! Cleanroom's control panel and tray icon.
//!
//! A thin client over the daemon's D-Bus interface. It holds no state of its own beyond
//! what it is displaying: closing it does not stop the effects, and two copies of it
//! cannot disagree about anything.
//!
//! ## Threading, and a feature-unification trap
//!
//! A tokio runtime is created in `main` and its `EnterGuard` is held for the life of the
//! process. That is not incidental — without it the GUI panics before showing a window:
//!
//!   thread 'main' panicked at zbus/src/abstractions/executor.rs:
//!   there is no reactor running, must be called from the context of a Tokio 1.x runtime
//!
//! The cause is cargo feature unification. `cleanroomd` and `cleanroom-ipc` depend on
//! `zbus` with `features = ["tokio"]`, which switches zbus's executor to tokio *globally*
//! for the whole build graph. Slint pulls zbus in twice on its own account —
//! `i-slint-backend-winit` directly, and `accesskit_unix` for AT-SPI accessibility — and
//! both run on the main thread during startup, where no runtime existed. The tray and the
//! accessibility bridge were the callers, not our code.
//!
//! Holding an enter guard on the main thread makes a runtime visible to all of them.
//! Polling still happens on a spawned task rather than by blocking the UI thread, so a slow
//! or absent daemon cannot stall the interface.
//!
//! ## What the tray can and cannot be relied on for
//!
//! Whether a tray icon appears is entirely up to the desktop. GNOME ships no
//! StatusNotifierItem host — the AppIndicator extension is third-party, and the official
//! Status Icons extension is XEmbed-only and explicitly will not support SNI. A bare
//! wlroots session often has no host either, since the host is whichever bar the user runs
//! and a minimal setup runs none.
//!
//! So what closing the window does depends on whether a tray was actually registered:
//!
//! * With a tray, the window hides and the preview is stopped on the spot. A registered
//!   tray keeps Slint's event loop alive after the last window closes, so without that
//!   explicit stop the process would linger invisibly with a live capture stream — which is
//!   precisely the "something still has my camera" bug this is written to avoid.
//! * With no tray, there is nothing left to reopen the window from, so closing quits.
//!
//! Every action in the tray menu is also reachable from `cleanroom-ctl`, because a tray that
//! never appears must not be the only way to do anything.

mod filechooser;
mod preview;

use anyhow::{Context, Result};
use cleanroom_ipc::{CleanroomProxy, Health, Status};
use slint::ComponentHandle;
use tokio::sync::mpsc;

slint::include_modules!();

/// Must match the `.desktop` basename byte for byte.
///
/// Wayland has no `WM_CLASS`; `xdg_toplevel.set_app_id` is the only identity signal a
/// compositor gets, and it looks up `<app_id>.desktop` in `$XDG_DATA_DIRS/applications`
/// for the icon. Slint sets no app_id by default — verified, `hyprctl clients` reports
/// `class: ''` — so omitting this is the same as getting it wrong: a generic placeholder
/// in every taskbar, dock and alt-tab.
const APP_ID: &str = "io.github.perfectra1n.Cleanroom";

/// The one disconnect message that actually means "start the daemon". The version
/// mismatch case gets its own text from `cleanroom_ipc::version_mismatch_message`.
const NOT_RUNNING_MSG: &str = "Not connected to the Cleanroom daemon. Start it with: cleanroomd";

/// A setting change requested by the UI.
struct SetRequest {
    key: &'static str,
    value: String,
}

/// Everything the UI needs from one poll, already flattened.
struct Snapshot {
    status: Status,
    background: i32,
    blur: f32,
    mirror: bool,
    denoise: bool,
    attenuation: f32,
    desaturate: f32,
    dim: f32,
    tighten: f32,
    feather: f32,
    fade_rise: f32,
    fade_fall: f32,
    motion_release: f32,
    guided_filter: bool,
    guided_radius: f32,
    matting_backend: i32,
    background_image: String,
    /// The configured `audio.device` — a PipeWire `node.name`, or `(unset)`. Fetched on
    /// every poll, unlike the device *list*, so the picker highlight tracks a change made
    /// from the CLI within half a second rather than within ten.
    audio_device: String,
    /// `None` when this poll skipped the expensive parts (see `DEVICE_POLL_EVERY`).
    devices: Option<Devices>,
}

/// Device lists and autostart state, refreshed on a slower cadence than the rest.
///
/// Enumerating cameras opens every `/dev/video*` to query its capabilities, and the
/// autostart check talks to the systemd user manager. Neither changes on the timescale a
/// status poll runs at, and doing them twice a second would be a lot of syscalls to learn
/// nothing.
#[derive(Clone, Default)]
struct Devices {
    cameras: Vec<(String, String)>,
    microphones: Vec<(String, String)>,
    autostart_on: bool,
    autostart_mechanism: String,
    autostart_instruction: String,
}

/// Poll the device lists every Nth status poll. 500 ms x 20 = every 10 s.
const DEVICE_POLL_EVERY: u32 = 20;

fn health_code(h: &Health) -> i32 {
    match h {
        Health::Nominal => 0,
        Health::Degraded => 1,
        Health::Failed => 2,
        Health::Idle => 3,
    }
}

fn mode_name(i: i32) -> &'static str {
    match i {
        0 => "off",
        2 => "replace",
        3 => "remove",
        _ => "blur",
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    // Multi-threaded so the reactor runs independently of the UI thread, and created
    // *before* any Slint object: Slint's winit backend and accessibility bridge both touch
    // zbus during construction and need a runtime already visible. The guard is held for
    // the life of the process — see the module docs for why.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("starting the async runtime")?;
    let _rt_guard = rt.enter();

    // Order matters, and the obvious reading of the docs is wrong. The app_id must be set
    // before the window is *shown*, but NOT before it is *created*: the Slint platform is
    // initialised lazily by the first window, and `set_xdg_app_id` explicitly refuses to
    // initialise one itself, failing with "No default Slint platform was selected".
    let ui = AppWindow::new()?;
    slint::set_xdg_app_id(APP_ID)?;

    // Best-effort by design: a desktop with no SNI host is a normal configuration.
    let tray = match Tray::new() {
        Ok(t) => {
            tracing::info!("system tray icon registered");
            Some(t)
        }
        Err(e) => {
            tracing::info!(
                error = %e,
                "no system tray available — normal on GNOME and on bare wlroots sessions. \
                 Everything here is also available from `cleanroom-ctl`."
            );
            None
        }
    };

    let (set_tx, set_rx) = mpsc::unbounded_channel::<SetRequest>();
    rt.spawn(dbus_loop(ui.as_weak(), set_rx));

    wire_controls(&ui, set_tx.clone());

    if let Some(t) = &tray {
        wire_tray(t, &ui, set_tx);
    }
    wire_close(&ui, tray.is_some());

    ui.show()?;
    slint::run_event_loop()?;
    Ok(())
}

/// Poll the daemon and apply UI requests. Runs as a task on the shared runtime.
async fn dbus_loop(ui: slint::Weak<AppWindow>, mut set_rx: mpsc::UnboundedReceiver<SetRequest>) {
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(500));
    let mut ticks: u32 = 0;
    loop {
        tokio::select! {
            // Applied immediately rather than on the next tick, so a control feels instant.
            Some(req) = set_rx.recv() => {
                if let Err(e) = apply(&req).await {
                    // Surfaced, not swallowed: a control that silently does nothing is the
                    // most confusing failure a UI can have.
                    tracing::error!(key = req.key, error = %e, "could not apply setting");
                }
            }
            _ = tick.tick() => {
                // Device enumeration and the autostart check ride a slower cadence; the
                // first poll includes them so the pickers are populated immediately.
                let with_devices = ticks.is_multiple_of(DEVICE_POLL_EVERY);
                ticks = ticks.wrapping_add(1);
                let outcome = poll_outcome(with_devices).await;
                let ui = ui.clone();
                // Touching the UI from another thread is not allowed; this hands the
                // update to the Slint event loop.
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui.upgrade() {
                        match outcome {
                            PollOutcome::Snapshot(s) => apply_snapshot(&ui, *s),
                            PollOutcome::Mismatch(msg) => {
                                ui.set_connected(false);
                                ui.set_connect_error(msg.into());
                            }
                            PollOutcome::Unreachable => {
                                ui.set_connected(false);
                                ui.set_connect_error(NOT_RUNNING_MSG.into());
                            }
                        }
                    }
                });
            }
        }
    }
}

/// Connect fresh each time.
///
/// Deliberate: a daemon started *after* the GUI is picked up automatically, and a daemon
/// that restarts does not leave the GUI holding a dead connection. At twice a second the
/// cost is irrelevant next to the robustness.
async fn proxy() -> zbus::Result<CleanroomProxy<'static>> {
    let conn = zbus::Connection::session().await?;
    CleanroomProxy::new(&conn).await
}

/// What one poll learned, with the two failure shapes kept apart.
///
/// Collapsing them into one was the original sin here: a `Status` whose D-Bus signature
/// this build can't decode used to read as "daemon not running", and the banner told the
/// user to start a daemon that was already up. `InterfaceVersion` is a bare `u32`, so it
/// decodes across every client generation and cleanly separates the two cases.
enum PollOutcome {
    Snapshot(Box<Snapshot>),
    /// Daemon reachable but incompatible; carries the shared "update me" message.
    Mismatch(String),
    Unreachable,
}

async fn poll_outcome(with_devices: bool) -> PollOutcome {
    let Ok(p) = proxy().await else {
        return PollOutcome::Unreachable;
    };
    // The property read doubles as the liveness probe: it is the first call on the
    // fresh connection, so a missing daemon fails here, before the version check.
    let Ok(daemon_version) = p.interface_version().await else {
        return PollOutcome::Unreachable;
    };
    if let Some(msg) = cleanroom_ipc::version_mismatch_message(daemon_version) {
        return PollOutcome::Mismatch(msg);
    }
    match poll(p, with_devices).await {
        Some(s) => PollOutcome::Snapshot(Box::new(s)),
        // Versions agree, so a decode failure can only mean the daemon went away
        // between the probe and the poll.
        None => PollOutcome::Unreachable,
    }
}

async fn poll(p: CleanroomProxy<'static>, with_devices: bool) -> Option<Snapshot> {
    let devices = if with_devices {
        let cameras = p
            .list_cameras()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|d| (d.id, d.description))
            .collect();
        let microphones = p
            .list_microphones()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|d| (d.id, d.description))
            .collect();
        let (mechanism, instruction, on) = p
            .autostart()
            .await
            .unwrap_or_else(|_| ("unknown".into(), String::new(), false));
        Some(Devices {
            cameras,
            microphones,
            autostart_on: on,
            autostart_mechanism: mechanism,
            autostart_instruction: instruction,
        })
    } else {
        None
    };

    Some(Snapshot {
        devices,
        background_image: p.get("video.background_image").await.unwrap_or_default(),
        audio_device: p.get("audio.device").await.unwrap_or_default(),
        status: p.status().await.ok()?,
        background: match p.get("video.background").await.ok()?.as_str() {
            "off" => 0,
            "replace" => 2,
            "remove" => 3,
            _ => 1,
        },
        blur: p
            .get("video.blur_strength")
            .await
            .ok()?
            .parse()
            .unwrap_or(0.6),
        mirror: p.get("video.mirror").await.ok()? == "true",
        denoise: p.get("audio.denoise.enabled").await.ok()? == "true",
        attenuation: p
            .get("audio.denoise.attenuation_db")
            .await
            .ok()?
            .parse()
            .unwrap_or(40.0),
        desaturate: p
            .get("video.background_desaturate")
            .await
            .ok()?
            .parse()
            .unwrap_or(0.0),
        dim: p
            .get("video.background_dim")
            .await
            .ok()?
            .parse()
            .unwrap_or(0.0),
        // `(unset)` is the honest answer for an optional that is deriving its value per
        // mode, and it must read as 0 on the slider rather than as a parse failure.
        tighten: p
            .get("video.matte_tighten")
            .await
            .ok()?
            .parse()
            .unwrap_or(0.0),
        feather: p
            .get("video.matte_feather")
            .await
            .ok()?
            .parse()
            .unwrap_or(0.0),
        fade_rise: p
            .get("video.matte_fade_rise")
            .await
            .ok()?
            .parse()
            .unwrap_or(0.55),
        fade_fall: p
            .get("video.matte_fade_fall")
            .await
            .ok()?
            .parse()
            .unwrap_or(0.22),
        motion_release: p
            .get("video.matte_motion_release")
            .await
            .ok()?
            .parse()
            .unwrap_or(0.25),
        guided_filter: p.get("video.guided_filter").await.ok()? == "true",
        guided_radius: p
            .get("video.guided_radius")
            .await
            .ok()?
            .parse()
            .unwrap_or(3.0),
        matting_backend: match p.get("video.matting_backend").await.ok()?.as_str() {
            "gpu" => 1,
            "cpu" => 2,
            _ => 0,
        },
    })
}

async fn apply(req: &SetRequest) -> Result<()> {
    let p = proxy().await.context("reaching the daemon")?;
    p.set(req.key, &req.value)
        .await
        .context("setting the value")?;
    Ok(())
}

/// The camera path the daemon reports it is using, parsed out of the health detail.
///
/// The detail is formatted "<camera> -> <vcam> (<mode>)", so the left-hand side is the
/// device in use. Read from there rather than from config because config may say `None`,
/// meaning "first usable camera", and the picker needs to highlight the concrete one.
fn current_camera_id(status: &Status) -> String {
    status
        .video_detail
        .split(" -> ")
        .next()
        .unwrap_or_default()
        .to_string()
}

thread_local! {
    /// Owns the preview thread, alongside the PipeWire node name it was started on.
    ///
    /// The name is kept because the node can change under us — the daemon can be
    /// reconfigured, or the source can come back under a different name — and a preview
    /// still linked to the previous one would sit there showing nothing with no way to
    /// notice. Comparing names is how a restart is detected.
    ///
    /// Thread-local because it can only be touched from the UI thread, which is the only
    /// place `apply_snapshot` and the close handler run.
    static PREVIEW_SLOT: std::cell::RefCell<Option<(String, preview::Preview)>> =
        const { std::cell::RefCell::new(None) };
}

/// The preview runs only while there is both a window to show it in and a node to read.
///
/// Visibility is half the rule because the preview is a real PipeWire consumer: a hidden
/// window that kept streaming would hold the camera awake with nobody looking at it, which
/// is indistinguishable from the daemon having gone rogue.
fn preview_should_run(window_visible: bool, pw_node: &str) -> bool {
    window_visible && !pw_node.is_empty()
}

/// What to say in the empty preview area, which is two quite different situations.
///
/// An empty `pw_node` is the daemon telling us the PipeWire source is off or has failed;
/// no amount of waiting will produce a frame, and saying "waiting…" forever would be a
/// lie. There is deliberately no fallback to the v4l2 device: that would take the single
/// capture slot the meeting apps need.
fn preview_hint(pw_node: &str) -> &'static str {
    if pw_node.is_empty() {
        "preview unavailable — the PipeWire camera source is off or failed; \
         check `cleanroom-ctl status`"
    } else {
        "waiting for the virtual camera…"
    }
}

/// What closing the window should do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloseAction {
    /// A tray icon exists, so the window can be brought back; hide it.
    HideToTray,
    /// Nothing would be left to reopen it from, so end the process.
    Quit,
}

/// Hiding is only honest when there is a tray to hide *into*.
fn close_action(have_tray: bool) -> CloseAction {
    if have_tray {
        CloseAction::HideToTray
    } else {
        CloseAction::Quit
    }
}

/// The banner shown when something outside Cleanroom is reading the virtual camera.
///
/// `consumers` is the authoritative count — it comes from the v4l2 device itself — while
/// `holders` is a best-effort naming that a sandboxed reader is simply invisible to. So the
/// count always decides the wording and the names only decorate it: "2 apps" with one name
/// listed is honest, "1 app" because only one could be named would not be.
///
/// The preview is not in this count. It consumes the PipeWire node, so everything reported
/// here is somebody else.
fn holder_banner_text(consumers: u32, holders: &[String]) -> String {
    match (consumers, holders) {
        (0, _) => String::new(),
        (1, [only]) => format!("{only} is using the virtual camera"),
        (1, _) => "1 app is using the virtual camera".to_string(),
        (n, []) => format!("{n} apps are using the virtual camera"),
        (n, named) => format!(
            "{n} apps are using the virtual camera: {}",
            named.join(", ")
        ),
    }
}

/// Start or stop the preview so it matches the window and the node the daemon publishes.
///
/// The 500 ms status poll is the single lifecycle mechanism here, deliberately: there is no
/// second timer and no visibility callback to disagree with it. Hiding the window therefore
/// stops the preview on the next tick and — because the same rule is re-evaluated every
/// tick — keeps it stopped. Closing is handled separately by `stop_preview_now`, so the
/// stream is released at once rather than up to half a second later.
fn sync_preview(ui: &AppWindow, pw_node: &str) {
    ui.set_preview_hint(preview_hint(pw_node).into());
    let run = preview_should_run(ui.window().is_visible(), pw_node);

    PREVIEW_SLOT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if !run {
            if slot.take().is_some() {
                ui.set_preview_running(false);
            }
            return;
        }
        if slot.as_ref().is_some_and(|(node, _)| node == pw_node) {
            return;
        }
        // Assigning None first joins any thread still on the previous node, so two capture
        // streams never overlap.
        *slot = None;
        tracing::info!(node = pw_node, "starting preview");
        *slot = Some((pw_node.to_string(), start_preview(ui, pw_node)));
    });
}

/// Stop the preview immediately and say so in the UI.
///
/// Called from the close handler: a tray keeps the event loop alive after the window goes,
/// and a headless process holding a camera stream is the exact failure this whole path
/// exists to prevent.
fn stop_preview_now(ui: &AppWindow) {
    PREVIEW_SLOT.with(|slot| *slot.borrow_mut() = None);
    ui.set_preview_running(false);
}

/// The frame sink: pixels in on a worker thread, a `slint::Image` out on the UI thread.
fn start_preview(ui: &AppWindow, pw_node: &str) -> preview::Preview {
    let weak = ui.as_weak();
    preview::Preview::start(pw_node.to_string(), move |rgb, pw, ph| {
        if let Some(ui) = weak.upgrade() {
            let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(&rgb, pw, ph);
            ui.set_preview_frame(slint::Image::from_rgb8(buf));
            ui.set_preview_running(true);
        }
    })
}

/// Closing the window must not leave a capture stream running behind it.
///
/// Slint 1.17 offers no "close and quit" response — the only choices are hiding the window
/// or refusing to close it — so quitting is done by calling `quit_event_loop` from inside
/// the handler and hiding anyway.
fn wire_close(ui: &AppWindow, have_tray: bool) {
    let weak = ui.as_weak();
    ui.window().on_close_requested(move || {
        if let Some(ui) = weak.upgrade() {
            stop_preview_now(&ui);
        }
        if close_action(have_tray) == CloseAction::Quit {
            let _ = slint::quit_event_loop();
        }
        slint::CloseRequestResponse::HideWindow
    });
}

fn apply_snapshot(ui: &AppWindow, s: Snapshot) {
    ui.set_connected(true);
    // Both taken before the status fields are moved into the UI below.
    let active_camera = current_camera_id(&s.status);
    let pw_node = s.status.pw_node.clone();

    ui.set_video_health(health_code(&s.status.video_health));
    ui.set_video_detail(s.status.video_detail.into());
    ui.set_audio_health(health_code(&s.status.audio_health));
    ui.set_audio_detail(s.status.audio_detail.into());
    ui.set_gpu_adapter(s.status.gpu_adapter.into());
    ui.set_matting_detail(s.status.matting_engine.into());
    ui.set_vcam_path(if s.status.vcam_path.is_empty() {
        "—".into()
    } else {
        s.status.vcam_path.into()
    });

    let st = &s.status.stats;
    ui.set_fps(st.fps as f32);
    ui.set_decode_ms(st.decode_ms as f32);
    ui.set_gpu_ms(st.gpu_ms as f32);
    ui.set_matting_ms(st.matting_ms as f32);
    ui.set_dropped(st.dropped.min(i32::MAX as u64) as i32);
    ui.set_matte_rejected(st.matte_rejected.min(i32::MAX as u64) as i32);
    ui.set_consumers(st.vcam_consumers as i32);
    ui.set_mic_in_db(st.mic_level_db);
    ui.set_mic_out_db(st.mic_level_out_db);
    ui.set_vcam_holder_banner(holder_banner_text(st.vcam_consumers, &s.status.vcam_holders).into());

    ui.set_background_image(s.background_image.clone().into());
    sync_preview(ui, &pw_node);

    // Only when this poll actually fetched them. Writing empty lists on the intervening
    // polls would make the pickers flicker empty nineteen times out of twenty.
    if let Some(d) = &s.devices {
        let cam_names: Vec<slint::SharedString> = d
            .cameras
            .iter()
            .map(|(id, desc)| format!("{desc}  ({id})").into())
            .collect();
        let cam_idx = d
            .cameras
            .iter()
            .position(|(id, _)| *id == active_camera)
            .map(|i| i as i32)
            .unwrap_or(-1);
        ui.set_camera_names(slint::ModelRc::new(slint::VecModel::from(cam_names)));
        ui.set_camera_index(cam_idx);

        let mic_names: Vec<slint::SharedString> =
            d.microphones.iter().map(|(_, desc)| desc.into()).collect();
        let mic_ids: Vec<slint::SharedString> =
            d.microphones.iter().map(|(id, _)| id.into()).collect();
        ui.set_microphone_names(slint::ModelRc::new(slint::VecModel::from(mic_names)));
        ui.set_microphone_ids(slint::ModelRc::new(slint::VecModel::from(mic_ids)));

        ui.set_autostart_on(d.autostart_on);
        ui.set_autostart_mechanism(d.autostart_mechanism.clone().into());
        ui.set_autostart_instruction(d.autostart_instruction.clone().into());
    }

    // Every poll, not only the device-list ones: the ids model persists between list
    // refreshes, so the highlight can track a config change made from the CLI at the
    // status cadence. An unmatched value — `(unset)`, or a device currently unplugged —
    // reads as no selection rather than as a wrong one.
    {
        use slint::Model;
        let ids = ui.get_microphone_ids();
        ui.set_microphone_index(mic_index(
            ids.iter().map(|id| id.to_string()),
            &s.audio_device,
        ));
    }

    // Reflect the daemon's values back, so a change made from the CLI or a second GUI
    // shows up here rather than leaving the two silently disagreeing.
    ui.set_background_mode(s.background);
    ui.set_blur_strength(s.blur);
    ui.set_mirror(s.mirror);
    ui.set_denoise(s.denoise);
    ui.set_attenuation(s.attenuation);
    ui.set_desaturate(s.desaturate);
    ui.set_dim(s.dim);
    ui.set_tighten(s.tighten);
    ui.set_feather(s.feather);
    ui.set_fade_rise(s.fade_rise);
    ui.set_fade_fall(s.fade_fall);
    ui.set_motion_release(s.motion_release);
    ui.set_guided_filter(s.guided_filter);
    ui.set_guided_radius(s.guided_radius);
    ui.set_matting_backend(s.matting_backend);
}

fn wire_controls(ui: &AppWindow, tx: mpsc::UnboundedSender<SetRequest>) {
    {
        // The ComboBox hands back the label it displays, so the device id is recovered from
        // the "(…)" suffix the label was built with rather than by index — an index would
        // be wrong the moment the list is re-sorted or a camera is unplugged mid-session.
        let tx = tx.clone();
        ui.on_set_camera(move |label| {
            let label = label.to_string();
            let id = label
                .rsplit_once('(')
                .map(|(_, rest)| rest.trim_end_matches(')').to_string())
                .unwrap_or(label);
            let _ = tx.send(SetRequest {
                key: "video.device",
                value: id,
            });
        });
    }
    {
        // The picker displays descriptions but the config stores the PipeWire node.name,
        // so the selection arrives as an index into the parallel ids model. Sending the
        // description was the original bug here: `target.object` matches names only, so
        // a description in the config made PipeWire silently bind some other microphone
        // while the status line echoed the configured string as if it were live.
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_set_microphone(move |index| {
            let Some(ui) = ui_weak.upgrade() else { return };
            use slint::Model;
            let Some(id) = usize::try_from(index)
                .ok()
                .and_then(|i| ui.get_microphone_ids().row_data(i))
            else {
                return;
            };
            let _ = tx.send(SetRequest {
                key: "audio.device",
                value: id.to_string(),
            });
        });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_choose_background_image(move || {
            let tx = tx.clone();
            let ui_weak = ui_weak.clone();
            // Spawned rather than awaited: the portal call blocks until the user picks a
            // file, and doing that on the UI thread would freeze the window behind the
            // dialog it just opened.
            tokio::spawn(async move {
                match filechooser::pick_image().await {
                    Ok(Some(path)) => {
                        let _ = tx.send(SetRequest {
                            key: "video.background_image",
                            value: path,
                        });
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // A bare wlroots session may have no FileChooser portal at all.
                        // Say so, and point at the path that always works.
                        tracing::warn!(
                            error = %e,
                            "no file dialog available — set it with: \
                             cleanroom-ctl set video.background_image /path/to/image.png"
                        );
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = ui_weak.upgrade() {
                                ui.set_autostart_instruction(
                                    "No file dialog is available on this desktop. Set it with:\n    \
                                     cleanroom-ctl set video.background_image /path/to/image.png"
                                        .into(),
                                );
                            }
                        });
                    }
                }
            });
        });
    }
    {
        let tx = tx.clone();
        ui.on_clear_background_image(move || {
            // The daemon's settings layer treats these words as "unset" for an optional.
            let _ = tx.send(SetRequest {
                key: "video.background_image",
                value: "unset".into(),
            });
        });
    }
    {
        let ui_weak = ui.as_weak();
        ui.on_set_autostart(move |on| {
            let ui_weak = ui_weak.clone();
            tokio::spawn(async move {
                let result = async {
                    let p = proxy().await.ok()?;
                    p.set_autostart(on).await.ok()
                }
                .await;
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui_weak.upgrade() {
                        match result {
                            Some((mechanism, instruction)) => {
                                ui.set_autostart_mechanism(mechanism.into());
                                ui.set_autostart_instruction(instruction.clone().into());
                                // Only claim it is on if the daemon actually did it.
                                ui.set_autostart_on(on && instruction.is_empty());
                            }
                            None => ui.set_autostart_on(!on),
                        }
                    }
                });
            });
        });
    }
    {
        // Declared in the .slint since the beginning and never connected to anything.
        ui.on_quit_daemon(move || {
            tokio::spawn(async move {
                if let Ok(p) = proxy().await {
                    let _ = p.shutdown().await;
                }
            });
        });
    }

    let send = move |key: &'static str, value: String| {
        let _ = tx.send(SetRequest { key, value });
    };

    let s = send.clone();
    ui.on_set_background(move |i| s("video.background", mode_name(i).to_string()));
    let s = send.clone();
    ui.on_set_blur(move |v| s("video.blur_strength", format!("{v}")));
    let s = send.clone();
    ui.on_set_mirror(move |b| s("video.mirror", b.to_string()));
    let s = send.clone();
    ui.on_set_denoise(move |b| s("audio.denoise.enabled", b.to_string()));
    let s = send.clone();
    ui.on_set_attenuation(move |v| s("audio.denoise.attenuation_db", format!("{}", v.round())));

    let s = send.clone();
    ui.on_set_desaturate(move |v| s("video.background_desaturate", format!("{v}")));
    let s = send.clone();
    ui.on_set_dim(move |v| s("video.background_dim", format!("{v}")));
    let s = send.clone();
    ui.on_set_tighten(move |v| s("video.matte_tighten", format!("{v}")));
    let s = send.clone();
    ui.on_set_feather(move |v| s("video.matte_feather", format!("{v}")));
    let s = send.clone();
    ui.on_set_fade_rise(move |v| s("video.matte_fade_rise", format!("{v}")));
    let s = send.clone();
    ui.on_set_fade_fall(move |v| s("video.matte_fade_fall", format!("{v}")));
    let s = send.clone();
    ui.on_set_motion_release(move |v| s("video.matte_motion_release", format!("{v}")));
    let s = send.clone();
    ui.on_set_guided_filter(move |b| s("video.guided_filter", b.to_string()));
    let s = send.clone();
    ui.on_set_guided_radius(move |v| s("video.guided_radius", format!("{}", v.round() as u32)));
    let s = send.clone();
    ui.on_set_matting_backend(move |i| s("video.matting_backend", backend_name(i).to_string()));
}

/// Which row of the microphone picker the configured device id occupies, or -1 for none.
///
/// -1 is the honest answer for `(unset)` and for a device that is not currently plugged
/// in; highlighting row 0 instead is exactly the "shows A50 Mono while capturing the
/// Scarlett" confusion this exists to prevent.
fn mic_index(ids: impl Iterator<Item = String>, device: &str) -> i32 {
    ids.enumerate()
        .find(|(_, id)| id == device)
        .map(|(i, _)| i as i32)
        .unwrap_or(-1)
}

/// Combo index to the config value. Order matches the `model` in `app.slint`.
fn backend_name(i: i32) -> &'static str {
    match i {
        1 => "gpu",
        2 => "cpu",
        _ => "auto",
    }
}

fn wire_tray(tray: &Tray, ui: &AppWindow, tx: mpsc::UnboundedSender<SetRequest>) {
    let send = move |key: &'static str, value: String| {
        let _ = tx.send(SetRequest { key, value });
    };

    {
        let weak = ui.as_weak();
        tray.on_show_window(move || {
            if let Some(ui) = weak.upgrade() {
                let _ = ui.show();
            }
        });
    }
    {
        let weak = ui.as_weak();
        let s = send.clone();
        tray.on_toggle_blur(move || {
            let Some(ui) = weak.upgrade() else { return };
            // Toggle between blur and off, leaving replace/green alone: someone using a
            // replacement background does not want the tray silently swapping it for blur.
            let next = if ui.get_background_mode() == 1 { 0 } else { 1 };
            ui.set_background_mode(next);
            s("video.background", mode_name(next).to_string());
        });
    }
    {
        let weak = ui.as_weak();
        let s = send.clone();
        tray.on_toggle_denoise(move || {
            let Some(ui) = weak.upgrade() else { return };
            let next = !ui.get_denoise();
            ui.set_denoise(next);
            s("audio.denoise.enabled", next.to_string());
        });
    }
    tray.on_quit(|| {
        let _ = slint::quit_event_loop();
    });

    // Keep the menu labels in step with the real state, so the tray never claims blur is
    // on while the window shows it off.
    let sync = Box::leak(Box::new(slint::Timer::default()));
    let weak_ui = ui.as_weak();
    let weak_tray = tray.as_weak();
    sync.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(500),
        move || {
            let (Some(ui), Some(tray)) = (weak_ui.upgrade(), weak_tray.upgrade()) else {
                return;
            };
            tray.set_blur_on(ui.get_background_mode() == 1);
            tray.set_denoise_on(ui.get_denoise());
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hiding into a tray that does not exist is how a GUI becomes unreachable: the window
    /// is gone, nothing in the panel brings it back, and the process is still running.
    #[test]
    fn closing_with_a_tray_hides_and_closing_without_one_quits() {
        assert_eq!(close_action(true), CloseAction::HideToTray);
        assert_eq!(close_action(false), CloseAction::Quit);
    }

    /// The preview is a real PipeWire consumer, so both halves of the rule matter: no node
    /// means nothing to read, and no visible window means nobody is looking.
    #[test]
    fn the_preview_runs_only_while_the_window_is_visible_and_a_node_is_published() {
        assert!(preview_should_run(true, "cleanroom_cam"));
        assert!(!preview_should_run(false, "cleanroom_cam"));
        assert!(!preview_should_run(true, ""));
        assert!(!preview_should_run(false, ""));
    }

    /// "Waiting…" against a source that will never publish is a lie the user cannot see
    /// through — it looks like a slow start rather than a disabled transport.
    #[test]
    fn an_empty_pw_node_shows_the_disabled_hint_rather_than_the_waiting_one() {
        assert!(preview_hint("cleanroom_cam").starts_with("waiting"));
        let off = preview_hint("");
        assert!(off.starts_with("preview unavailable"), "got {off}");
        assert!(
            off.contains("cleanroom-ctl status"),
            "the hint must say where to look next, got {off}"
        );
    }

    /// The picker highlight must track the *configured id*, and only the configured id.
    /// It was previously never set at all, so the ComboBox sat on row 0 — showing
    /// "A50 Mono" while the daemon captured a different device entirely.
    #[test]
    fn the_microphone_highlight_follows_the_configured_id_or_nothing() {
        let ids = || {
            [
                "alsa_input.usb-Logitech_A50-00.mono-fallback".to_string(),
                "alsa_input.usb-Focusrite_Scarlett-00.HiFi__Mic1__source".to_string(),
            ]
            .into_iter()
        };
        assert_eq!(
            mic_index(ids(), "alsa_input.usb-Focusrite_Scarlett-00.HiFi__Mic1__source"),
            1
        );
        assert_eq!(mic_index(ids(), "(unset)"), -1, "no device means no highlight");
        assert_eq!(
            mic_index(ids(), "alsa_input.unplugged"),
            -1,
            "an absent device must not fall back to row 0"
        );
        assert_eq!(mic_index([].into_iter(), "anything"), -1);
    }

    /// The count is authoritative and the names are best-effort, so the banner has to read
    /// correctly when there are fewer names than readers — including none at all.
    #[test]
    fn no_holders_means_no_banner_and_holders_are_named_in_it() {
        assert_eq!(holder_banner_text(0, &[]), "");
        assert_eq!(
            holder_banner_text(0, &["chrome (41234)".to_string()]),
            "",
            "nothing is streaming, whatever a stale name list says"
        );

        assert_eq!(
            holder_banner_text(1, &["chrome (41234)".to_string()]),
            "chrome (41234) is using the virtual camera"
        );

        let two = ["chrome (41234)".to_string(), "zoom (555)".to_string()];
        assert_eq!(
            holder_banner_text(2, &two),
            "2 apps are using the virtual camera: chrome (41234), zoom (555)"
        );

        // A Flatpak reader is invisible to the /proc scan, so this is a real case and not a
        // defensive one: the count still has to be reported.
        assert_eq!(
            holder_banner_text(2, &[]),
            "2 apps are using the virtual camera"
        );
        assert_eq!(
            holder_banner_text(1, &[]),
            "1 app is using the virtual camera"
        );
    }
}
