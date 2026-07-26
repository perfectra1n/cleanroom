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
//! So the tray is decoration, never the only entry point: closing the window quits the GUI
//! rather than hiding into a tray that may not exist, and everything in the tray menu is
//! also reachable from `cleanroom-ctl`.

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
    background_image: String,
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

    // The preview consumes the virtual camera the same way a meeting app does, so it is
    // WYSIWYG by construction.
    //
    // Started from the status snapshot rather than from a repeating UI timer: the vcam path
    // is not known until the daemon has been polled at least once, and the snapshot is
    // already the thing that learns it. One fewer mechanism, and it cannot start before
    // there is a path to start on.
    let preview_slot: std::rc::Rc<std::cell::RefCell<Option<preview::Preview>>> =
        std::rc::Rc::new(std::cell::RefCell::new(None));
    PREVIEW_SLOT.with(|s| *s.borrow_mut() = Some(preview_slot));

    if let Some(t) = &tray {
        wire_tray(t, &ui, set_tx);
    }

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
                let snap = poll(with_devices).await;
                let ui = ui.clone();
                // Touching the UI from another thread is not allowed; this hands the
                // update to the Slint event loop.
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = ui.upgrade() {
                        match snap {
                            Some(s) => apply_snapshot(&ui, s),
                            None => ui.set_connected(false),
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

async fn poll(with_devices: bool) -> Option<Snapshot> {
    let p = proxy().await.ok()?;

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
    /// Owns the preview thread. Thread-local because it can only be touched from the UI
    /// thread, which is also the only place `apply_snapshot` runs.
    static PREVIEW_SLOT: std::cell::RefCell<
        Option<std::rc::Rc<std::cell::RefCell<Option<preview::Preview>>>>,
    > = const { std::cell::RefCell::new(None) };
}

/// Start the preview once the daemon has told us which device to read.
fn ensure_preview(ui: &AppWindow, vcam_path: &str) {
    if vcam_path.is_empty() || vcam_path == "—" {
        return;
    }
    PREVIEW_SLOT.with(|slot| {
        let slot = slot.borrow();
        let Some(slot) = slot.as_ref() else { return };
        if slot.borrow().is_some() {
            return;
        }
        tracing::info!(path = vcam_path, "starting preview");
        let weak = ui.as_weak();
        *slot.borrow_mut() = Some(preview::Preview::start(
            vcam_path.to_string(),
            move |rgb, pw, ph| {
                if let Some(ui) = weak.upgrade() {
                    let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
                        &rgb, pw, ph,
                    );
                    ui.set_preview_frame(slint::Image::from_rgb8(buf));
                    ui.set_preview_running(true);
                }
            },
        ));
    });
}

fn apply_snapshot(ui: &AppWindow, s: Snapshot) {
    ui.set_connected(true);
    // Both taken before the status fields are moved into the UI below.
    let active_camera = current_camera_id(&s.status);
    let vcam_path = s.status.vcam_path.clone();

    ui.set_video_health(health_code(&s.status.video_health));
    ui.set_video_detail(s.status.video_detail.into());
    ui.set_audio_health(health_code(&s.status.audio_health));
    ui.set_audio_detail(s.status.audio_detail.into());
    ui.set_gpu_adapter(s.status.gpu_adapter.into());
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

    ui.set_background_image(s.background_image.clone().into());
    ensure_preview(ui, &vcam_path);

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
        ui.set_microphone_names(slint::ModelRc::new(slint::VecModel::from(mic_names)));

        ui.set_autostart_on(d.autostart_on);
        ui.set_autostart_mechanism(d.autostart_mechanism.clone().into());
        ui.set_autostart_instruction(d.autostart_instruction.clone().into());
    }

    // Reflect the daemon's values back, so a change made from the CLI or a second GUI
    // shows up here rather than leaving the two silently disagreeing.
    ui.set_background_mode(s.background);
    ui.set_blur_strength(s.blur);
    ui.set_mirror(s.mirror);
    ui.set_denoise(s.denoise);
    ui.set_attenuation(s.attenuation);
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
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_set_microphone(move |label| {
            // Microphones are listed by description, so map back through the model to the
            // PipeWire node.name, which is what config stores.
            let Some(ui) = ui_weak.upgrade() else { return };
            use slint::Model;
            let names = ui.get_microphone_names();
            if let Some(i) = names.iter().position(|n| n.as_str() == label.as_str()) {
                ui.set_microphone_index(i as i32);
            }
            let _ = tx.send(SetRequest {
                key: "audio.device",
                value: label.to_string(),
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
