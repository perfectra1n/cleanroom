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
}

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

    ui.show()?;
    slint::run_event_loop()?;
    Ok(())
}

/// Poll the daemon and apply UI requests. Runs as a task on the shared runtime.
async fn dbus_loop(ui: slint::Weak<AppWindow>, mut set_rx: mpsc::UnboundedReceiver<SetRequest>) {
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(500));
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
                let snap = poll().await;
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

async fn poll() -> Option<Snapshot> {
    let p = proxy().await.ok()?;
    Some(Snapshot {
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

fn apply_snapshot(ui: &AppWindow, s: Snapshot) {
    ui.set_connected(true);

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

    // Reflect the daemon's values back, so a change made from the CLI or a second GUI
    // shows up here rather than leaving the two silently disagreeing.
    ui.set_background_mode(s.background);
    ui.set_blur_strength(s.blur);
    ui.set_mirror(s.mirror);
    ui.set_denoise(s.denoise);
    ui.set_attenuation(s.attenuation);
}

fn wire_controls(ui: &AppWindow, tx: mpsc::UnboundedSender<SetRequest>) {
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
