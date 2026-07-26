//! Command-line control for the Cleanroom daemon.
//!
//! Everything the GUI can do is reachable from here. That is a design commitment rather
//! than a convenience: it keeps the daemon's D-Bus surface honest, and it means a
//! headless server or a compositor keybind can drive Cleanroom without a GUI at all.
//!
//! Global hotkeys are deliberately *not* implemented in-process. The Wayland
//! GlobalShortcuts portal is unimplemented on sway/river/niri, and on Hyprland it does
//! not even show which app registered a binding. Binding a compositor key to
//! `cleanroom-ctl set video.background off` is more portable and more predictable.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cleanroom_ipc::{CleanroomProxy, Health};

#[derive(Parser)]
#[command(
    name = "cleanroom-ctl",
    about = "Control the Cleanroom daemon",
    long_about = "Control the Cleanroom daemon.\n\n\
                  The daemon is D-Bus activated, so any command here starts it if it is not \
                  already running."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show what the daemon is doing.
    Status,
    /// Read one setting, e.g. `video.blur_strength`.
    Get { key: String },
    /// Write one setting. Applied immediately and saved.
    Set { key: String, value: String },
    /// List every setting with its current value.
    Keys,
    /// List cameras and microphones the daemon can use.
    Devices,
    /// Check the environment for the things that usually go wrong.
    Doctor,
    /// Re-read the config file, discarding unsaved runtime changes.
    Reload,
    /// Ask the daemon to shut down cleanly.
    Shutdown,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let connection = zbus::Connection::session()
        .await
        .context("connecting to the D-Bus session bus")?;
    let proxy = CleanroomProxy::new(&connection).await.context(
        "reaching the Cleanroom daemon. If it is not installed as a D-Bus activatable \
         service, start it with `cleanroomd`",
    )?;

    match cli.command {
        Command::Status => status(&proxy).await?,
        Command::Get { key } => println!("{}", proxy.get(&key).await?),
        Command::Set { key, value } => {
            proxy.set(&key, &value).await?;
            println!("{key} = {}", proxy.get(&key).await?);
        }
        Command::Keys => {
            for (k, v) in proxy.keys().await? {
                println!("{k:<34} {v}");
            }
        }
        Command::Devices => devices(&proxy).await?,
        Command::Doctor => {
            for line in proxy.doctor().await? {
                println!("{line}");
            }
        }
        Command::Reload => {
            proxy.reload().await?;
            println!("config reloaded");
        }
        Command::Shutdown => {
            proxy.shutdown().await?;
            println!("shutdown requested");
        }
    }
    Ok(())
}

async fn status(proxy: &CleanroomProxy<'_>) -> Result<()> {
    let s = proxy.status().await?;

    // Health first and unmissable. The whole point of tracking degradation explicitly is
    // that someone sees it, so it leads rather than hiding under the numbers.
    println!("video    {}  {}", marker(&s.video_health), s.video_detail);
    println!("audio    {}  {}", marker(&s.audio_health), s.audio_detail);
    println!();
    println!("gpu      {}", s.gpu_adapter);
    if !s.vcam_path.is_empty() {
        println!("vcam     {}", s.vcam_path);
    }
    // Reported separately because the two transports fail independently: v4l2loopback
    // reaches Chrome and Zoom, the PipeWire node reaches Flatpak and portal apps, and
    // knowing which one is up is the difference between "my camera is broken" and "my
    // camera is broken *in this app*".
    if !s.pw_node.is_empty() {
        println!("pw node  {}", s.pw_node);
    }

    let st = &s.stats;
    println!();
    println!(
        "video    {:.1} fps   decode {:.2} ms   gpu {:.2} ms   matting {:.2} ms   {} dropped",
        st.fps, st.decode_ms, st.gpu_ms, st.matting_ms, st.dropped
    );
    println!(
        "         {} consumer(s) reading the virtual camera",
        st.vcam_consumers
    );
    // Only shown once it has happened. A permanent "0 rejected" trains people to skip the
    // line, which is the opposite of what a counter that only matters when non-zero needs.
    if st.matte_rejected > 0 {
        println!(
            "         {} matte(s) rejected as degenerate since startup",
            st.matte_rejected
        );
    }
    println!(
        "audio    mic {:.1} dBFS in -> {:.1} dBFS out",
        st.mic_level_db, st.mic_level_out_db
    );
    Ok(())
}

/// A health marker that survives being piped through a log or a terminal without colour.
fn marker(h: &Health) -> &'static str {
    match h {
        Health::Nominal => "[ ok ]",
        Health::Idle => "[idle]",
        Health::Degraded => "[WARN]",
        Health::Failed => "[FAIL]",
    }
}

async fn devices(proxy: &CleanroomProxy<'_>) -> Result<()> {
    let cams = proxy.list_cameras().await?;
    println!("cameras:");
    if cams.is_empty() {
        println!("  (none)");
    }
    for d in cams {
        println!(
            "  {:<24} {}{}",
            d.id,
            d.description,
            if d.available { "" } else { "   [in use]" }
        );
    }

    let mics = proxy.list_microphones().await?;
    println!("\nmicrophones:");
    if mics.is_empty() {
        println!("  (none)");
    }
    for d in mics {
        println!(
            "  {:<24} {}{}",
            d.id,
            d.description,
            if d.available { "" } else { "   [in use]" }
        );
    }
    Ok(())
}
