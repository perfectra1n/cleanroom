//! The Cleanroom daemon.
//!
//! Owns the camera, the microphone and the GPU, and outlives the GUI. Everything is
//! driven over the D-Bus session bus so the GUI, the CLI and `busctl` are equal.

mod audio_pipeline;
mod doctor;
mod service;
mod settings;
mod state;
mod video_pipeline;

use anyhow::{Context, Result};
use cleanroom_core::{ConfigPaths, LoadOutcome};
use state::Shared;
use tracing_subscriber::prelude::*;
use zbus::fdo::{DBusProxy, RequestNameFlags, RequestNameReply};
use zbus::names::WellKnownName;

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();

    let paths = ConfigPaths::discover().context("locating the config directory")?;
    let (config, outcome) = cleanroom_core::persist::load(&paths).with_context(|| {
        format!(
            "loading {}. Refusing to start rather than overwrite it with defaults",
            paths.primary.display()
        )
    })?;

    match outcome {
        LoadOutcome::Loaded => tracing::info!(path = %paths.primary.display(), "config loaded"),
        LoadOutcome::CreatedDefault => {
            tracing::info!(path = %paths.primary.display(), "no config yet; starting from defaults")
        }
        // Worth a warning rather than an info: the user has silently lost whatever
        // changed between the backup and the corruption.
        LoadOutcome::RecoveredFromBackup => tracing::warn!(
            path = %paths.primary.display(),
            "config was corrupt; recovered from backup. Changes since the last good save are lost"
        ),
    }

    let shared = Shared::new(config, paths);

    let connection = zbus::connection::Builder::session()
        .context("connecting to the D-Bus session bus")?
        .serve_at(
            cleanroom_ipc::OBJECT_PATH,
            service::Service {
                shared: shared.clone(),
            },
        )?
        .build()
        .await
        .context("publishing the Cleanroom interface")?;

    // Single-instance guard.
    //
    // DO_NOT_QUEUE without ALLOW_REPLACEMENT means: if someone else already owns the
    // name, fail immediately rather than waiting behind them or stealing it. Exiting 0
    // — not 1 — is deliberate: a second launch is a normal thing for a user to do (a
    // stale autostart entry, clicking the launcher twice), not a failure, and pairing a
    // non-zero exit with Restart=on-failure would produce a restart loop.
    let name = WellKnownName::try_from(cleanroom_ipc::BUS_NAME)?;
    let dbus = DBusProxy::new(&connection).await?;
    let reply = dbus
        .request_name(name.clone(), RequestNameFlags::DoNotQueue.into())
        .await
        .context("requesting the well-known bus name")?;

    match reply {
        RequestNameReply::PrimaryOwner => {}
        other => {
            tracing::info!(
                ?other,
                "another cleanroomd already owns {}; exiting quietly",
                cleanroom_ipc::BUS_NAME
            );
            return Ok(());
        }
    }

    tracing::info!(
        name = cleanroom_ipc::BUS_NAME,
        path = cleanroom_ipc::OBJECT_PATH,
        "cleanroomd is up"
    );

    // Started after the name is held, so a losing second instance never touches the
    // camera. Failures inside surface as health over D-Bus rather than as a dead daemon.
    let mut video = video_pipeline::VideoPipeline::start(shared.clone());
    let mut audio = audio_pipeline::AudioPipeline::start(shared.clone());

    // Fold the audio levels into the published stats. The video thread owns the rest of
    // PipelineStats, so this only touches the two mic fields.
    {
        let shared = shared.clone();
        let levels = audio.level_handle();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_millis(500));
            loop {
                tick.tick().await;
                let (i, o) = levels.dbfs();
                shared.update_mic_levels(i, o);
            }
        });
    }

    run_until_shutdown(&shared).await;

    // Controlled teardown.
    //
    // This is not politeness. Dropping an ONNX Runtime session that owns a WebGPU/Dawn
    // context segfaults reproducibly on both Nvidia and AMD adapters, *after* all work
    // has completed (see docs/spike-results.md). Under Restart=on-failure a segfault at
    // exit becomes an endless restart loop, so the GPU-owning subsystems have to be shut
    // down in order, before we return.
    tracing::info!("shutting down");
    // Joined, not detached: the thread owns the camera and the loopback fd, and exiting
    // while it still holds them leaves the device half-streaming for the next process.
    video.shutdown();
    audio.shutdown();
    drop(connection);
    tracing::info!("clean exit");
    Ok(())
}

async fn run_until_shutdown(shared: &Shared) {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("installing SIGTERM handler");

    tokio::select! {
        _ = tokio::signal::ctrl_c() => tracing::info!("SIGINT"),
        _ = sigterm.recv() => tracing::info!("SIGTERM"),
        _ = shared.wait_for_shutdown() => tracing::info!("shutdown requested"),
    }
}

fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new("cleanroom=info,cleanroomd=info,warn")
    });

    let registry = tracing_subscriber::registry().with(filter);

    // Under systemd, log to the journal with structured fields intact. Otherwise log to
    // stderr for a human. `connect()` failing just means we are not under systemd.
    match tracing_journald::layer() {
        Ok(journald) => registry.with(journald).init(),
        Err(_) => registry
            .with(tracing_subscriber::fmt::layer().with_target(false))
            .init(),
    }
}
