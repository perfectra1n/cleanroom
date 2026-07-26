//! Releasing devices across a system suspend.
//!
//! A daemon that holds `/dev/nvidia*` open 24/7 is a daemon that can block system suspend,
//! and this one holds a lot of them — 32 fds on `/dev/nvidia0` alone on the reference
//! machine, plus `/dev/nvidiactl` and the DRM render node.
//!
//! ## A delay inhibitor is not optional
//!
//! Subscribing to `PrepareForSleep` alone is useless: the signal fires and the system
//! suspends regardless, giving no time to act on it. logind only waits for processes
//! holding a **delay** inhibitor lock, and only until the lock is dropped or
//! `InhibitDelayMaxSec` (5 s by default) expires. So the shape is:
//!
//! 1. take an `Inhibit("sleep", …, "delay")` lock at startup and hold the fd;
//! 2. on `PrepareForSleep(true)`, tear down, *then* drop the fd — dropping it is the signal
//!    that we are ready;
//! 3. on `PrepareForSleep(false)`, rebuild and take a fresh lock for next time.
//!
//! Getting step 2 backwards — dropping the lock before tearing down — looks like it works
//! and silently does nothing, because the machine suspends while the teardown is still in
//! flight.
//!
//! ## What can and cannot actually be released
//!
//! The wgpu device and instance can be dropped, which releases their handles. The ONNX
//! Runtime session cannot: dropping one that owns a Dawn context makes the *process exit*
//! segfault, so `Matter` deliberately leaks it (see `impl Drop for Matter`). Since that
//! session holds its own Dawn/Vulkan context, some `/dev/nvidia*` handles survive a
//! suspend cycle no matter what we do here.
//!
//! That is a real limitation and is reported rather than hidden. It is also the clearest
//! cost of the teardown defect: the leak is not just memory, it is the reason a full GPU
//! release is impossible today.

use crate::state::Shared;
use std::sync::Arc;
use std::time::Duration;
use zbus::zvariant::OwnedFd;

/// How long to wait for the video thread to acknowledge a suspend request.
///
/// Under logind's default `InhibitDelayMaxSec` of 5 s, so we release the lock deliberately
/// rather than having it taken away mid-teardown.
const ACK_TIMEOUT: Duration = Duration::from_secs(4);

#[zbus::proxy(
    interface = "org.freedesktop.login1.Manager",
    default_service = "org.freedesktop.login1",
    default_path = "/org/freedesktop/login1"
)]
trait Login1Manager {
    /// Returns a file descriptor. The lock is held for exactly as long as it is open, so
    /// the returned fd must be kept alive and dropped deliberately.
    fn inhibit(&self, what: &str, who: &str, why: &str, mode: &str) -> zbus::Result<OwnedFd>;

    #[zbus(signal)]
    fn prepare_for_sleep(&self, start: bool) -> zbus::Result<()>;
}

/// Watch logind and drive the pipeline's suspend handshake.
///
/// Runs until the daemon shuts down. A failure to reach logind is logged once and then
/// dropped: a machine with no logind (a container, a minimal session) should still run
/// Cleanroom, it just will not release devices for a suspend that is not coming.
pub async fn watch(shared: Arc<Shared>) {
    // The **system** bus. login1 is not on the session bus, and connecting to the wrong
    // one fails in a way that reads like "logind is missing".
    let connection = match zbus::Connection::system().await {
        Ok(c) => c,
        Err(e) => {
            tracing::info!(error = %e, "no system bus; suspend handling disabled");
            return;
        }
    };

    let manager = match Login1ManagerProxy::new(&connection).await {
        Ok(m) => m,
        Err(e) => {
            tracing::info!(error = %e, "logind unavailable; suspend handling disabled");
            return;
        }
    };

    let mut signals = match manager.receive_prepare_for_sleep().await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "could not subscribe to PrepareForSleep");
            return;
        }
    };

    let mut lock = take_lock(&manager).await;
    if lock.is_none() {
        tracing::warn!(
            "could not take a delay inhibitor; the GPU and camera will not be released \
             before a suspend"
        );
    }

    use futures_util::StreamExt;
    while let Some(signal) = signals.next().await {
        let Ok(args) = signal.args() else { continue };

        if args.start {
            tracing::info!("system is suspending; releasing devices");
            shared.request_suspend();

            // Wait for the video thread to say it has let go. Polling rather than a
            // condvar because the other side is a plain OS thread in a blocking loop, and
            // a missed notification here would mean suspending with the camera still open.
            let deadline = std::time::Instant::now() + ACK_TIMEOUT;
            while !shared.suspend_acknowledged() && std::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            if !shared.suspend_acknowledged() {
                tracing::warn!(
                    "video thread did not release its devices within {:?}; suspending anyway",
                    ACK_TIMEOUT
                );
            }

            // Only now. Dropping the fd is what tells logind we are ready, so doing it any
            // earlier means the machine suspends while the teardown is still running.
            drop(lock.take());
        } else {
            tracing::info!("system resumed; reacquiring devices");
            shared.clear_suspend();
            lock = take_lock(&manager).await;
        }
    }
}

async fn take_lock(manager: &Login1ManagerProxy<'_>) -> Option<OwnedFd> {
    match manager
        .inhibit(
            "sleep",
            "Cleanroom",
            "Releasing the camera and GPU before suspend",
            "delay",
        )
        .await
    {
        Ok(fd) => Some(fd),
        Err(e) => {
            tracing::warn!(error = %e, "could not take a delay inhibitor lock");
            None
        }
    }
}

/// Read the loaded Nvidia driver's userspace version, if this is an Nvidia machine.
///
/// Recorded at startup and compared later. A package upgrade replaces the userspace `.so`s
/// while the *old* kernel module stays loaded, so new contexts fail with a version mismatch
/// while the one we already hold keeps working perfectly. The result is a daemon that works
/// until it is restarted, at which point it does not — and nothing connects the two events
/// unless somebody says so at the time.
pub fn driver_version() -> Option<String> {
    std::fs::read_to_string("/sys/module/nvidia/version")
        .ok()
        .map(|s| s.trim().to_string())
}
