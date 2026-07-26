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
        // Populated by cleanroom-video once the capture backend lands. Returning an
        // empty list is honest; inventing entries would not be.
        Vec::new()
    }

    async fn list_microphones(&self) -> Vec<DeviceInfo> {
        Vec::new()
    }

    async fn get(&self, key: &str) -> zbus::fdo::Result<String> {
        settings::get(&self.shared.config(), key)
            .map_err(|e| zbus::fdo::Error::InvalidArgs(e.to_string()))
    }

    async fn set(&self, key: &str, value: &str) -> zbus::fdo::Result<()> {
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
        doctor::run(&self.shared.config())
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
