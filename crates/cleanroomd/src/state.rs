//! Shared daemon state.
//!
//! The one design rule worth stating: health is *stored*, not derived at query time.
//! Any subsystem that degrades must call [`Shared::set_video_health`] or
//! [`Shared::set_audio_health`] with a reason, and that reason is what the CLI prints
//! and the GUI banners. There is no code path that quietly does something different
//! from what the config asked for.

// The health-reporting setters have no callers until the audio and video pipelines land.
// They are written now, with the rest of the state, because retrofitting "report every
// degradation" onto working code is exactly how the prior art ended up with a silent CPU
// fallback ladder — the reporting has to exist before there is anything tempted to skip it.
#![allow(dead_code)]

use cleanroom_core::{Config, ConfigPaths};
use cleanroom_ipc::{Health, PipelineStats, Status};
use std::sync::{Arc, RwLock};
use tokio::sync::Notify;

/// A health value together with the human-readable reason for it.
#[derive(Debug, Clone)]
pub struct HealthState {
    pub health: Health,
    pub detail: String,
}

impl HealthState {
    pub fn nominal(detail: impl Into<String>) -> Self {
        Self {
            health: Health::Nominal,
            detail: detail.into(),
        }
    }
    pub fn idle(detail: impl Into<String>) -> Self {
        Self {
            health: Health::Idle,
            detail: detail.into(),
        }
    }
    pub fn degraded(detail: impl Into<String>) -> Self {
        Self {
            health: Health::Degraded,
            detail: detail.into(),
        }
    }
    pub fn failed(detail: impl Into<String>) -> Self {
        Self {
            health: Health::Failed,
            detail: detail.into(),
        }
    }
}

impl Default for HealthState {
    fn default() -> Self {
        Self {
            health: Health::Idle,
            detail: "not started".into(),
        }
    }
}

pub struct Shared {
    config: RwLock<Config>,
    paths: ConfigPaths,
    video: RwLock<HealthState>,
    audio: RwLock<HealthState>,
    stats: RwLock<PipelineStats>,
    gpu_adapter: RwLock<String>,
    vcam_path: RwLock<String>,
    pw_node: RwLock<String>,

    /// What the PipeWire registry watcher sees. Owned here rather than by the audio
    /// pipeline because the D-Bus side must be able to answer ListMicrophones whether or
    /// not the audio thread happens to be running.
    pub audio_registry: Arc<cleanroom_audio::RegistryView>,

    /// Signalled to ask the run loop to stop. Shutdown is explicit rather than a
    /// process::exit because the GPU context must be torn down in a controlled order:
    /// dropping an ONNX Runtime session that owns a Dawn context segfaults, and under
    /// `Restart=on-failure` a segfault on exit becomes an endless restart loop.
    shutdown: Notify,
}

impl Shared {
    pub fn new(config: Config, paths: ConfigPaths) -> Arc<Self> {
        Arc::new(Self {
            config: RwLock::new(config),
            paths,
            video: RwLock::new(HealthState::default()),
            audio: RwLock::new(HealthState::default()),
            stats: RwLock::new(PipelineStats::default()),
            gpu_adapter: RwLock::new("not initialised".into()),
            vcam_path: RwLock::new(String::new()),
            pw_node: RwLock::new(String::new()),
            audio_registry: cleanroom_audio::RegistryView::new(),
            shutdown: Notify::new(),
        })
    }

    pub fn config(&self) -> Config {
        self.config.read().expect("config lock poisoned").clone()
    }

    pub fn paths(&self) -> &ConfigPaths {
        &self.paths
    }

    /// Replace the config and persist it atomically.
    ///
    /// Persist failure is reported, not swallowed — but the in-memory change stands, so
    /// a read-only config directory degrades to "works until restart" rather than
    /// "settings silently do nothing".
    pub fn replace_config(&self, new: Config) -> Result<(), cleanroom_core::ConfigError> {
        *self.config.write().expect("config lock poisoned") = new.clone();
        cleanroom_core::persist::save(&self.paths, &new)
    }

    pub fn set_video_health(&self, s: HealthState) {
        Self::log_transition("video", &s);
        *self.video.write().expect("video lock poisoned") = s;
    }

    pub fn set_audio_health(&self, s: HealthState) {
        Self::log_transition("audio", &s);
        *self.audio.write().expect("audio lock poisoned") = s;
    }

    fn log_transition(what: &str, s: &HealthState) {
        match s.health {
            Health::Nominal | Health::Idle => {
                tracing::info!(subsystem = what, health = %s.health, detail = %s.detail, "state")
            }
            // Degradation is a warning at minimum. It must never be reported only at
            // debug level, or it is effectively silent.
            Health::Degraded => {
                tracing::warn!(subsystem = what, detail = %s.detail, "running degraded")
            }
            Health::Failed => tracing::error!(subsystem = what, detail = %s.detail, "failed"),
        }
    }

    /// Update only the microphone fields, leaving the video telemetry alone. The two
    /// pipelines publish independently and must not clobber each other.
    pub fn update_mic_levels(&self, in_db: f32, out_db: f32) {
        if let Ok(mut s) = self.stats.write() {
            s.mic_level_db = in_db;
            s.mic_level_out_db = out_db;
        }
    }

    pub fn set_stats(&self, s: PipelineStats) {
        let mut guard = self.stats.write().expect("stats lock poisoned");
        // Preserve the audio fields: the video thread owns everything else, and a plain
        // overwrite here would make the mic meters flicker to zero once a second.
        let (mic_in, mic_out) = (guard.mic_level_db, guard.mic_level_out_db);
        *guard = PipelineStats {
            mic_level_db: mic_in,
            mic_level_out_db: mic_out,
            ..s
        };
    }

    pub fn set_gpu_adapter(&self, s: impl Into<String>) {
        *self.gpu_adapter.write().expect("gpu lock poisoned") = s.into();
    }

    pub fn set_pw_node(&self, s: impl Into<String>) {
        *self.pw_node.write().expect("pw lock poisoned") = s.into();
    }

    pub fn set_vcam_path(&self, s: impl Into<String>) {
        *self.vcam_path.write().expect("vcam lock poisoned") = s.into();
    }

    pub fn status(&self) -> Status {
        let v = self.video.read().expect("video lock poisoned").clone();
        let a = self.audio.read().expect("audio lock poisoned").clone();
        Status {
            video_health: v.health,
            video_detail: v.detail,
            audio_health: a.health,
            audio_detail: a.detail,
            gpu_adapter: self.gpu_adapter.read().expect("gpu lock poisoned").clone(),
            vcam_path: self.vcam_path.read().expect("vcam lock poisoned").clone(),
            pw_node: self.pw_node.read().expect("pw lock poisoned").clone(),
            stats: self.stats.read().expect("stats lock poisoned").clone(),
        }
    }

    /// Ask the daemon to stop.
    pub fn request_shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    /// Resolves once shutdown has been requested.
    pub async fn wait_for_shutdown(&self) {
        self.shutdown.notified().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared() -> Arc<Shared> {
        let dir = std::env::temp_dir().join(format!("cleanroom-test-{}", std::process::id()));
        Shared::new(Config::default(), ConfigPaths::at(dir.join("config.toml")))
    }

    #[test]
    fn health_defaults_to_idle_not_nominal() {
        // Reporting Nominal before anything has started would be a lie, and it is the
        // kind of lie that makes a broken pipeline look fine.
        let s = shared();
        assert_eq!(s.status().video_health, Health::Idle);
        assert_eq!(s.status().audio_health, Health::Idle);
    }

    #[test]
    fn health_and_reason_are_reported_together() {
        let s = shared();
        s.set_video_health(HealthState::degraded(
            "MJPEG decode fell back to a slower path",
        ));
        let st = s.status();
        assert_eq!(st.video_health, Health::Degraded);
        assert!(st.video_detail.contains("MJPEG"));
    }
}
