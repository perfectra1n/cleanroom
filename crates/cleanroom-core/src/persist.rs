//! Config persistence.
//!
//! This module exists because of a specific, expensive bug in the prior art, described
//! in its own commit as "Stop silent config wipes". Two failures compounded:
//!
//!   1. Saving used a plain truncating write, so a crash mid-save left invalid TOML.
//!   2. Loading did `except Exception: return AppConfig()` — silently returning fresh
//!      defaults. The next settings change then *persisted* those defaults over the
//!      backup, making the loss permanent and invisible.
//!
//! It was found while investigating an unrelated complaint ("the denoiser has no
//! effect"), which is the tell: silent data loss does not report itself.
//!
//! So the contract here is deliberately narrow:
//!
//!   * **Saving is atomic.** Nothing observes a half-written file.
//!   * **A missing file is not an error** — it is first run, and yields defaults.
//!   * **A corrupt file is never silently replaced by defaults.** We fall back to the
//!     backup, loudly, and if that fails too we return an error and let the caller
//!     decide. Refusing to start beats erasing someone's settings.

use crate::config::Config;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not determine a config directory (is $HOME or $XDG_CONFIG_HOME set?)")]
    NoConfigDir,

    #[error("i/o error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not serialise config: {0}")]
    Serialise(#[from] toml::ser::Error),

    /// Both the config and its backup failed to parse. Deliberately fatal: the
    /// alternative is overwriting whatever the user had with defaults.
    #[error(
        "config at {path} is corrupt ({source}), and the backup could not be used either. \
         Refusing to overwrite it with defaults — move it aside to start fresh."
    )]
    CorruptBeyondRecovery {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

/// How the config that is now in memory came to be. The daemon reports this over D-Bus
/// so a recovered-from-backup situation is visible rather than merely logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOutcome {
    /// Loaded normally.
    Loaded,
    /// No file existed. This is first run, not a fault.
    CreatedDefault,
    /// The primary file was unreadable or unparseable and the backup was used instead.
    /// The user should be told: they have lost whatever changed since the backup.
    RecoveredFromBackup,
}

/// Resolved paths for the config triple.
#[derive(Debug, Clone)]
pub struct ConfigPaths {
    pub primary: PathBuf,
    pub backup: PathBuf,
    pub temp: PathBuf,
}

impl ConfigPaths {
    /// `$XDG_CONFIG_HOME/cleanroom/config.toml`, falling back to `$HOME/.config`.
    pub fn discover() -> Result<Self, ConfigError> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .ok_or(ConfigError::NoConfigDir)?;
        Ok(Self::at(base.join("cleanroom").join("config.toml")))
    }

    /// Derive the triple from an explicit primary path. Useful for tests and for a
    /// `--config` flag.
    pub fn at(primary: impl Into<PathBuf>) -> Self {
        let primary = primary.into();
        let backup = with_extra_extension(&primary, "bak");
        let temp = with_extra_extension(&primary, "tmp");
        Self {
            primary,
            backup,
            temp,
        }
    }
}

fn with_extra_extension(p: &Path, ext: &str) -> PathBuf {
    let mut s = p.as_os_str().to_os_string();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> ConfigError + '_ {
    move |source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Load the config, recovering from the backup if the primary is unusable.
pub fn load(paths: &ConfigPaths) -> Result<(Config, LoadOutcome), ConfigError> {
    let primary_err = match fs::read_to_string(&paths.primary) {
        Ok(text) => match toml::from_str::<Config>(&text) {
            Ok(cfg) => return Ok((cfg, LoadOutcome::Loaded)),
            Err(e) => {
                tracing::error!(
                    path = %paths.primary.display(),
                    error = %e,
                    "config is corrupt; attempting to recover from backup"
                );
                Some(e)
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // First run. Not a fault, and not something to shout about.
            tracing::info!(path = %paths.primary.display(), "no config yet; using defaults");
            return Ok((Config::default(), LoadOutcome::CreatedDefault));
        }
        Err(e) => {
            return Err(ConfigError::Io {
                path: paths.primary.clone(),
                source: e,
            });
        }
    };

    // Primary was present but unparseable. Try the backup before doing anything
    // destructive.
    if let Ok(text) = fs::read_to_string(&paths.backup) {
        if let Ok(cfg) = toml::from_str::<Config>(&text) {
            tracing::warn!(
                backup = %paths.backup.display(),
                "recovered config from backup; changes since the last successful save are lost"
            );
            return Ok((cfg, LoadOutcome::RecoveredFromBackup));
        }
        tracing::error!(backup = %paths.backup.display(), "backup is also unparseable");
    }

    // Both gone. Returning defaults here is what caused permanent loss in the prior
    // art, because the next save would overwrite the user's file with them.
    Err(ConfigError::CorruptBeyondRecovery {
        path: paths.primary.clone(),
        source: primary_err.expect("set on the parse-failure path"),
    })
}

/// Save atomically.
///
/// Order matters and is the whole point:
///   1. write the new content to `config.toml.tmp` and fsync it
///   2. rename any existing `config.toml` to `config.toml.bak`
///   3. rename `config.toml.tmp` over `config.toml` — atomic on POSIX
///   4. fsync the directory so the renames survive a power loss
///
/// At no instant does `config.toml` contain a partial write, and the previous good
/// version is always one rename away.
pub fn save(paths: &ConfigPaths, config: &Config) -> Result<(), ConfigError> {
    let dir = paths.primary.parent().ok_or(ConfigError::NoConfigDir)?;
    fs::create_dir_all(dir).map_err(io_err(dir))?;

    let text = toml::to_string_pretty(config)?;

    {
        let mut f = fs::File::create(&paths.temp).map_err(io_err(&paths.temp))?;
        f.write_all(text.as_bytes()).map_err(io_err(&paths.temp))?;
        // Without this the rename can land while the contents are still in page cache,
        // which turns a power loss into an empty file rather than the old one.
        f.sync_all().map_err(io_err(&paths.temp))?;
    }

    if paths.primary.exists() {
        fs::rename(&paths.primary, &paths.backup).map_err(io_err(&paths.primary))?;
    }
    fs::rename(&paths.temp, &paths.primary).map_err(io_err(&paths.temp))?;

    // Renames are metadata operations; fsync the directory to persist them.
    // Best-effort: some filesystems refuse an O_RDONLY directory fsync, and failing the
    // whole save over that would be worse than the durability we lose.
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BackgroundMode;

    fn paths_in(dir: &tempfile::TempDir) -> ConfigPaths {
        ConfigPaths::at(dir.path().join("config.toml"))
    }

    #[test]
    fn missing_config_is_first_run_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let (cfg, outcome) = load(&paths_in(&dir)).expect("missing file must not error");
        assert_eq!(outcome, LoadOutcome::CreatedDefault);
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let p = paths_in(&dir);

        let mut cfg = Config::default();
        cfg.video.blur_strength = 0.85;
        cfg.video.background = BackgroundMode::Replace;
        save(&p, &cfg).unwrap();

        let (back, outcome) = load(&p).unwrap();
        assert_eq!(outcome, LoadOutcome::Loaded);
        assert_eq!(back, cfg);
    }

    #[test]
    fn second_save_leaves_a_usable_backup() {
        let dir = tempfile::tempdir().unwrap();
        let p = paths_in(&dir);

        let mut first = Config::default();
        first.video.blur_strength = 0.1;
        save(&p, &first).unwrap();

        let mut second = Config::default();
        second.video.blur_strength = 0.9;
        save(&p, &second).unwrap();

        assert!(
            p.backup.exists(),
            "a backup must exist after the second save"
        );
        let backup: Config = toml::from_str(&fs::read_to_string(&p.backup).unwrap()).unwrap();
        assert_eq!(
            backup.video.blur_strength, 0.1,
            "backup must hold the previous version"
        );
    }

    #[test]
    fn corrupt_primary_recovers_from_backup() {
        let dir = tempfile::tempdir().unwrap();
        let p = paths_in(&dir);

        let mut good = Config::default();
        good.video.blur_strength = 0.42;
        save(&p, &good).unwrap();
        // Second save so a backup exists, then corrupt the primary the way a crash
        // mid-write would.
        save(&p, &good).unwrap();
        fs::write(&p.primary, "this is not toml {{{").unwrap();

        let (cfg, outcome) = load(&p).expect("must recover rather than fail");
        assert_eq!(outcome, LoadOutcome::RecoveredFromBackup);
        assert_eq!(cfg.video.blur_strength, 0.42);
    }

    #[test]
    fn corrupt_primary_and_backup_errors_rather_than_returning_defaults() {
        // The regression test for the actual historical bug. Returning
        // Config::default() here is what made the loss permanent, because the next
        // save wrote those defaults over the user's file.
        let dir = tempfile::tempdir().unwrap();
        let p = paths_in(&dir);

        fs::create_dir_all(dir.path()).unwrap();
        fs::write(&p.primary, "garbage {{{").unwrap();
        fs::write(&p.backup, "also garbage }}}").unwrap();

        let err = load(&p).expect_err("must not silently return defaults");
        assert!(matches!(err, ConfigError::CorruptBeyondRecovery { .. }));
    }

    #[test]
    fn no_temp_file_is_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let p = paths_in(&dir);
        save(&p, &Config::default()).unwrap();
        assert!(
            !p.temp.exists(),
            "the temp file must be renamed away, not left"
        );
    }

    #[test]
    fn save_creates_missing_directories() {
        let dir = tempfile::tempdir().unwrap();
        let p = ConfigPaths::at(dir.path().join("nested").join("deeper").join("config.toml"));
        save(&p, &Config::default()).expect("must create parent directories");
        assert!(p.primary.exists());
    }
}
