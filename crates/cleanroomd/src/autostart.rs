//! Starting the daemon with the session, on desktops that disagree about how.
//!
//! There is no portable answer, so the decision is made **at toggle time** rather than
//! baked into packaging. On bare Hyprland *neither* standard mechanism works:
//! `graphical-session.target` never activates when it is launched from a TTY, and Hyprland
//! runs no XDG autostart either ([#5169], closed *not planned*). A `.desktop` file dropped
//! in `~/.config/autostart` there is simply ignored, silently, which is the worst possible
//! outcome for a checkbox labelled "start automatically".
//!
//! So: ask the session what it actually supports, do that, and when the answer is "neither",
//! say so and hand over the exact line to paste.
//!
//! `Type=dbus` activation sits underneath all three regardless — any `cleanroom-ctl` call
//! or GUI launch starts the daemon on demand — so even the worst case is degraded rather
//! than broken.
//!
//! [#5169]: https://github.com/hyprwm/Hyprland/issues/5169

use std::path::PathBuf;

/// Which mechanism was used, or would be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mechanism {
    /// A systemd user unit wanted by `graphical-session.target`.
    SystemdUser,
    /// An XDG autostart desktop entry.
    XdgAutostart,
    /// Neither works here; the user has to add a line to their compositor config.
    CompositorExecOnce,
}

impl Mechanism {
    pub fn as_str(&self) -> &'static str {
        match self {
            Mechanism::SystemdUser => "systemd-user",
            Mechanism::XdgAutostart => "xdg-autostart",
            Mechanism::CompositorExecOnce => "compositor-exec-once",
        }
    }
}

/// Desktops known to honour XDG autostart.
///
/// An allow-list rather than a deny-list, and that direction is deliberate. Getting it
/// wrong in the permissive direction means writing a file that is silently ignored and
/// telling the user autostart is on — the exact failure this module exists to avoid. Being
/// wrong in the conservative direction only means offering the exec-once line to someone
/// who did not strictly need it, which is visible and harmless.
const XDG_AUTOSTART_DESKTOPS: &[&str] = &[
    "GNOME",
    "KDE",
    "XFCE",
    "X-Cinnamon",
    "MATE",
    "LXQt",
    "Budgie",
    "Pantheon",
    "Deepin",
];

/// The desktop entry we write, matching the D-Bus name and the GUI's `app_id`.
fn autostart_entry_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(
        base.join("autostart")
            .join(format!("{}.desktop", cleanroom_ipc::BUS_NAME)),
    )
}

/// Whether this session honours XDG autostart, by `$XDG_CURRENT_DESKTOP`.
fn desktop_honours_xdg_autostart() -> bool {
    let Ok(current) = std::env::var("XDG_CURRENT_DESKTOP") else {
        return false;
    };
    // The variable is colon-separated and can carry several names, e.g. "ubuntu:GNOME".
    current.split(':').any(|name| {
        XDG_AUTOSTART_DESKTOPS
            .iter()
            .any(|k| k.eq_ignore_ascii_case(name))
    })
}

/// The daemon's own absolute path.
///
/// Absolute is not a nicety. systemd's xdg-autostart generator emits **nothing at all** for
/// a relative `Exec=`, silently, so a desktop entry saying `Exec=cleanroomd` produces no
/// unit and no error and no autostart.
fn own_exe() -> Option<String> {
    std::fs::read_link("/proc/self/exe")
        .ok()
        .map(|p| p.display().to_string())
}

/// The line to paste into a compositor config when nothing else works.
pub fn exec_once_line() -> String {
    format!(
        "exec-once = {}",
        own_exe().unwrap_or_else(|| "cleanroomd".into())
    )
}

/// What autostart is currently set to, and how it would be done.
pub struct Report {
    pub mechanism: Mechanism,
    pub enabled: bool,
    /// Non-empty only when the user has to do something by hand.
    pub instruction: String,
}

/// Decide which mechanism applies *now*, and report whether it is already in effect.
pub async fn status(connection: &zbus::Connection) -> Report {
    let mechanism = choose(connection).await;
    let enabled = match mechanism {
        Mechanism::SystemdUser => unit_is_wanted(connection).await,
        Mechanism::XdgAutostart => autostart_entry_path().is_some_and(|p| p.exists()),
        // Cannot be detected: we have no idea what is in someone's compositor config, and
        // guessing "on" would be a lie. Reported as off with the line to add.
        Mechanism::CompositorExecOnce => false,
    };
    Report {
        instruction: instruction_for(&mechanism),
        mechanism,
        enabled,
    }
}

fn instruction_for(m: &Mechanism) -> String {
    match m {
        Mechanism::CompositorExecOnce => format!(
            "This session neither activates graphical-session.target nor runs XDG \
             autostart, so add this to your compositor config:\n    {}",
            exec_once_line()
        ),
        _ => String::new(),
    }
}

/// Turn autostart on or off using whichever mechanism this session supports.
pub async fn set(connection: &zbus::Connection, on: bool) -> Result<Report, String> {
    let mechanism = choose(connection).await;

    match mechanism {
        Mechanism::SystemdUser => {
            set_unit(connection, on).await?;
        }
        Mechanism::XdgAutostart => {
            let path = autostart_entry_path().ok_or("no config directory to write into")?;
            if on {
                let exe = own_exe().ok_or("could not resolve our own path")?;
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                }
                std::fs::write(&path, desktop_entry(&exe)).map_err(|e| e.to_string())?;
            } else if path.exists() {
                std::fs::remove_file(&path).map_err(|e| e.to_string())?;
            }
        }
        Mechanism::CompositorExecOnce => {
            // Nothing to write. Editing somebody's compositor config from a daemon is not
            // a thing to do uninvited, so the line is returned for them to paste.
        }
    }

    Ok(Report {
        instruction: instruction_for(&mechanism),
        enabled: on && mechanism != Mechanism::CompositorExecOnce,
        mechanism,
    })
}

/// The desktop entry text.
///
/// Three keys are deliberately absent, each because including it breaks something:
///
/// * `OnlyShowIn`/`NotShowIn` — the registry has no entry for Hyprland, sway, niri or
///   river, so either key excludes every wlroots compositor.
/// * `X-GNOME-Autostart-Phase` — any value but `Application` is fatal on GNOME 49+, and its
///   presence makes systemd's xdg-autostart generator emit no unit at all.
fn desktop_entry(exe: &str) -> String {
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=Cleanroom\n\
         Comment=Camera and microphone effects\n\
         Exec={exe}\n\
         Icon={id}\n\
         Terminal=false\n\
         Categories=AudioVideo;Video;Settings;\n\
         StartupWMClass={id}\n\
         X-GNOME-Autostart-enabled=true\n",
        id = cleanroom_ipc::BUS_NAME
    )
}

const UNIT: &str = "cleanroomd.service";

/// systemd's per-file result triple: (type, unit path, destination). Named because the
/// bare tuple trips clippy's complexity lint and reads as noise at the call site.
type UnitFileChange = (String, String, String);

#[zbus::proxy(
    interface = "org.freedesktop.systemd1.Manager",
    default_service = "org.freedesktop.systemd1",
    default_path = "/org/freedesktop/systemd1"
)]
trait Systemd1Manager {
    fn get_unit(&self, name: &str) -> zbus::Result<zbus::zvariant::OwnedObjectPath>;

    fn enable_unit_files(
        &self,
        files: &[&str],
        runtime: bool,
        force: bool,
    ) -> zbus::Result<(bool, Vec<UnitFileChange>)>;

    fn disable_unit_files(
        &self,
        files: &[&str],
        runtime: bool,
    ) -> zbus::Result<Vec<UnitFileChange>>;

    fn start_unit(&self, name: &str, mode: &str) -> zbus::Result<zbus::zvariant::OwnedObjectPath>;

    fn reload(&self) -> zbus::Result<()>;
}

#[zbus::proxy(
    interface = "org.freedesktop.systemd1.Unit",
    default_service = "org.freedesktop.systemd1"
)]
trait Systemd1Unit {
    #[zbus(property)]
    fn active_state(&self) -> zbus::Result<String>;

    #[zbus(property)]
    fn wanted_by(&self) -> zbus::Result<Vec<String>>;
}

/// Is `graphical-session.target` actually active?
///
/// Asked rather than assumed, because the answer is "no" on the reference machine. Hyprland
/// launched from a TTY never activates it, so a unit wanted by that target would be enabled,
/// look correct in `systemctl --user is-enabled`, and never start.
async fn graphical_session_active(connection: &zbus::Connection) -> bool {
    let Ok(manager) = Systemd1ManagerProxy::new(connection).await else {
        return false;
    };
    let Ok(path) = manager.get_unit("graphical-session.target").await else {
        return false;
    };
    let Ok(unit) = Systemd1UnitProxy::builder(connection).path(path) else {
        return false;
    };
    let Ok(unit) = unit.build().await else {
        return false;
    };
    unit.active_state().await.as_deref() == Ok("active")
}

async fn choose(connection: &zbus::Connection) -> Mechanism {
    if graphical_session_active(connection).await {
        Mechanism::SystemdUser
    } else if desktop_honours_xdg_autostart() {
        Mechanism::XdgAutostart
    } else {
        Mechanism::CompositorExecOnce
    }
}

async fn unit_is_wanted(connection: &zbus::Connection) -> bool {
    let Ok(manager) = Systemd1ManagerProxy::new(connection).await else {
        return false;
    };
    let Ok(path) = manager.get_unit(UNIT).await else {
        return false;
    };
    let Ok(builder) = Systemd1UnitProxy::builder(connection).path(path) else {
        return false;
    };
    let Ok(unit) = builder.build().await else {
        return false;
    };
    unit.wanted_by()
        .await
        .map(|w| w.iter().any(|t| t == "graphical-session.target"))
        .unwrap_or(false)
}

async fn set_unit(connection: &zbus::Connection, on: bool) -> Result<(), String> {
    let manager = Systemd1ManagerProxy::new(connection)
        .await
        .map_err(|e| format!("reaching the systemd user manager: {e}"))?;

    if on {
        manager
            .enable_unit_files(&[UNIT], false, true)
            .await
            .map_err(|e| format!("enabling {UNIT}: {e}"))?;
        manager.reload().await.ok();
        // Started too, so "enable" means what a user expects rather than "next login".
        manager
            .start_unit(UNIT, "replace")
            .await
            .map_err(|e| format!("starting {UNIT}: {e}"))?;
    } else {
        manager
            .disable_unit_files(&[UNIT], false)
            .await
            .map_err(|e| format!("disabling {UNIT}: {e}"))?;
        manager.reload().await.ok();
        // Deliberately not stopped: turning off "start automatically" is a statement about
        // future logins, not a request to kill the camera someone is currently using.
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each of these keys is absent for a specific reason, and each was a real trap.
    #[test]
    fn the_desktop_entry_omits_the_keys_that_break_things() {
        let e = desktop_entry("/usr/bin/cleanroomd");
        assert!(
            !e.contains("OnlyShowIn") && !e.contains("NotShowIn"),
            "either key excludes every wlroots compositor, since the registry has no \
             entry for Hyprland, sway, niri or river"
        );
        assert!(
            !e.contains("X-GNOME-Autostart-Phase"),
            "any value but Application is fatal on GNOME 49+, and its presence makes the \
             systemd xdg-autostart generator emit no unit at all"
        );
    }

    /// A relative Exec= makes systemd's generator emit nothing, silently.
    #[test]
    fn the_desktop_entry_uses_an_absolute_exec() {
        let e = desktop_entry("/usr/bin/cleanroomd");
        assert!(e.contains("Exec=/usr/bin/cleanroomd"));
    }

    /// The basename, Icon, StartupWMClass and the GUI's app_id must all be the same string,
    /// or the window gets a generic icon in every taskbar and dock.
    #[test]
    fn the_desktop_entry_agrees_with_the_bus_name() {
        let e = desktop_entry("/usr/bin/cleanroomd");
        assert!(e.contains(&format!("Icon={}", cleanroom_ipc::BUS_NAME)));
        assert!(e.contains(&format!("StartupWMClass={}", cleanroom_ipc::BUS_NAME)));
        let p = autostart_entry_path().expect("a config dir in the test environment");
        assert_eq!(
            p.file_name().unwrap().to_string_lossy(),
            format!("{}.desktop", cleanroom_ipc::BUS_NAME)
        );
    }

    /// Colon-separated because that is what real sessions set, e.g. "ubuntu:GNOME".
    #[test]
    fn a_compound_desktop_name_is_still_recognised() {
        // SAFETY: single-threaded test, and the variable is read only through this helper.
        unsafe { std::env::set_var("XDG_CURRENT_DESKTOP", "ubuntu:GNOME") };
        assert!(desktop_honours_xdg_autostart());

        unsafe { std::env::set_var("XDG_CURRENT_DESKTOP", "Hyprland") };
        assert!(
            !desktop_honours_xdg_autostart(),
            "Hyprland runs no XDG autostart; claiming otherwise writes a file that is \
             silently ignored"
        );

        unsafe { std::env::remove_var("XDG_CURRENT_DESKTOP") };
        assert!(!desktop_honours_xdg_autostart(), "unset means unknown");
    }

    #[test]
    fn the_exec_once_line_is_pasteable() {
        let l = exec_once_line();
        assert!(l.starts_with("exec-once = "), "{l}");
    }
}
