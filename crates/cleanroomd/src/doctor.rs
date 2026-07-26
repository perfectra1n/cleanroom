//! Environment checks.
//!
//! This is the highest-leverage support tool in the project. Nearly every way Cleanroom
//! can appear broken is an environment problem that produces a *misleading* symptom:
//! a camera that is invisible to Chrome, a virtual mic nothing lists, a GPU path that
//! quietly is not being used, a kernel module that vanished after an upgrade. Each of
//! those looks like "the app is broken" and is not.
//!
//! So each check names the symptom it explains, not just the condition it tests.
//!
//! Everything here is read-only. `doctor` diagnoses; it never reconfigures the system.

use cleanroom_core::Config;
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Ok,
    Info,
    /// Works, but something will bite later or is not what the user expects.
    Warn,
    /// Will not work as configured.
    Fail,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Level::Ok => "OK  ",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Fail => "FAIL",
        })
    }
}

#[derive(Debug, Clone)]
pub struct Check {
    pub level: Level,
    pub name: String,
    pub detail: String,
    /// What to actually do about it. Absent when there is nothing to do.
    pub fix: Option<String>,
}

impl Check {
    fn new(level: Level, name: &str, detail: impl Into<String>) -> Self {
        Self {
            level,
            name: name.into(),
            detail: detail.into(),
            fix: None,
        }
    }
    fn ok(name: &str, detail: impl Into<String>) -> Self {
        Self::new(Level::Ok, name, detail)
    }
    fn info(name: &str, detail: impl Into<String>) -> Self {
        Self::new(Level::Info, name, detail)
    }
    fn warn(name: &str, detail: impl Into<String>) -> Self {
        Self::new(Level::Warn, name, detail)
    }
    fn fail(name: &str, detail: impl Into<String>) -> Self {
        Self::new(Level::Fail, name, detail)
    }
    fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }
}

impl fmt::Display for Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {:<28} {}", self.level, self.name, self.detail)?;
        if let Some(fix) = &self.fix {
            write!(f, "\n              -> {fix}")?;
        }
        Ok(())
    }
}

pub fn run(config: &Config) -> Vec<Check> {
    let mut out = Vec::new();
    out.extend(check_gpu(config));
    out.extend(check_v4l2loopback());
    out.extend(check_secure_boot());
    out.extend(check_cameras());
    out.extend(check_pipewire());
    out.extend(check_realtime());
    out.extend(check_browsers());
    out
}

// --- gpu ----------------------------------------------------------------------------

fn check_gpu(config: &Config) -> Vec<Check> {
    let mut out = Vec::new();

    let icd_dirs = [
        "/run/opengl-driver/share/vulkan/icd.d",
        "/usr/share/vulkan/icd.d",
        "/etc/vulkan/icd.d",
    ];
    let mut icds = Vec::new();
    for d in icd_dirs {
        if let Ok(entries) = std::fs::read_dir(d) {
            for e in entries.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.ends_with(".json") && !n.contains("i686") {
                    icds.push(n);
                }
            }
        }
    }

    if icds.is_empty() {
        out.push(Check::fail("vulkan icds", "no Vulkan ICD found").with_fix(
            "install your GPU's Vulkan driver (mesa-vulkan-drivers, or the NVIDIA driver)",
        ));
    } else {
        icds.sort();
        icds.dedup();
        out.push(Check::ok("vulkan icds", icds.join(", ")));
    }

    // Render nodes. Naming which one we will use matters: "adapter 0" on a machine with
    // a dGPU and an iGPU is a coin flip, and on the reference machine the difference is
    // 4.5ms vs 38.8ms per frame.
    let mut nodes = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/dev/dri") {
        for e in entries.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("renderD") {
                nodes.push(format!("/dev/dri/{n}"));
            }
        }
    }
    nodes.sort();

    match nodes.len() {
        0 => out.push(
            Check::fail("drm render nodes", "none found — no GPU is usable")
                .with_fix("check that your user is in the 'render' or 'video' group"),
        ),
        1 => out.push(Check::ok("drm render nodes", nodes[0].clone())),
        _ => {
            let chosen = config
                .gpu
                .render_node
                .as_ref()
                .map(|p| p.display().to_string());
            match chosen {
                Some(c) => out.push(Check::ok(
                    "drm render nodes",
                    format!("{} (pinned to {c})", nodes.join(", ")),
                )),
                None => out.push(
                    Check::warn(
                        "drm render nodes",
                        format!("{} present, none pinned", nodes.join(", ")),
                    )
                    .with_fix(
                        "several GPUs are available and the pipeline will pick automatically. \
                         If it picks the slow one, set gpu.render_node explicitly: \
                         `cleanroom-ctl set gpu.render_node /dev/dri/renderD128`",
                    ),
                ),
            }
        }
    }

    out
}

// --- v4l2loopback -------------------------------------------------------------------

fn check_v4l2loopback() -> Vec<Check> {
    let mut out = Vec::new();

    if !Path::new("/sys/module/v4l2loopback").exists() {
        out.push(
            Check::fail("v4l2loopback", "kernel module not loaded").with_fix(
                "the virtual camera needs it: `sudo modprobe v4l2loopback`. If that fails \
                     after a kernel upgrade, the DKMS build did not run — reinstall \
                     v4l2loopback-dkms. On NixOS add it to boot.extraModulePackages.",
            ),
        );
        return out;
    }

    let version = std::fs::read_to_string("/sys/module/v4l2loopback/version")
        .unwrap_or_default()
        .trim()
        .to_string();

    // The control device is what lets us allocate a device at runtime instead of
    // hardcoding /dev/video10 and fighting OBS and every other producer for it. Added in
    // v4l2loopback 0.13.
    if Path::new("/dev/v4l2loopback").exists() {
        out.push(Check::ok(
            "v4l2loopback",
            format!("version {version}, control device present (dynamic allocation available)"),
        ));
    } else {
        out.push(
            Check::warn(
                "v4l2loopback",
                format!("version {version}, no /dev/v4l2loopback control device"),
            )
            .with_fix(
                "without the control device (v4l2loopback < 0.13) we cannot allocate our own \
                 node and must share a fixed one, which collides with OBS's virtual camera. \
                 Upgrade v4l2loopback if you can.",
            ),
        );
    }

    // A global `options v4l2loopback` line is a footgun worth naming: modprobe merges
    // those across every package that ships one, and video_nr/card_label/exclusive_caps
    // are parallel arrays, so two apps each declaring their own device produce
    // undefined-order garbage rather than two devices.
    let mut conflicting = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/etc/modprobe.d") {
        for e in entries.flatten() {
            if let Ok(text) = std::fs::read_to_string(e.path())
                && text.contains("v4l2loopback")
                && text.contains("options")
            {
                conflicting.push(e.file_name().to_string_lossy().to_string());
            }
        }
    }
    if conflicting.len() > 1 {
        out.push(
            Check::warn(
                "v4l2loopback modprobe",
                format!("multiple options files: {}", conflicting.join(", ")),
            )
            .with_fix(
                "modprobe merges these into one argument list where video_nr/card_label/\
                 exclusive_caps are parallel arrays. Two apps each declaring a device \
                 usually yields neither. Prefer runtime allocation via /dev/v4l2loopback.",
            ),
        );
    }

    out
}

// --- secure boot --------------------------------------------------------------------

fn check_secure_boot() -> Vec<Check> {
    // Secure Boot refusing an unsigned DKMS module is the single largest support burden
    // for anything that needs v4l2loopback, and its symptom — "Key was rejected by
    // service" — does not mention Secure Boot at all.
    let lockdown = std::fs::read_to_string("/sys/kernel/security/lockdown").unwrap_or_default();
    let locked = lockdown.contains("[integrity]") || lockdown.contains("[confidentiality]");
    let module_loaded = Path::new("/sys/module/v4l2loopback").exists();

    if !locked {
        return vec![Check::ok("secure boot", "not enforcing module signatures")];
    }
    if module_loaded {
        return vec![Check::ok(
            "secure boot",
            "kernel lockdown active, but v4l2loopback is loaded — it must already be signed",
        )];
    }
    vec![
        Check::fail(
            "secure boot",
            "kernel lockdown is active and v4l2loopback is not loaded",
        )
        .with_fix(
            "an unsigned DKMS module cannot load under Secure Boot. Enroll a MOK: \
             Debian/Ubuntu sign automatically but you must accept the key at the blue \
             MokManager screen on reboot; Fedora needs `kmodgenca` then \
             `mokutil --import /etc/pki/akmods/certs/public_key.der`; Arch has no automatic \
             path and needs a sign-file pacman hook.",
        ),
    ]
}

// --- cameras ------------------------------------------------------------------------

fn check_cameras() -> Vec<Check> {
    use cleanroom_video::NodeKind;

    let mut out = Vec::new();
    let all = cleanroom_video::enumerate();

    let usable: Vec<String> = all
        .iter()
        .filter(|d| d.is_usable_input())
        .map(|d| format!("{} ({})", d.path.display(), d.card))
        .collect();

    // Everything we deliberately will not offer as an input, and why. Naming the reason
    // matters: "my camera isn't listed" is a common report, and for a UVC metadata node
    // the honest answer is "that node cannot capture video, use the other one".
    let excluded: Vec<String> = all
        .iter()
        .filter(|d| !d.is_usable_input())
        .map(|d| {
            let why = if d.is_virtual {
                "virtual camera"
            } else {
                match d.kind {
                    NodeKind::Metadata => "metadata node, cannot capture",
                    NodeKind::Output => "output only",
                    NodeKind::Capture => "unavailable",
                    NodeKind::Other => "not a video node",
                }
            };
            format!("{} ({why})", d.path.display())
        })
        .collect();

    if usable.is_empty() {
        out.push(
            Check::warn("cameras", "no usable capture device found").with_fix(
                "check that a camera is plugged in and that your user is in the 'video' group",
            ),
        );
    } else {
        out.push(Check::ok("cameras", usable.join("; ")));
    }

    if !excluded.is_empty() {
        out.push(Check::info("cameras (not offered)", excluded.join("; ")));
    }

    // An inaccessible node is nearly always another process holding the camera, which is
    // the single most common cause of "Cleanroom says the camera is busy".
    let busy: Vec<String> = all
        .iter()
        .filter(|d| !d.accessible)
        .map(|d| d.path.display().to_string())
        .collect();
    if !busy.is_empty() {
        out.push(Check::warn("cameras (in use)", busy.join(", ")).with_fix(
            "another process holds these. Common culprits: a browser tab with camera \
                 permission, OBS, or PipeWire itself streaming the v4l2 node.",
        ));
    }

    out
}

// --- pipewire -----------------------------------------------------------------------

fn check_pipewire() -> Vec<Check> {
    let mut out = Vec::new();

    let runtime = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/run/user/1000".into());
    let sock = Path::new(&runtime).join("pipewire-0");
    if !sock.exists() {
        out.push(
            Check::fail("pipewire", format!("no socket at {}", sock.display()))
                .with_fix("start PipeWire: `systemctl --user start pipewire`"),
        );
        return out;
    }
    out.push(Check::ok(
        "pipewire",
        format!("socket at {}", sock.display()),
    ));

    // The libcamera monitor double-enumerates UVC cameras, exposes only RAW/YUYV modes,
    // and holds the device fd open. Since we open /dev/video* directly for MJPEG, that
    // contention is a real problem rather than a cosmetic one.
    out.push(
        Check::info(
            "pipewire camera",
            "Cleanroom opens the camera directly rather than through PipeWire",
        )
        .with_fix(
            "PipeWire's v4l2 node advertises YUY2 only and does not pass a UVC camera's MJPG \
         modes through, which pins 1080p to about 5fps over USB2. If capture reports the \
         device is busy, PipeWire is holding it: disable the node with a WirePlumber rule \
         matching node.nick (NOT media.class — that property does not exist yet when the \
         create-node hook evaluates rules, so such a rule silently never fires).",
        ),
    );

    out
}

// --- realtime -----------------------------------------------------------------------

fn check_realtime() -> Vec<Check> {
    // The audio thread wants RT priority. PAM's limits.conf does NOT apply to systemd
    // user units, so an RLIMIT_RTPRIO of 0 is normal here and is not itself a failure —
    // rtkit or the Realtime portal is the supported route.
    let rtkit = Path::new("/usr/share/dbus-1/system-services/org.freedesktop.RealtimeKit1.service")
        .exists()
        || Path::new("/run/current-system/sw/share/dbus-1/system-services/org.freedesktop.RealtimeKit1.service").exists()
        || Path::new("/proc/self").exists() && rtkit_running();

    if rtkit {
        vec![Check::ok(
            "realtime scheduling",
            "rtkit is available; the audio thread will request RT priority through it",
        )]
    } else {
        vec![
            Check::warn("realtime scheduling", "rtkit not detected").with_fix(
                "without RT priority the audio thread can be preempted and glitch under load. \
                 Install rtkit, or raise DefaultLimitRTPRIO in /etc/systemd/system.conf. \
                 Note PAM limits.conf does not apply to systemd user units.",
            ),
        ]
    }
}

fn rtkit_running() -> bool {
    std::fs::read_dir("/proc")
        .map(|entries| {
            entries.flatten().any(|e| {
                std::fs::read_to_string(e.path().join("comm"))
                    .map(|c| c.trim() == "rtkit-daemon")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

// --- browsers -----------------------------------------------------------------------

fn check_browsers() -> Vec<Check> {
    let mut out = Vec::new();

    // Firefox: where a distro has flipped PipeWire camera support on (Fedora ships it
    // that way), Firefox goes through the camera portal and may not see a v4l2loopback
    // device at all — PipeWire does not create source nodes when a device's capabilities
    // change, which is exactly what exclusive_caps=1 does when a producer attaches.
    let home = std::env::var("HOME").unwrap_or_default();
    let ff_profiles = format!("{home}/.mozilla/firefox");
    if Path::new(&ff_profiles).exists() {
        out.push(
            Check::info(
                "firefox",
                "installed — if Cleanroom's camera is missing there but works elsewhere",
            )
            .with_fix(
                "set media.webrtc.camera.allow-pipewire=false in about:config, or select the \
             PipeWire-published 'Cleanroom Camera' node instead of the v4l2loopback one.",
            ),
        );
    }

    // Chromium/Electron: needs exclusive_caps=1 to accept a loopback node as a camera,
    // and enumerates cameras once at startup — so the device must exist before the app
    // launches, which is why the daemon keeps a producer attached permanently.
    out.push(
        Check::info(
            "chromium / electron",
            "Zoom, Discord, Teams and Chrome all enumerate cameras once at startup",
        )
        .with_fix(
            "start Cleanroom before them. If the camera is listed but black, the producer was \
         not attached when they enumerated — restart the app, not the daemon.",
        ),
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_runs_without_panicking_and_says_something() {
        // It must be safe to run on any machine, including one missing every optional
        // piece — a diagnostic that crashes is worse than no diagnostic.
        let checks = run(&Config::default());
        assert!(!checks.is_empty());
        for c in &checks {
            assert!(!c.name.is_empty());
            assert!(!c.detail.is_empty(), "check '{}' has no detail", c.name);
        }
    }

    #[test]
    fn every_actionable_check_says_what_to_do() {
        // A Fail or Warn with no fix is a dead end for the user.
        for c in run(&Config::default()) {
            if matches!(c.level, Level::Fail | Level::Warn) {
                assert!(
                    c.fix.is_some(),
                    "actionable check '{}' has no fix hint",
                    c.name
                );
            }
        }
    }

    #[test]
    fn display_includes_the_fix() {
        let c = Check::warn("thing", "detail").with_fix("do the thing");
        let s = c.to_string();
        assert!(s.contains("WARN"));
        assert!(s.contains("do the thing"));
    }
}
