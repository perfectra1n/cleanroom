//! Guards for the things that break with **no error at all**.
//!
//! docs/pitfalls.md keeps a table of changes that fail silently. Several already have unit
//! tests next to the code they protect — the YUY2 output constant, `device_caps` over
//! `capabilities`, unknown-consumers-as-in-use, the DeepFilterNet attenuation clamp, the
//! `v4l2_event` struct offsets. The rest live in *files*, not in Rust, so nothing in the
//! build has any opinion about them until here.
//!
//! These are in cleanroom-core only because it is the crate every other one depends on and
//! it has no hardware requirements; they are about the repository, not about this crate.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/cleanroom-core.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
}

/// A WirePlumber rule matching `media.class` **silently never fires**.
///
/// That property does not exist yet when the v4l2 `create-node` hook evaluates rules, so
/// the rule is valid, loads without complaint, and does nothing. The valid keys there are
/// `node.name`, `node.nick`, `node.description` and `api.v4l2.path`.
#[test]
fn the_wireplumber_camera_rule_matches_node_nick_not_media_class() {
    let conf = read("packaging/wireplumber/51-cleanroom-camera.conf");
    assert!(
        conf.contains("node.nick"),
        "the rule must match node.nick, or it never fires"
    );

    // Only the commentary may mention media.class, and only to warn about it.
    for (n, line) in conf.lines().enumerate() {
        let code = line.split('#').next().unwrap_or("");
        assert!(
            !code.contains("media.class"),
            "line {}: matching media.class in a v4l2 create-node rule silently never \
             fires — that property is not set yet when the hook runs",
            n + 1
        );
    }
}

/// `Restart=always` turns a normal event into an endless restart loop.
///
/// The single-instance guard exits **0** when it loses the name race, because launching a
/// second copy is a normal thing for a user to do. `always` restarts on a clean exit too.
#[test]
fn the_unit_restarts_on_failure_never_always() {
    // Directives only. The unit's own comments explain why Restart=always is wrong, and a
    // naive `contains` matches that explanation and fails on a correct file — which is
    // exactly what the first version of this test did.
    let unit = read("packaging/systemd/cleanroomd.service");
    let directives: Vec<&str> = unit
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with(';'))
        .collect();

    assert!(
        directives.iter().any(|l| *l == "Restart=on-failure"),
        "the unit must specify Restart=on-failure"
    );
    assert!(
        !directives.iter().any(|l| l.starts_with("Restart=always")),
        "Restart=always plus a single-instance guard that exits 0 is a restart loop"
    );
    // With lingering, this starts a device-holding daemon at boot with no session.
    assert!(
        !directives.iter().any(|l| *l == "WantedBy=default.target"),
        "WantedBy=default.target starts the daemon at boot with no session"
    );
}

/// Three `.desktop` keys each break something, and none of them produce an error.
#[test]
fn the_desktop_entry_omits_the_keys_that_silently_break_things() {
    let desktop = read("packaging/desktop/io.github.perfectra1n.Cleanroom.desktop");
    for (n, line) in desktop.lines().enumerate() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        assert!(
            !line.starts_with("OnlyShowIn") && !line.starts_with("NotShowIn"),
            "line {}: the registry has no entry for Hyprland, sway, niri or river, so \
             either key excludes every wlroots compositor",
            n + 1
        );
        assert!(
            !line.starts_with("X-GNOME-Autostart-Phase"),
            "line {}: any value but Application is fatal on GNOME 49+, and its presence \
             makes systemd's xdg-autostart generator emit no unit at all",
            n + 1
        );
    }
}

/// The `.desktop` basename, its `Icon=`, its `StartupWMClass=` and the GUI's `app_id` must
/// all be the same string. Wayland has no `WM_CLASS`, so `app_id` is the only identity a
/// compositor gets, and a mismatch means a generic icon everywhere.
#[test]
fn the_desktop_identity_agrees_with_the_bus_name() {
    const ID: &str = "io.github.perfectra1n.Cleanroom";
    let desktop = read(&format!("packaging/desktop/{ID}.desktop"));
    assert!(desktop.contains(&format!("Icon={ID}")), "Icon must be {ID}");
    assert!(
        desktop.contains(&format!("StartupWMClass={ID}")),
        "StartupWMClass must be {ID}"
    );

    let gui = read("crates/cleanroom-gui/src/main.rs");
    assert!(
        gui.contains(&format!("APP_ID: &str = \"{ID}\"")),
        "the GUI's APP_ID must match the .desktop basename byte for byte"
    );

    let unit = read("packaging/systemd/cleanroomd.service");
    assert!(unit.contains(&format!("BusName={ID}")), "BusName must be {ID}");
}

/// wgpu is pinned to 29 by Slint, which offers `unstable-wgpu-28` and `-29` and nothing
/// newer. Bumping it stops the texture types unifying across the GUI boundary — which is a
/// compile error rather than a silent one, but a confusing one that reads as a Slint bug.
#[test]
fn wgpu_stays_pinned_to_the_version_slint_can_interop_with() {
    let gpu = read("crates/cleanroom-gpu/Cargo.toml");
    assert!(
        gpu.contains("wgpu = \"29\"") || gpu.contains("wgpu = { version = \"29\""),
        "wgpu must stay at 29 until Slint offers a newer unstable-wgpu-* feature"
    );
}

/// zbus's `tokio` feature switches its executor **globally** for the whole build graph,
/// which makes `accesskit_unix` panic and silently removes screen-reader support.
///
/// Checked as text across every manifest, because the failure is a runtime panic in a
/// dependency and nothing in a normal build says a word about it.
#[test]
fn no_manifest_enables_zbus_with_the_tokio_feature() {
    for crate_dir in [
        "cleanroom-core",
        "cleanroom-ipc",
        "cleanroom-audio",
        "cleanroom-video",
        "cleanroom-gpu",
        "cleanroom-matting",
        "cleanroomd",
        "cleanroom-ctl",
        "cleanroom-gui",
    ] {
        let path = format!("crates/{crate_dir}/Cargo.toml");
        let manifest = read(&path);
        for (n, line) in manifest.lines().enumerate() {
            let code = line.split('#').next().unwrap_or("");
            if !code.contains("zbus") {
                continue;
            }
            assert!(
                !code.contains("tokio"),
                "{path} line {}: enabling zbus's tokio feature switches its executor \
                 globally and makes accesskit_unix panic, silently disabling screen-reader \
                 support",
                n + 1
            );
        }
    }
}

/// modprobe merges every `options` line for a module into ONE argument list, in which
/// `video_nr`, `card_label` and `exclusive_caps` are parallel arrays. Two packages each
/// declaring a device usually yields neither, so ours must not claim `video_nr`.
#[test]
fn the_modprobe_example_does_not_pin_a_device_number() {
    let conf = read("packaging/modprobe-v4l2loopback.conf");
    for (n, line) in conf.lines().enumerate() {
        let code = line.split('#').next().unwrap_or("");
        if code.trim().is_empty() {
            continue;
        }
        assert!(
            !code.contains("video_nr"),
            "line {}: pinning video_nr fights every other package that ships an options \
             line for this module",
            n + 1
        );
    }
    assert!(
        conf.contains("exclusive_caps=1"),
        "exclusive_caps=1 is what makes the node flip to a capture device for consumers"
    );
}
