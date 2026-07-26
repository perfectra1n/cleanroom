//! Downloading the model weights.
//!
//! Neither set is vendored, for different reasons, and both reasons are worth knowing
//! before running this:
//!
//! * **Robust Video Matting** is GPL-3.0, which is compatible with this project. It is not
//!   vendored because the mobilenetv3 export is ~15 MB and a git repository is the wrong
//!   place for it.
//! * **DeepFilterNet's** weights carry *no licence grant at all*. The upstream README
//!   licenses "all **code** in this repository", and weights are not code.
//!   [Issue #697](https://github.com/Rikorose/DeepFilterNet/issues/697) asks exactly this
//!   and has been unanswered since July 2026, on a repository dormant since October 2024.
//!   Debian and nixpkgs both redistribute them, so the practical risk is low — but that is
//!   an inference from silence, not a grant, and it is the user's call to make rather than
//!   ours to make quietly on their behalf.
//!
//! This is why fetching is a deliberate command rather than something a package does. It
//! also means every distro gets the same answer, instead of each packager reaching a
//! different conclusion about redistribution.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::PathBuf;

struct Model {
    name: &'static str,
    file: &'static str,
    url: &'static str,
    /// Pinned. A weights file that silently changed is a model that silently changed, and
    /// "the matte got worse after I reinstalled" is not a debuggable report.
    sha256: &'static str,
    licence: &'static str,
    /// Whether the licensing position needs explicit acknowledgement.
    needs_consent: bool,
}

const MODELS: &[Model] = &[
    Model {
        name: "Robust Video Matting (mobilenetv3, fp32)",
        file: "rvm_mobilenetv3_fp32.onnx",
        url: "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        sha256: "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
        licence: "GPL-3.0, same as Cleanroom.",
        needs_consent: false,
    },
    Model {
        name: "DeepFilterNet3 (ONNX)",
        file: "DeepFilterNet3_onnx.tar.gz",
        // Not a release asset. Upstream keeps the weights in the repository tree, and the
        // obvious releases/download/v0.5.6/... URL is a 404 — checked.
        url: "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3_onnx.tar.gz",
        sha256: "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
        licence: "NO licence grant. Upstream licenses the code; weights are not code. \
                  Issue #697 asks for clarification and is unanswered since July 2026.",
        needs_consent: true,
    },
];

/// Where weights are looked for at runtime, and so where they are written.
///
/// Matches `cleanroom_matting::find_model` and `cleanroom_audio::find_model` — writing
/// somewhere they do not look would be a download that appears to succeed and changes
/// nothing.
fn target_dir() -> Result<PathBuf> {
    if let Some(d) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(d).join("cleanroom"));
    }
    let home = std::env::var_os("HOME").context("neither XDG_DATA_HOME nor HOME is set")?;
    Ok(PathBuf::from(home).join(".local/share/cleanroom"))
}

pub fn run(assume_yes: bool, force: bool) -> Result<()> {
    let dir = target_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    println!("Model weights are downloaded to {}\n", dir.display());

    for m in MODELS {
        let dest = dir.join(m.file);
        if dest.exists() && !force {
            println!("[skip] {} — already present at {}", m.name, dest.display());
            continue;
        }

        println!("{}", m.name);
        println!("  licence: {}", m.licence);

        if m.needs_consent && !assume_yes && !confirm()? {
            println!("  skipped.\n");
            continue;
        }

        println!("  downloading {} ...", m.url);
        let bytes = download(m.url).with_context(|| format!("downloading {}", m.url))?;

        let got = hex(&Sha256::digest(&bytes));
        if got != m.sha256 {
            // Refuse rather than warn. A mismatch is either a corrupted download or a
            // changed artefact, and both are things to stop on — silently accepting the
            // second means the model can change under a user who never chose that.
            bail!(
                "checksum mismatch for {}\n  expected {}\n  got      {}\n\
                 Refusing to install. If upstream legitimately republished this file, the \
                 pin in crates/cleanroom-ctl/src/fetch.rs needs updating deliberately.",
                m.file,
                m.sha256,
                got
            );
        }

        // Write to a temporary and rename, so an interrupted download cannot leave a
        // half-written file where the daemon will find it and try to load it.
        let tmp = dest.with_extension("part");
        std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &dest).with_context(|| format!("installing {}", dest.display()))?;
        println!("  installed {} ({} bytes)\n", dest.display(), bytes.len());
    }

    println!("Done. `cleanroom-ctl doctor` will confirm what is present.");
    Ok(())
}

fn confirm() -> Result<bool> {
    print!("  Download anyway? [y/N] ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn download(url: &str) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    ureq::get(url)
        .call()?
        .into_body()
        .into_reader()
        .read_to_end(&mut body)?;
    Ok(body)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writing somewhere the daemon does not look would be a download that appears to
    /// succeed and changes nothing, which is a particularly annoying failure to diagnose.
    #[test]
    fn the_download_directory_is_where_the_daemon_looks() {
        let dir = target_dir().expect("HOME or XDG_DATA_HOME in the test environment");
        let s = dir.display().to_string();
        assert!(
            s.ends_with("/cleanroom"),
            "must be the cleanroom data dir, got {s}"
        );
        assert!(
            s.contains(".local/share") || std::env::var_os("XDG_DATA_HOME").is_some(),
            "must match find_model's search path, got {s}"
        );
    }

    /// The DeepFilterNet licence position is the reason this command exists rather than
    /// being something a package does silently. If the prompt ever stops being required,
    /// that has to be a deliberate edit and not an accident.
    #[test]
    fn deepfilternet_requires_explicit_consent() {
        let dfn = MODELS
            .iter()
            .find(|m| m.file.contains("DeepFilterNet"))
            .expect("DeepFilterNet must be listed");
        assert!(dfn.needs_consent);
        assert!(
            dfn.licence.contains("NO licence grant"),
            "the licence line must state the position plainly: {}",
            dfn.licence
        );

        let rvm = MODELS
            .iter()
            .find(|m| m.file.contains("rvm"))
            .expect("RVM must be listed");
        assert!(
            !rvm.needs_consent,
            "RVM is GPL-3.0 and compatible; prompting for it trains people to say yes"
        );
    }

    #[test]
    fn every_model_pins_a_full_sha256() {
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{} has a malformed digest", m.file);
            assert!(m.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}
