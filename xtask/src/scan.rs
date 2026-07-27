//! Finding the code, in an order that never changes.
//!
//! The ledger has to be byte-identical for identical source on any machine, or `record`
//! produces spurious diffs and people stop trusting it. Every drift vector is closed here
//! rather than hoped away:
//!
//! * **Walk order** — `walkdir` yields in filesystem order, which differs between machines
//!   and filesystems. Results are collected and sorted by their `/`-joined relative path,
//!   compared as bytes so no locale or Unicode collation is involved.
//! * **Symlinks** — not followed. The repo root carries a `result -> /nix/store/...`
//!   symlink after any local `nix build`, and descending into the Nix store would be
//!   memorable.
//! * **Map iteration** — `BTreeMap`/`BTreeSet` only, never `HashMap`.
//! * **Floats** — there are none anywhere in this tool, so there is nothing to round.
//! * **Path separators** — built from `Path::components()` and joined with `/`, never
//!   `to_string_lossy()` on a platform path.
//! * **`#[cfg]`** — never evaluated. The measurement therefore does not depend on features,
//!   target or toolchain, which is also why `#[cfg(test)]` is *classified* rather than
//!   conditionally compiled away.
//!
//! There is deliberately no timestamp field. It would be useful exactly once and would
//! make every re-record a diff. `git blame` already answers "when".
//!
//! # Scope
//!
//! `crates/*/src/**/*.rs` and nothing else. Not `tests/`, not `examples/`, not `benches/`,
//! not `build.rs`, not `spikes/`.
//!
//! Test code has different economics — a 200-line table-driven test is good, and
//! `.unwrap()` in a test *is* the loud failure you want. A tool that argues against writing
//! tests gets switched off. The spikes are excluded on the flake's own reasoning: it
//! already calls them "not products", and blocking an edit to a go/no-go probe on a
//! product complexity budget is pure cry-wolf. Measured effect of this one decision on
//! this tree: the debt counters drop from ~130 to ~37.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use syn::visit::Visit;
use walkdir::WalkDir;

use crate::baseline::{Limits, Subject, Totals};
use crate::metrics::{FileMetrics, FileWalker, FnMetrics, Source, Unmeasured};

pub struct Census {
    pub fns: Vec<FnMetrics>,
    pub files: Vec<FileMetrics>,
    pub unmeasured: Vec<Unmeasured>,
}

impl Census {
    pub fn totals(&self) -> Totals {
        Totals {
            allow_lints: self.files.iter().map(|f| f.allow_lints).sum(),
            unsafe_blocks: self.files.iter().map(|f| f.unsafe_blocks).sum(),
            unwraps: self.files.iter().map(|f| f.unwraps).sum(),
            panics: self.files.iter().map(|f| f.panics).sum(),
        }
    }

    /// Everything the ratchet has an opinion about, in a stable order: every function,
    /// then every file. Functions and whole files ratchet through the same code path.
    pub fn subjects(&self, limits: &Limits) -> Vec<Subject> {
        let mut out: Vec<Subject> = self
            .fns
            .iter()
            .map(|f| Subject::from_fn(f, limits))
            .collect();
        out.extend(
            self.files
                .iter()
                .map(|f| Subject::from_file(&f.path, f.lines, limits)),
        );
        out
    }

    pub fn unmeasured_set(&self) -> BTreeSet<(String, String)> {
        self.unmeasured
            .iter()
            .map(|u| (u.file.clone(), u.mac.clone()))
            .collect()
    }
}

/// Walk up from the current directory to the one holding the workspace manifest.
///
/// The same trick `crates/cleanroom-core/tests/packaging_invariants.rs` uses, so the tool
/// works from any subdirectory rather than only from the root.
pub fn repo_root() -> Result<PathBuf> {
    let mut dir = std::env::current_dir()?;
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file() {
            let text = std::fs::read_to_string(&manifest)?;
            if text.contains("[workspace]") {
                return Ok(dir);
            }
        }
        if !dir.pop() {
            bail!("no workspace Cargo.toml above the current directory");
        }
    }
}

/// Read the workspace members, keep the ones under `crates/`.
///
/// Not `cargo metadata`: that resolves the entire ONNX/Slint/wgpu graph and needs the dev
/// shell just to enumerate files, for information the root manifest already states. Not
/// `git ls-files` either — a subprocess, and wrong on a dirty tree.
fn member_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let text = std::fs::read_to_string(root.join("Cargo.toml"))?;
    let doc: toml::Value = toml::from_str(&text).context("parsing the workspace Cargo.toml")?;
    let members = doc
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .context("no [workspace] members")?;

    let mut dirs: Vec<PathBuf> = members
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|m| m.starts_with("crates/"))
        .map(|m| root.join(m))
        .collect();
    dirs.sort();
    Ok(dirs)
}

fn crate_ident(dir: &Path) -> Result<String> {
    let text = std::fs::read_to_string(dir.join("Cargo.toml"))?;
    let doc: toml::Value = toml::from_str(&text)?;
    let name = doc
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .with_context(|| format!("no package.name in {}", dir.display()))?;
    Ok(name.replace('-', "_"))
}

/// `src/frame.rs` -> `frame`; `src/lib.rs` and `src/main.rs` -> ``; `src/a/mod.rs` -> `a`;
/// `src/a/b.rs` -> `a::b`.
fn module_path(rel_to_src: &Path) -> String {
    let mut parts: Vec<String> = rel_to_src
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if let Some(last) = parts.pop() {
        let stem = last.trim_end_matches(".rs");
        if !matches!(stem, "lib" | "main" | "mod") {
            parts.push(stem.to_string());
        }
    }
    parts.join("::")
}

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

pub fn scan(root: &Path) -> Result<Census> {
    let mut fns = Vec::new();
    let mut files = Vec::new();
    let mut unmeasured = Vec::new();

    for dir in member_dirs(root)? {
        let src = dir.join("src");
        if !src.is_dir() {
            continue;
        }
        let ident = crate_ident(&dir)?;

        // Collect then sort: walkdir's own order is filesystem order.
        let mut paths: Vec<PathBuf> = WalkDir::new(&src)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .filter(|p| p.extension().is_some_and(|x| x == "rs"))
            .collect();
        paths.sort_by_key(|p| rel(root, p).into_bytes());

        for path in paths {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;

            // A file we cannot parse must never score zero. That is precisely the silent
            // degradation this project exists to not reproduce.
            let parsed = syn::parse_file(&text).with_context(|| {
                format!(
                    "parsing {} — the ratchet cannot measure it, and \
                                          scoring it zero would be a silent hole",
                    rel(root, &path)
                )
            })?;

            let source = Source::new(&text);
            let module = module_path(path.strip_prefix(&src).unwrap_or(&path));
            let mut walker = FileWalker::new(&source, rel(root, &path), &ident, &module);
            walker.visit_file(&parsed);

            fns.append(&mut walker.fns);
            unmeasured.append(&mut walker.unmeasured);
            walker.file.lines = source.effective_outside(&walker.excluded);
            files.push(walker.file);
        }
    }

    fns.sort_by(|a, b| (&a.file, &a.key).cmp(&(&b.file, &b.key)));
    files.sort_by(|a, b| a.path.cmp(&b.path));
    unmeasured.sort();
    unmeasured.dedup();

    Ok(Census {
        fns,
        files,
        unmeasured,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_paths_collapse_lib_main_and_mod() {
        assert_eq!(module_path(Path::new("frame.rs")), "frame");
        assert_eq!(module_path(Path::new("lib.rs")), "");
        assert_eq!(module_path(Path::new("main.rs")), "");
        assert_eq!(module_path(Path::new("a/mod.rs")), "a");
        assert_eq!(module_path(Path::new("a/b.rs")), "a::b");
    }

    /// The whole point of sorting the walk: two runs over the same tree must agree, and so
    /// must a run whose input arrived in a different order.
    #[test]
    fn scanning_the_real_workspace_is_deterministic() {
        let root = repo_root().expect("running inside the workspace");
        let a = scan(&root).expect("scan");
        let b = scan(&root).expect("scan");
        assert_eq!(a.fns, b.fns);
        assert_eq!(a.files, b.files);
        assert_eq!(a.unmeasured, b.unmeasured);
    }

    /// Guards the determinism rule that is easiest to break by accident later.
    #[test]
    fn this_crate_uses_no_hashmap() {
        let here = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for entry in WalkDir::new(&here).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() {
                continue;
            }
            let text = std::fs::read_to_string(entry.path()).unwrap();
            // Assembled at runtime rather than written as a literal: spelled out, the
            // needle appears in this file and the test flags its own source.
            let needles = [format!("Hash{}", "Map"), format!("Hash{}", "Set")];
            for (n, line) in text.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                assert!(
                    !needles.iter().any(|nd| code.contains(nd.as_str())),
                    "{}:{}: hash-map iteration order is not stable between runs, which \
                     would make the ledger churn for no reason. Use BTreeMap/BTreeSet.",
                    entry.path().display(),
                    n + 1
                );
            }
        }
    }
}
