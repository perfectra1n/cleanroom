//! Naming the processes that hold a device node open.
//!
//! This is the `/proc/*/fd` scan that [`crate::consumers`] rejects as a *counter*, kept
//! deliberately for the one job it is still good at: putting a human-readable name next
//! to a consumer that we already know exists.
//!
//! The reason it cannot count is worth repeating here, because the failure is invisible.
//! The `/proc/PID/fd` magic links of a process outside our user namespace fail the
//! kernel's ptrace-mode check, so a Flatpak'd or bubblewrap'd browser reading the camera
//! is indistinguishable from nobody reading it — and the scan reports a confident empty
//! list rather than an error. Believing that emptiness is how a camera goes dark in the
//! middle of a call.
//!
//! So the split is: [`ConsumerWatch`](crate::ConsumerWatch) is the authoritative counter
//! and the only thing a power-save decision may consult, and this module is a *namer*
//! whose output is only ever shown to a person. "2 consumers" is the truth; "chrome
//! (41234)" is a helpful annotation on it, and an empty holder list next to a non-zero
//! count means "sandboxed", not "wrong".
//!
//! The scan itself is read-only and swallows every error. A process that exits between
//! `read_dir("/proc")` and `read_link` of its fds is completely normal, not a fault.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Stand-in for a `comm` we could not read — the process is gone, or is not ours.
const UNKNOWN_COMM: &str = "?";

/// A process holding an open fd to a device node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    pub pid: u32,
    pub comm: String,
}

impl std::fmt::Display for Holder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.comm, self.pid)
    }
}

/// Best-effort: name every process with `path` open.
///
/// `exclude_pid` drops one process from the result, which is how the daemon keeps its own
/// producer fd on the loopback device out of a list meant to describe *consumers*.
///
/// An empty list is not evidence of an idle device — see the module docs. Never branch a
/// power-save decision on it.
pub fn holders_of(path: &Path, exclude_pid: Option<u32>) -> Vec<Holder> {
    // The kernel hands back fully resolved paths in the fd links, so the target has to be
    // resolved too or a `/tmp` that is really `/private/tmp` never matches. A path nobody
    // can canonicalize (typically: it does not exist) is compared as written, which
    // simply matches nothing.
    let target = match path.canonicalize() {
        Ok(resolved) => resolved,
        Err(_) => path.to_path_buf(),
    };

    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };

    entries
        .flatten()
        .filter_map(|entry| numeric_pid(&entry.file_name()))
        .filter(|pid| Some(*pid) != exclude_pid)
        .filter(|pid| pid_holds(&fd_dir(*pid), &target))
        .map(|pid| Holder {
            pid,
            comm: comm_of(pid),
        })
        .collect()
}

/// `/proc` also holds `self`, `net`, `sys` and friends; only the all-digit names are pids.
fn numeric_pid(name: &OsStr) -> Option<u32> {
    let name = name.to_str()?;
    if !name.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    name.parse().ok()
}

fn fd_dir(pid: u32) -> PathBuf {
    PathBuf::from(format!("/proc/{pid}/fd"))
}

/// Whether any fd of one process resolves to `target`.
///
/// Split out of [`holders_of`] so neither function has to nest a directory walk inside a
/// directory walk: an unreadable `fd_dir` — the sandboxed case, and also every process
/// belonging to another user — is just `false` here.
fn pid_holds(fd_dir: &Path, target: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(fd_dir) else {
        return false;
    };
    entries
        .flatten()
        .filter_map(|entry| std::fs::read_link(entry.path()).ok())
        .any(|link| link == target)
}

/// The process name from `/proc/PID/comm`, or `"?"` if it cannot be read.
///
/// `comm` carries a trailing newline and is truncated by the kernel to 15 bytes, so this
/// is a short name like `ffplay`, not a command line. That is what we want: a command
/// line can be arbitrarily long and can contain anything a caller typed.
fn comm_of(pid: u32) -> String {
    let Ok(raw) = std::fs::read_to_string(format!("/proc/{pid}/comm")) else {
        return UNKNOWN_COMM.to_string();
    };
    let name = raw.trim();
    if name.is_empty() {
        return UNKNOWN_COMM.to_string();
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path in the temp dir that no other test (or concurrent run) will collide with.
    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("cleanroom-holders-{}-{tag}", std::process::id()))
    }

    #[test]
    fn a_process_with_the_file_open_is_named_with_comm_and_pid() {
        let path = temp_path("held");
        let created = std::fs::File::create(&path);
        assert!(created.is_ok(), "could not create {}", path.display());
        let Ok(file) = created else { return };

        // Scan while the fd is still open, then tidy up before asserting so a failure
        // does not leave the fixture behind.
        let found = holders_of(&path, None);
        drop(file);
        let _ = std::fs::remove_file(&path);

        let me = std::process::id();
        let mine = found.iter().find(|h| h.pid == me);
        assert!(
            mine.is_some(),
            "our own pid should hold {}, got {found:?}",
            path.display()
        );
        let Some(holder) = mine else { return };
        // The test binary's own `comm`, whatever cargo called it.
        assert_eq!(holder.comm, comm_of(me));
        assert_ne!(holder.comm, UNKNOWN_COMM, "our own comm is always readable");
    }

    #[test]
    fn excluding_our_own_pid_hides_the_daemons_producer_fd() {
        let path = temp_path("excluded");
        let created = std::fs::File::create(&path);
        assert!(created.is_ok(), "could not create {}", path.display());
        let Ok(file) = created else { return };

        let found = holders_of(&path, Some(std::process::id()));
        drop(file);
        let _ = std::fs::remove_file(&path);

        // Nothing else in the system has this file open, so excluding ourselves is the
        // difference between one holder and none — which is exactly the shape the daemon
        // needs when it wants "consumers other than me".
        assert!(found.is_empty(), "expected no holders, got {found:?}");
    }

    #[test]
    fn a_path_nobody_holds_reports_no_holders() {
        let path = temp_path("absent");
        let _ = std::fs::remove_file(&path);
        let found = holders_of(&path, None);
        assert!(
            found.is_empty(),
            "{} does not exist, so nothing can hold it; got {found:?}",
            path.display()
        );
    }

    #[test]
    fn holders_display_as_comm_then_pid() {
        let h = Holder {
            pid: 41234,
            comm: "chrome".to_string(),
        };
        assert_eq!(h.to_string(), "chrome (41234)");
    }
}
