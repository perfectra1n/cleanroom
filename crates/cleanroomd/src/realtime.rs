//! Requesting real-time scheduling for the audio thread, and reporting what was granted.
//!
//! doctor already claimed "the audio thread will request RT priority through it" — but
//! nothing in the tree ever did. There was no `sched_setscheduler`, no rtkit call and no
//! `RLIMIT_RTTIME` anywhere. Checking that rtkit *exists* while never asking it for
//! anything is precisely the kind of reassuring non-fact this project is supposed to avoid,
//! so the request comes first and the check reports the real answer afterwards.
//!
//! ## Order matters
//!
//! `RLIMIT_RTTIME` is set **before** asking for anything. Two reasons, and both bite:
//!
//! * rtkit refuses a thread that has no `RLIMIT_RTTIME` — it is how rtkit ensures a
//!   runaway real-time thread cannot lock the machine, and the refusal reads as a generic
//!   permission error.
//! * without it the kernel has no backstop either, so a spinning RT thread at a priority
//!   above the display server takes the session with it.
//!
//! ## Why rtkit rather than just `limits.conf`
//!
//! PAM's `limits.conf` does not apply to systemd user units, so `RLIMIT_RTPRIO` is
//! normally 0 there and the direct `sched_setscheduler` call fails with `EPERM`. That is
//! expected rather than a misconfiguration, and rtkit is the supported route.

use std::sync::Arc;

/// The priority we ask for.
///
/// Deliberately low. The audio thread has 10 ms of ring buffer to play with and only needs
/// to beat ordinary CPU-bound work; asking for something near the top would put us above
/// the compositor, where being wrong is a frozen desktop rather than a glitch.
const RT_PRIORITY: i32 = 5;

/// The RT time budget, in microseconds.
///
/// A hop is 10 ms of audio and the work is one DeepFilterNet inference, so 200 ms is two
/// orders of magnitude of headroom. If a thread ever burns that much CPU without blocking,
/// it is wedged and the kernel should stop it.
const RT_TIME_LIMIT_US: u64 = 200_000;

/// What happened when real-time scheduling was requested.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RtStatus {
    /// Not attempted yet.
    #[default]
    Unknown,
    /// Granted, with the policy actually in force afterwards.
    Granted { via: &'static str, policy: i32 },
    /// Refused. `why` is what to tell the user.
    Denied { why: String },
}

impl std::fmt::Display for RtStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RtStatus::Unknown => write!(f, "not requested yet"),
            RtStatus::Granted { via, policy } => {
                write!(f, "granted via {via} (SCHED policy {policy})")
            }
            RtStatus::Denied { why } => write!(f, "not granted: {why}"),
        }
    }
}

/// Ask for real-time scheduling on the calling thread, then verify what was granted.
///
/// The verification is the point. `sched_setscheduler` returning success is not the same
/// as running with a real-time policy — rtkit in particular grants asynchronously, from
/// another process — so the answer comes from reading the policy back rather than from the
/// return code of the request.
pub fn request_for_current_thread() -> RtStatus {
    if let Err(e) = set_rttime_limit() {
        // Not fatal on its own, but it will almost certainly make rtkit refuse, so say so
        // now rather than reporting a confusing permission error later.
        tracing::warn!(error = %e, "could not set RLIMIT_RTTIME; rtkit will likely refuse");
    }

    // Try directly first. On a system where limits.conf does apply — a plain login shell,
    // a container with the capability — this succeeds and rtkit never has to be involved.
    match set_scheduler_directly() {
        Ok(()) => {
            let policy = current_policy();
            if is_realtime(policy) {
                return RtStatus::Granted {
                    via: "sched_setscheduler",
                    policy,
                };
            }
            // Succeeded but did not take. Worth reporting rather than trusting the call.
            return RtStatus::Denied {
                why: format!("sched_setscheduler reported success but the policy is {policy}"),
            };
        }
        Err(e) if e == libc::EPERM => {
            // Expected under a systemd user unit. Fall through to rtkit.
            tracing::debug!("sched_setscheduler denied (expected under a user unit); asking rtkit");
        }
        Err(e) => {
            return RtStatus::Denied {
                why: format!(
                    "sched_setscheduler failed: {}",
                    std::io::Error::from_raw_os_error(e)
                ),
            };
        }
    }

    match rtkit_request() {
        Ok(()) => {
            let policy = current_policy();
            if is_realtime(policy) {
                RtStatus::Granted {
                    via: "rtkit",
                    policy,
                }
            } else {
                RtStatus::Denied {
                    why: format!(
                        "rtkit accepted the request but the thread is still on policy {policy}"
                    ),
                }
            }
        }
        Err(why) => RtStatus::Denied { why },
    }
}

/// Return the calling thread to ordinary scheduling.
///
/// Called at the top of every pipeline (re)start, and load-bearing: the thread keeps its
/// SCHED_RR policy from the previous run, and the DeepFilterNet model load that follows
/// is a long continuous CPU burst. Under the `RLIMIT_RTTIME` budget rtkit made us set,
/// that burst is indistinguishable from a wedged real-time thread, and the kernel's
/// answer to one of those is SIGKILL to the whole process — no signal handler, no log
/// line, exit code 137. Observed on every microphone switch in a debug build, where the
/// unoptimised model load comfortably exceeds the 200 ms budget.
///
/// First start is a no-op (the thread is not real-time yet), which is exactly the
/// symmetry wanted: every pass through `run_once` does its heavy lifting at normal
/// priority and is promoted afterwards.
pub fn demote_current_thread() {
    // The SCHED_RESET_ON_FORK flag rtkit set must be passed back, not dropped: the
    // kernel reads a policy without it as a request to *clear* the flag, which needs
    // CAP_SYS_NICE — so a plain SCHED_OTHER demotion fails with EPERM precisely on the
    // rtkit-promoted thread it exists for. Observed: the warning below fired and the
    // process was still RTTIME-killed a moment later.
    let keep_flag = current_policy() & SCHED_RESET_ON_FORK;
    let param = libc::sched_param { sched_priority: 0 };
    // SAFETY: pid 0 means the calling thread; `param` is valid for the call. Dropping to
    // SCHED_OTHER requires no privilege, unlike acquiring a real-time policy.
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER | keep_flag, &param) };
    if rc != 0 {
        // Worst case the model load runs against the RT budget, which is where we were.
        tracing::warn!(
            error = %std::io::Error::last_os_error(),
            "could not return the audio thread to normal scheduling before reload"
        );
    }
}

fn set_rttime_limit() -> std::io::Result<()> {
    let lim = libc::rlimit {
        rlim_cur: RT_TIME_LIMIT_US,
        rlim_max: RT_TIME_LIMIT_US,
    };
    // SAFETY: `lim` is a valid, fully-initialised rlimit for the duration of the call.
    let rc = unsafe { libc::setrlimit(libc::RLIMIT_RTTIME, &lim) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Returns the raw errno on failure, since `EPERM` is a normal outcome we branch on.
fn set_scheduler_directly() -> Result<(), i32> {
    let param = libc::sched_param {
        sched_priority: RT_PRIORITY,
    };
    // SAFETY: pid 0 means the calling thread; `param` is valid for the call.
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_RR, &param) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(libc::EINVAL))
    }
}

/// The scheduling policy currently in force for this thread.
pub fn current_policy() -> i32 {
    // SAFETY: pid 0 means the calling thread; no pointers involved.
    unsafe { libc::sched_getscheduler(0) }
}

/// `SCHED_RESET_ON_FORK`, which `sched_getscheduler` returns OR-ed into the policy.
///
/// Not exposed by the libc crate, so it is spelled out here. It is set by rtkit on every
/// thread it promotes — deliberately, so a real-time thread cannot fork a real-time child
/// and escape the budget.
const SCHED_RESET_ON_FORK: i32 = 0x4000_0000;

/// Whether a policy returned by `sched_getscheduler` is a real-time one.
///
/// The mask is load-bearing. rtkit grants `SCHED_RR | SCHED_RESET_ON_FORK`, so the raw
/// value comes back as 1073741826 rather than 2, and comparing for equality reports "not
/// granted" on a thread that was granted — which is worse than not checking at all, since
/// it sends someone off to debug rtkit permissions that are working correctly. Observed
/// exactly that on the reference machine before this mask existed.
pub fn is_realtime(policy: i32) -> bool {
    let base = policy & !SCHED_RESET_ON_FORK;
    base == libc::SCHED_RR || base == libc::SCHED_FIFO
}

/// Ask rtkit to promote this thread.
///
/// Synchronous D-Bus on the *system* bus, from the audio thread, before the loop starts —
/// so a slow or absent rtkit costs startup latency, never a glitch mid-stream.
fn rtkit_request() -> Result<(), String> {
    // A blocking call from a thread that has no async runtime, so this builds its own
    // connection rather than borrowing the daemon's.
    let tid = gettid();

    let result: Result<(), Box<dyn std::error::Error>> = (|| {
        let conn = zbus::blocking::Connection::system()?;
        let proxy = zbus::blocking::Proxy::new(
            &conn,
            "org.freedesktop.RealtimeKit1",
            "/org/freedesktop/RealtimeKit1",
            "org.freedesktop.RealtimeKit1",
        )?;
        // MakeThreadRealtime takes a *thread* id, not a pid, and a u32 priority.
        proxy.call::<_, _, ()>("MakeThreadRealtime", &(tid, RT_PRIORITY as u32))?;
        Ok(())
    })();

    result.map_err(|e| {
        format!(
            "rtkit refused or is unavailable ({e}). PAM's limits.conf does not apply to \
             systemd user units, so RLIMIT_RTPRIO of 0 is normal there and is not itself \
             the fault"
        )
    })
}

fn gettid() -> u64 {
    // SAFETY: gettid takes no arguments and cannot fail.
    unsafe { libc::syscall(libc::SYS_gettid) as u64 }
}

/// Publish the outcome so doctor and the GUI can report it rather than guess.
pub fn publish(shared: &Arc<crate::state::Shared>, status: RtStatus) {
    tracing::info!(status = %status, "real-time scheduling");
    shared.set_rt_status(status);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_denial_says_why_and_names_the_usual_cause() {
        let s = RtStatus::Denied {
            why: "rtkit refused or is unavailable (x). PAM's limits.conf does not apply".into(),
        };
        assert!(s.to_string().contains("limits.conf"));
    }

    /// SCHED_OTHER is 0 and is what an unprivileged thread runs under. Reading it as
    /// real-time would make doctor report success on every machine.
    #[test]
    fn only_rr_and_fifo_count_as_realtime() {
        assert!(is_realtime(libc::SCHED_RR));
        assert!(is_realtime(libc::SCHED_FIFO));
        assert!(!is_realtime(libc::SCHED_OTHER));
        assert!(!is_realtime(libc::SCHED_IDLE));
    }

    /// rtkit sets SCHED_RESET_ON_FORK on everything it promotes, so sched_getscheduler
    /// returns 1073741826 rather than 2. Comparing for equality reported "not granted" on a
    /// thread that had just been granted — observed on the reference machine — which is
    /// worse than no check, because it sends someone to debug working rtkit permissions.
    #[test]
    fn the_reset_on_fork_flag_does_not_hide_a_real_time_policy() {
        assert!(
            is_realtime(libc::SCHED_RR | SCHED_RESET_ON_FORK),
            "SCHED_RR|SCHED_RESET_ON_FORK (what rtkit actually grants) must count"
        );
        assert!(is_realtime(libc::SCHED_FIFO | SCHED_RESET_ON_FORK));
        assert_eq!(
            libc::SCHED_RR | SCHED_RESET_ON_FORK,
            1073741826,
            "the value observed from sched_getscheduler after rtkit granted the request"
        );
        // The flag must not turn a non-real-time policy into a real-time one either.
        assert!(!is_realtime(libc::SCHED_OTHER | SCHED_RESET_ON_FORK));
    }

    /// The check must reflect reality: this test thread has not asked for anything, so it
    /// must not be reported as real-time.
    #[test]
    fn an_ordinary_thread_reports_as_not_realtime() {
        assert!(!is_realtime(current_policy()));
    }
}
