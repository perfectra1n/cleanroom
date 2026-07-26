//! Does dropping an ORT session that owns a Dawn context still segfault?
//!
//! The daemon leaks its session on purpose (see `impl Drop for Matter`), which is a real
//! cost: it forecloses any controlled shutdown, and it is the reason `Restart=on-failure`
//! is safe only by accident. The leak should go away the moment upstream lets it.
//!
//! This is the harness that answers the question, so it can be re-run against a new driver,
//! a new `ort`, or a new Dawn without editing library code:
//!
//! ```sh
//! nix develop -c cargo run --release -p cleanroom-matting --example teardown_check
//! ```
//!
//! Exit 0 means a real drop completed and the leak in `Drop` can be removed. A SIGSEGV
//! (shell reports 139) means it is still needed. Anything else is a different problem.
//!
//! Inference runs first on purpose: the fault appears *after* successful work, not on a
//! session that was never used, so tearing down an idle session would prove nothing.

use cleanroom_matting::{INFER_H, INFER_W, Matter};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let model = match cleanroom_matting::find_model() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };

    let px = (INFER_W * INFER_H) as usize;
    let frame = vec![128u8; px * 4];

    println!("creating session");
    let mut matter = Matter::new(&model).expect("session must build");

    println!("running inference (the fault only shows up after real work)");
    for _ in 0..4 {
        matter.infer(&frame).expect("inference must succeed");
    }

    // SAFETY-adjacent: this is the whole point of the example. If it segfaults, it segfaults.
    unsafe { std::env::set_var("CLEANROOM_DROP_ORT_SESSION", "1") };

    println!("dropping the session for real...");
    drop(matter);

    println!("SURVIVED — a controlled teardown works; the leak in Drop can be removed.");
}
