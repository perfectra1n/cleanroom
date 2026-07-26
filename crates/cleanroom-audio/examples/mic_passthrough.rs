//! Publish a virtual microphone that passes the hardware mic through untouched.
//!
//! Proves the PipeWire plumbing — node class, direction, format negotiation and the
//! quantum-to-hop bridge — before any denoiser is involved.
//!
//!     nix develop -c cargo run -p cleanroom-audio --example mic_passthrough [node.name]
//!     wpctl status | grep -i cleanroom

use cleanroom_audio::{SharedAudio, VirtualMic, to_dbfs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    // Optional argument: the node.name of the hardware mic to pin to. Without it we bind
    // the system default, which the library warns about — that is the self-capture trap.
    let target = match std::env::args().nth(1) {
        Some(s) => Some(cleanroom_core::CaptureTarget::new(s)?),
        None => None,
    };

    let shared = SharedAudio::new();
    let stop = Arc::new(AtomicBool::new(false));

    // Report levels from a side thread so we can see audio actually flowing, and see
    // overruns if the bridge ever falls behind.
    {
        let shared = shared.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let i = *shared.level_in.lock().unwrap();
                let o = *shared.level_out.lock().unwrap();
                let over = shared.bridge.lock().unwrap().overruns;
                println!(
                    "in {:6.1} dBFS   out {:6.1} dBFS   overruns {over}",
                    to_dbfs(i),
                    to_dbfs(o)
                );
            }
        });
    }

    let stop_check = stop.clone();
    VirtualMic::run(
        shared,
        target,
        cleanroom_core::VIRTUAL_MIC_NODE,
        "Cleanroom Microphone",
        |inp, outp| outp.copy_from_slice(inp),
        move || stop_check.load(Ordering::Relaxed),
    )?;
    Ok(())
}
