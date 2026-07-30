//! Watch the PipeWire node's "somebody is watching" flag move in real time.
//!
//! The v4l2loopback side has [`ConsumerWatch`](cleanroom_video::ConsumerWatch) and a
//! STREAMON count. The PipeWire side has no counter at all, so the node infers it from its
//! own stream state: this node connects as a DRIVER, and WirePlumber only runs a DRIVER
//! source node while something is linked to it, which makes `Streaming` mean "at least one
//! consumer" and every other state mean "none". That inference is what power save now
//! consults, and this example is how you check it is telling the truth.
//!
//! Usage:
//!
//! ```text
//! cargo run -p cleanroom-video --example pw_activity
//! ```
//!
//! It runs for 30 seconds, publishing 640x480@30 black frames, and prints every change of
//! the flag. In a second terminal, link a consumer and watch it flip:
//!
//! ```text
//! gst-launch-1.0 pipewiresrc target-object=cleanroom_cam ! videoconvert ! autovideosink
//! ```
//!
//! Two things to know about that command. The node *name* is `cleanroom_cam` whether the
//! daemon or this example published it — only the description differs, and this one calls
//! itself "Cleanroom PW activity probe" — so either stop `cleanroomd` first or pass this
//! node's `object.serial` (from `pw-dump`) as `target-object` instead, which is unambiguous.
//! Otherwise `target-object=cleanroom_cam` may well pick the daemon's node and this example
//! will sit there reporting nobody.
//!
//! And `gst-launch-1.0` is not the only option — anything that links to the node will do.
//! Not `pw-record`, though: that is the audio tool and will not touch a `Video/Source`. The
//! GUI preview is the real-world consumer, and it links to this same node through
//! [`PwCapture`](cleanroom_video::PwCapture); a dozen lines against that API make a
//! consumer of your own when the GStreamer plugin is missing or broken.
//!
//! With no PipeWire at all — a session with no daemon running, a container — this prints a
//! note on stderr and exits 0. That is not a failure of the example.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Deliberately not the daemon's "Cleanroom Camera": if both are up, the description is
/// the only thing telling them apart in `pw-cli ls Node` or a portal picker.
const DESCRIPTION: &str = "Cleanroom PW activity probe";

/// How long to watch before tidying up.
const RUN_FOR: usize = 30;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let node = cleanroom_core::node::VIRTUAL_CAM_NODE;
    let slot = cleanroom_video::FrameSlot::new(640, 480, 30);
    let active = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));

    println!("publishing `{node}` as \"{DESCRIPTION}\" — 640x480@30, black frames");
    println!("link a consumer to see the flag move, e.g.");
    println!("  gst-launch-1.0 pipewiresrc target-object={node} ! videoconvert ! autovideosink");
    println!("  (stop cleanroomd first, or target this node's object.serial from `pw-dump`)");
    println!("watching for {RUN_FOR}s\n");

    let active_thread = active.clone();
    let stop_thread = stop.clone();
    let pw = std::thread::spawn(move || {
        cleanroom_video::PwSource::run(
            slot,
            DESCRIPTION,
            move || stop_thread.load(Ordering::Relaxed),
            active_thread,
        )
    });

    let mut last: Option<bool> = None;
    for _ in 0..RUN_FOR {
        // A thread that has already ended means the node never came up. Stop watching a
        // flag nobody is going to write.
        if pw.is_finished() {
            break;
        }
        let now = active.load(Ordering::Relaxed);
        if Some(now) != last {
            println!(
                "active: {}",
                if now {
                    "YES — at least one consumer is linked (power save keeps the camera awake)"
                } else {
                    "no — nobody is watching this node"
                }
            );
            last = Some(now);
        }
        std::thread::sleep(Duration::from_secs(1));
    }

    stop.store(true, Ordering::Relaxed);
    match pw.join() {
        Ok(Ok(())) => println!(
            "\nnode withdrawn; final flag = {}",
            active.load(Ordering::Relaxed)
        ),
        // Almost always "no PipeWire here". Reported, not raised: an example that cannot
        // reach a session bus has nothing to demonstrate, and that is not an error.
        Ok(Err(e)) => eprintln!("no PipeWire node was published: {e}"),
        Err(_) => eprintln!("the PipeWire thread ended abnormally"),
    }
    Ok(())
}
