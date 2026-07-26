//! Watch the virtual camera's consumer count change in real time.
//!
//! Run the passthrough example in one terminal, this in another, then start and stop
//! `ffplay /dev/video10` and watch the count move.

use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(dev) = cleanroom_video::enumerate()
        .into_iter()
        .find(|d| d.is_virtual && d.accessible)
    else {
        eprintln!("no loopback device found");
        return Ok(());
    };

    println!("watching {} ({})", dev.path.display(), dev.card);
    println!(
        "start/stop `ffplay {}` to see it move\n",
        dev.path.display()
    );

    let mut watch = cleanroom_video::ConsumerWatch::open(&dev.path)?;
    let mut last = None;

    for _ in 0..60 {
        let now = watch.poll(Duration::from_millis(500));
        if now != last {
            println!(
                "consumers: {}   -> {}",
                now.map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown".into()),
                if watch.in_use() {
                    "IN USE (capture runs)"
                } else {
                    "idle (capture can stop, LED off)"
                }
            );
            last = now;
        }
    }
    Ok(())
}
