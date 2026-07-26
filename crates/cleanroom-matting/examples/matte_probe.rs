//! Inspect the alpha matte the network actually produces for a still image.
//!
//! This exists because "the background is not blurred" and "the whole frame is blurred"
//! look identical in a small preview, and they have opposite causes. The composite can
//! only be as good as the alpha it is handed, so when the output looks wrong the first
//! question is whether the *matte* is wrong — and answering it inside the daemon means
//! reading a number through a GPU readback, a temporal filter and a guided upsample.
//!
//! Here there is nothing in the way: a PNG goes in, the same `Matter` the daemon uses runs
//! on it, and the raw alpha comes out.
//!
//! ```text
//! cargo run -p cleanroom-matting --example matte_probe -- frame.png
//! ```
//!
//! A healthy portrait gives a bimodal matte: a large mass near 0 (the room), a large mass
//! near 1 (the person), and little in between. A matte that is entirely near 0 means the
//! network found no subject, which composites as "everything is background" — every pixel
//! blurred, the person included.

use cleanroom_matting::{INFER_H, INFER_W, Matter};

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: matte_probe <image>");
            std::process::exit(2);
        }
    };

    // The model is a *video* matting network with recurrent state, so a single pass on a
    // cold state is not what the daemon ever sees. Running the same still repeatedly is the
    // closest stand-in for a static scene and lets the state settle the way it would after
    // a second of real video.
    let passes: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let img = image::open(&path).expect("opening the image");
    // Straight scale to the inference size, matching what the GPU downscale pass does.
    // Both are 16:9, so nothing is distorted.
    let img = img
        .resize_exact(INFER_W, INFER_H, image::imageops::FilterType::Triangle)
        .to_rgba8();

    let model = cleanroom_matting::find_model().expect("locating the RVM weights");
    println!("model  {}", model.display());
    println!("input  {path} -> {INFER_W}x{INFER_H}, {passes} passes\n");

    // The provider under test is chosen explicitly: this probe exists to compare them.
    let backend = std::env::var("CLEANROOM_PROBE_BACKEND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cleanroom_matting::Backend::Cpu);
    println!("engine {backend}");
    let mut matter =
        Matter::new(&model, backend, INFER_W, INFER_H).expect("creating the matting session");

    let mut alpha = Vec::new();
    for pass in 0..passes {
        let out = matter.infer(img.as_raw()).expect("inference");
        alpha = out.to_vec();
        if pass + 1 == passes {
            break;
        }
    }

    report(&alpha);
    println!("\nrejected as degenerate: {}", matter.rejected);
}

fn report(alpha: &[u8]) {
    let n = alpha.len() as f64;
    let mean = alpha.iter().map(|&v| v as f64).sum::<f64>() / n / 255.0;
    let lo = alpha.iter().copied().min().unwrap_or(0);
    let hi = alpha.iter().copied().max().unwrap_or(0);

    // The shape of the distribution is the diagnosis, not the mean. A mean of 0.1 is a
    // small subject if the matte is bimodal and a dead network if it is flat.
    let mut hist = [0u64; 10];
    for &v in alpha {
        hist[(v as usize * 10 / 256).min(9)] += 1;
    }

    println!("alpha  min {lo}  max {hi}  mean {mean:.3}");
    println!("\ndistribution (fraction of pixels per decile of alpha)");
    for (i, &count) in hist.iter().enumerate() {
        let frac = count as f64 / n;
        let bar = "#".repeat((frac * 60.0).round() as usize);
        println!(
            "  {:.1}-{:.1}  {frac:6.3}  {bar}",
            i as f64 / 10.0,
            (i + 1) as f64 / 10.0
        );
    }

    println!("\nspatial map (' '=background, '.'=fringe, '#'=foreground)");
    let (cols, rows) = (64usize, 18usize);
    for r in 0..rows {
        let mut line = String::with_capacity(cols);
        for c in 0..cols {
            let x = c * INFER_W as usize / cols;
            let y = r * INFER_H as usize / rows;
            let v = alpha[y * INFER_W as usize + x];
            line.push(match v {
                0..=63 => ' ',
                64..=191 => '.',
                _ => '#',
            });
        }
        println!("  |{line}|");
    }
}
