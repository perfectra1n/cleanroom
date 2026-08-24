//! Measure how the denoiser's gain behaves on a real capture, per config.
//!
//! This is the calibration tool behind the SNR-threshold defaults (see
//! [`cleanroom_audio::SnrThresholds`]): it replays a raw capture through the denoiser
//! and reports, per 0.25 s bucket, how much gain was applied to speech versus noise.
//! A healthy config shows speech near 0 dB with a small spread; a pumping config shows
//! speech-bucket gains swinging by 10 dB or more.
//!
//! ## Workflow
//!
//! Record the raw microphone (and optionally the processed virtual mic, for A/B):
//!
//!     pw-record --target <hw-node> --rate 48000 --channels 1 --format f32 raw.wav
//!     sox raw.wav -r 48000 -c 1 -e float -b 32 -t raw capture.raw
//!
//! Replay through a candidate config, optionally rendering the enhanced audio so the
//! configs can be compared by ear (`sox -r 48000 -c 1 -e float -b 32 -t raw out.raw
//! out.wav`, then `pw-play out.wav`):
//!
//!     cargo run --release -p cleanroom-audio --example envelope_probe -- \
//!         capture.raw [atten_db] [pf_beta] [gate_db] [passthrough_db] [df_db] [out.raw]
//!
//! Omitted arguments fall back to the production defaults.

use cleanroom_audio::{Denoiser, HOP, SnrThresholds, find_model};

const SR: f32 = 48_000.0;
const BUCKET_SECS: f32 = 0.25;
/// Bucket RMS above this is counted as speech, below (but not near-silence) as noise.
const SPEECH_FLOOR_DB: f64 = -35.0;
const NOISE_FLOOR_DB: f64 = -70.0;

struct Args {
    path: String,
    atten_db: f32,
    pf_beta: f32,
    thresholds: SnrThresholds,
    render_to: Option<String>,
}

const USAGE: &str =
    "usage: envelope_probe <capture.raw> [atten] [pf] [gate] [passthrough] [df] [out.raw]";

fn parse_args() -> Args {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect(USAGE);
    let d = SnrThresholds::default();
    // A numeric slot that fails to parse is an error, never a silent fallback: this
    // tool's numbers set production defaults, and its worst failure mode is measuring
    // a config the user did not ask for. (An output path given before all five numbers
    // would otherwise be eaten as a threshold and the render silently skipped.)
    let mut f = |name: &str, fallback: f32| -> f32 {
        let Some(s) = a.next() else { return fallback };
        s.parse().unwrap_or_else(|_| {
            eprintln!("argument '{name}': expected a number, got {s:?}\n{USAGE}");
            std::process::exit(2);
        })
    };
    Args {
        path,
        atten_db: f(
            "atten",
            cleanroom_core::config::DenoiseConfig::default().attenuation_db,
        ),
        pf_beta: f(
            "pf",
            cleanroom_core::config::DenoiseConfig::default().post_filter_beta,
        ),
        thresholds: SnrThresholds {
            gate_db: f("gate", d.gate_db),
            passthrough_db: f("passthrough", d.passthrough_db),
            df_db: f("df", d.df_db),
        },
        render_to: a.next(),
    }
}

fn db(x: f64) -> f64 {
    if x > 1e-12 { 20.0 * x.log10() } else { -120.0 }
}

fn stats(v: &[f64]) -> String {
    if v.is_empty() {
        return "n=0".into();
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|g| (g - m) * (g - m)).sum::<f64>() / v.len() as f64).sqrt();
    let min = v.iter().cloned().fold(f64::MAX, f64::min);
    let max = v.iter().cloned().fold(f64::MIN, f64::max);
    format!(
        "n={:<3} mean {m:6.1}  sd {sd:4.1}  min {min:6.1}  max {max:5.1}",
        v.len()
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    let bytes = std::fs::read(&args.path)?;
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let model = find_model()?;
    let mut d = Denoiser::new(&model, args.atten_db, args.pf_beta, args.thresholds)?;

    let hops_per_bucket = (SR * BUCKET_SECS / HOP as f32) as usize;
    let mut inp = [0.0f32; HOP];
    let mut outp = [0.0f32; HOP];
    let (mut in_acc, mut out_acc, mut n_acc) = (0.0f64, 0.0f64, 0usize);
    let mut speech = Vec::new();
    let mut noise = Vec::new();
    let mut rendered: Vec<u8> = Vec::new();

    for (h, hop) in samples.chunks_exact(HOP).enumerate() {
        inp.copy_from_slice(hop);
        d.process(&inp, &mut outp);
        if args.render_to.is_some() {
            for s in &outp {
                rendered.extend_from_slice(&s.to_le_bytes());
            }
        }

        in_acc += inp.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>();
        out_acc += outp.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>();
        n_acc += HOP;
        if (h + 1) % hops_per_bucket != 0 {
            continue;
        }

        let irms = db((in_acc / n_acc as f64).sqrt());
        let orms = db((out_acc / n_acc as f64).sqrt());
        if irms > SPEECH_FLOOR_DB {
            speech.push(orms - irms);
        } else if irms > NOISE_FLOOR_DB {
            noise.push(orms - irms);
        }
        (in_acc, out_acc, n_acc) = (0.0, 0.0, 0);
    }

    let t = args.thresholds;
    println!(
        "atten {:4.1}  pf {:4.2}  thr({:5.1},{:4.1},{:4.1})\n  speech: {}\n  noise:  {}",
        args.atten_db,
        args.pf_beta,
        t.gate_db,
        t.passthrough_db,
        t.df_db,
        stats(&speech),
        stats(&noise),
    );
    if let Some(p) = args.render_to {
        std::fs::write(&p, &rendered)?;
        println!("rendered enhanced audio to {p}");
    }
    Ok(())
}
