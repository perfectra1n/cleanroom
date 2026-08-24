//! Measure what DeepFilterNet actually does to a known signal.
//!
//! Feeds speech-shaped tone plus broadband noise through the denoiser and reports the
//! change in noise floor and in signal level. A denoiser that flattens everything and one
//! that does nothing both "run fine" — this distinguishes them.
//!
//!     nix develop -c cargo run --release -p cleanroom-audio --example denoise_measure

use cleanroom_audio::{Denoiser, HOP, SnrThresholds, find_model};

const SR: f32 = 48_000.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = find_model()?;
    println!("model: {}\n", model.display());

    let mut d = Denoiser::new(&model, 40.0, 0.02, SnrThresholds::default())?;

    // Two seconds of signal. The first half is noise only, so we can measure the floor
    // the denoiser leaves behind; the second half adds a voice-band tone complex.
    let hops = (SR * 2.0 / HOP as f32) as usize;
    let mut seed = 0x1234_5678u32;
    let mut noise_in = Vec::new();
    let mut noise_out = Vec::new();
    let mut sig_in = Vec::new();
    let mut sig_out = Vec::new();

    let mut inp = [0.0f32; HOP];
    let mut outp = [0.0f32; HOP];

    for h in 0..hops {
        let speech_active = h > hops / 2;
        for (i, s) in inp.iter_mut().enumerate() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((seed >> 16) as f32 / 32768.0 - 1.0) * 0.05;

            let t = (h * HOP + i) as f32 / SR;
            // A crude voice: fundamental plus two harmonics in the speech band.
            let voice = if speech_active {
                0.25 * ((t * 2.0 * std::f32::consts::PI * 160.0).sin()
                    + 0.5 * (t * 2.0 * std::f32::consts::PI * 480.0).sin()
                    + 0.3 * (t * 2.0 * std::f32::consts::PI * 1200.0).sin())
            } else {
                0.0
            };
            *s = voice + noise;
        }

        d.process(&inp, &mut outp);

        // Skip the first few hops: the model is recurrent and needs to settle.
        if h < 20 {
            continue;
        }
        if speech_active {
            sig_in.extend_from_slice(&inp);
            sig_out.extend_from_slice(&outp);
        } else {
            noise_in.extend_from_slice(&inp);
            noise_out.extend_from_slice(&outp);
        }
    }

    let rms = |v: &[f32]| -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt()
    };
    let db = |x: f32| if x > 1e-9 { 20.0 * x.log10() } else { -120.0 };

    let nin = db(rms(&noise_in));
    let nout = db(rms(&noise_out));
    let sin_ = db(rms(&sig_in));
    let sout = db(rms(&sig_out));

    println!(
        "noise-only section:  {nin:6.1} dB -> {nout:6.1} dB   ({:+.1} dB)",
        nout - nin
    );
    println!(
        "speech section:      {sin_:6.1} dB -> {sout:6.1} dB   ({:+.1} dB)",
        sout - sin_
    );
    println!("\nlast local SNR estimate: {:.1} dB", d.last_lsnr);

    let noise_reduction = nin - nout;
    let speech_loss = sin_ - sout;
    println!(
        "\nnoise suppressed by {noise_reduction:.1} dB, speech attenuated by {speech_loss:.1} dB"
    );

    // A denoiser that suppresses noise and speech equally is just a volume control.
    if noise_reduction > speech_loss + 6.0 {
        println!("VERDICT: suppressing noise selectively — working as intended");
    } else if noise_reduction < 1.0 {
        println!("VERDICT: no measurable suppression");
    } else {
        println!("VERDICT: attenuating everything roughly equally — that is a volume control");
    }
    Ok(())
}
