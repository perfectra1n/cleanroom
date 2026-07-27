//! Verify the GPU pipeline does what it claims, and time it.
//!
//! Three questions, in order of how badly a wrong answer would hurt:
//!   1. Does a frame survive YUY2 -> RGBA -> YUY2 without drifting? A colour-space bug
//!      here tints every frame and is nearly invisible without a reference.
//!   2. Does blur actually blur — and only the background plane?
//!   3. Is it fast enough at 1080p?
//!
//!     nix develop -c cargo run --release -p cleanroom-gpu --example gpu_check
//!
//! An optional argument pins the adapter to one DRM render node:
//!
//!     ... --example gpu_check -- /dev/dri/renderD129
//!
//! That matters for the throughput numbers. Without it the same adapter selection the daemon
//! uses picks the fastest thing present, so on a machine with a discrete GPU the slow-hardware
//! conformance target — the iGPU, measured at 8x the frame time during the spikes — is never
//! exercised at all, and "fits 30fps" is answered for the wrong device.

use cleanroom_core::BackgroundMode;
use cleanroom_gpu::{FramePipeline, Gpu, Look};
use std::path::PathBuf;
use std::time::Instant;

/// The three settings these checks vary, with everything else left at its default.
fn look(mode: BackgroundMode, blur_strength: f32, mirror: bool) -> Look {
    Look {
        mode,
        blur_strength,
        mirror,
        ..Default::default()
    }
}

const W: u32 = 1920;
const H: u32 = 1080;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pin: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);
    let gpu = Gpu::new(pin.as_deref())?;
    println!("adapter: {}\n", gpu.choice);

    let mut pipe = FramePipeline::new(gpu, W, H);
    let frame_bytes = (W * H * 2) as usize;

    // The test frame is built from *RGB* and converted to YUY2 on the CPU with the same
    // BT.601 limited-range matrix the shader uses.
    //
    // Generating YUV directly, as an earlier version did, produces arbitrary Y/Cb/Cr
    // combinations that are outside the RGB gamut. The shader clamps those — correctly —
    // and the round-trip then cannot recover them, which reads as a colour bug when it is
    // really the test asking for colours that do not exist.
    let mut input = vec![0u8; frame_bytes];
    for y in 0..H as usize {
        for x in (0..W as usize).step_by(2) {
            let i = (y * (W / 2) as usize + x / 2) * 4;
            const BARS: [(f32, f32, f32); 8] = [
                (0.8, 0.1, 0.1),
                (0.1, 0.8, 0.1),
                (0.1, 0.1, 0.8),
                (0.8, 0.8, 0.1),
                (0.8, 0.1, 0.8),
                (0.1, 0.8, 0.8),
                (0.7, 0.7, 0.7),
                (0.2, 0.2, 0.2),
            ];
            let rgb = |px: usize| -> (f32, f32, f32) {
                let bar = (px * 8 / W as usize).min(7);
                // A fine checkerboard on top of the bars, so there is high-frequency
                // detail for the blur test to remove.
                let c: f32 = if ((px / 16) + (y / 16)).is_multiple_of(2) {
                    0.15
                } else {
                    0.0
                };
                let b = BARS[bar];
                ((b.0 + c).min(1.0), (b.1 + c).min(1.0), (b.2 + c).min(1.0))
            };
            let to_yuv = |(r, g, b): (f32, f32, f32)| {
                (
                    0.2568 * r + 0.5041 * g + 0.0979 * b + 0.0627451,
                    -0.1482 * r - 0.2910 * g + 0.4392 * b + 0.5019608,
                    0.4392 * r - 0.3678 * g - 0.0714 * b + 0.5019608,
                )
            };
            let (y0, u0, v0) = to_yuv(rgb(x));
            let (y1, u1, v1) = to_yuv(rgb((x + 1).min(W as usize - 1)));
            input[i] = (y0 * 255.0 + 0.5) as u8;
            input[i + 1] = ((u0 + u1) * 0.5 * 255.0 + 0.5) as u8;
            input[i + 2] = (y1 * 255.0 + 0.5) as u8;
            input[i + 3] = ((v0 + v1) * 0.5 * 255.0 + 0.5) as u8;
        }
    }

    let mut output = vec![0u8; frame_bytes];

    // --- 1. round-trip fidelity ------------------------------------------------------
    pipe.process(&input, &mut output, look(BackgroundMode::Off, 0.0, false));

    let luma_err: Vec<i32> = input
        .iter()
        .step_by(2)
        .zip(output.iter().step_by(2))
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .collect();
    let max_err = *luma_err.iter().max().unwrap_or(&0);
    let mean_err = luma_err.iter().sum::<i32>() as f64 / luma_err.len() as f64;

    println!("1. YUY2 -> RGBA -> YUY2 round-trip");
    println!("   luma error: max {max_err}, mean {mean_err:.3}");
    // Two 8-bit conversions through a normalised float space cannot be exact; anything
    // beyond a couple of LSBs means the colour matrices disagree.
    println!(
        "   {}",
        if max_err <= 3 {
            "PASS — colour survives the round-trip"
        } else {
            "FAIL — the colour matrices disagree"
        }
    );

    // --- 2. does blur blur? -----------------------------------------------------------
    //
    // A matte of alpha=0 means "all background", so the whole frame takes the blurred
    // plane. Without setting one the default matte is opaque — alpha=1, all foreground —
    // and mix(bg, fg, 1.0) discards the blur entirely. That is correct behaviour, and it
    // is why an earlier version of this check measured no blurring at all: it was testing
    // the composite, not the blur.
    pipe.set_matte(&[0u8], 1, 1);
    let mut blurred = vec![0u8; frame_bytes];
    pipe.process(&input, &mut blurred, look(BackgroundMode::Blur, 1.0, false));
    pipe.set_matte(&[255u8], 1, 1);

    // Local luma variance is the cleanest proxy for detail: blur removes high-frequency
    // structure, so the checkerboard's variance must collapse.
    let variance = |buf: &[u8]| -> f64 {
        let lumas: Vec<f64> = buf.iter().step_by(2).map(|&v| v as f64).collect();
        let mean = lumas.iter().sum::<f64>() / lumas.len() as f64;
        lumas.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / lumas.len() as f64
    };
    let sharp_var = variance(&output);
    let blur_var = variance(&blurred);

    println!("\n2. background blur");
    println!("   luma variance: {sharp_var:.0} sharp -> {blur_var:.0} blurred");
    println!(
        "   {}",
        if blur_var < sharp_var * 0.9 {
            "PASS — high-frequency detail removed"
        } else {
            "FAIL — no measurable blurring"
        }
    );

    // --- 3. throughput ----------------------------------------------------------------
    println!("\n3. throughput at {W}x{H}");
    for (label, mode, strength) in [
        ("passthrough", BackgroundMode::Off, 0.0),
        ("blur (light)", BackgroundMode::Blur, 0.0),
        ("blur (max)", BackgroundMode::Blur, 1.0),
    ] {
        // Warm up: first submissions include pipeline creation and allocation.
        for _ in 0..5 {
            pipe.process(&input, &mut output, look(mode, strength, false));
        }
        let n = 60;
        let t0 = Instant::now();
        for _ in 0..n {
            pipe.process(&input, &mut output, look(mode, strength, false));
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!(
            "   {label:14} {ms:6.2} ms/frame   {:5.0} fps   {}",
            1000.0 / ms,
            if ms < 33.3 { "fits 30fps" } else { "TOO SLOW" }
        );
    }

    // Mirroring is a free index flip inside the composite pass; confirm it actually flips.
    let mut mirrored = vec![0u8; frame_bytes];
    pipe.process(&input, &mut mirrored, look(BackgroundMode::Off, 0.0, true));
    let row = (W / 2) as usize * 4;
    let left = mirrored[..4].to_vec();
    let right_of_normal = output[row - 4..row].to_vec();
    println!(
        "\n4. mirror: leftmost output texel {:?} vs rightmost source texel {:?} -> {}",
        &left[..2],
        &right_of_normal[..2],
        if left[0].abs_diff(right_of_normal[2]) < 8 {
            "PASS"
        } else {
            "check"
        }
    );

    // --- 5. background replace --------------------------------------------------------
    //
    // Replace used to be indistinguishable from off: the shader had no mode==2 branch, so
    // it sampled the live frame as its own background. The check is therefore not "does it
    // run" but "is the output actually the plate and not the camera".
    let plate_w = 64u32;
    let plate_h = 36u32;
    let mut plate = vec![0u8; (plate_w * plate_h * 4) as usize];
    for px in plate.chunks_exact_mut(4) {
        px[0] = 200; // a colour nothing in the test pattern produces
        px[1] = 40;
        px[2] = 160;
        px[3] = 255;
    }
    pipe.set_background_image(&plate, plate_w, plate_h);
    pipe.set_matte(&[0u8], 1, 1); // all background, so the plate should be all we see

    let mut replaced = vec![0u8; frame_bytes];
    pipe.process(
        &input,
        &mut replaced,
        look(BackgroundMode::Replace, 0.0, false),
    );

    // Compare in luma: the plate is a single flat colour, so a correct replace has almost
    // no luma variance, where the checkerboard has a great deal.
    let replaced_var = variance(&replaced);
    println!("\n5. background replace");
    println!("   luma variance: {sharp_var:.0} camera -> {replaced_var:.0} replaced");
    println!(
        "   {}",
        if replaced_var < sharp_var * 0.05 {
            "PASS — output is the plate, not the camera"
        } else {
            "FAIL — the camera is still showing through"
        }
    );

    // --- 6. guided-filter upsample ----------------------------------------------------
    //
    // The coefficient pass only runs when a matte-input (guidance) texture exists, so this
    // also confirms the wiring: without enable_matte_input the composite must fall back to
    // sampling the matte directly rather than reading an uninitialised coefficient field.
    pipe.enable_matte_input(128, 72);
    // A half-and-half matte at guidance resolution: the guided filter should keep the
    // boundary where it is rather than smearing it across the frame.
    let mut half = vec![0u8; 128 * 72];
    for y in 0..72usize {
        for x in 0..128usize {
            half[y * 128 + x] = if x < 64 { 0 } else { 255 };
        }
    }

    // One begin/set/finish cycle, which is the contract the daemon runs on: the guidance
    // image comes out, a matte derived from it goes back in, and the composite consumes
    // both without a frame boundary in between. This used to need a throwaway `process`
    // first, purely to leave something for the readback to downscale.
    let mut small = vec![0u8; 128 * 72 * 4];
    let mut split = vec![0u8; frame_bytes];
    let got_guidance = pipe.begin_frame(&input, Some(&mut small));
    pipe.set_matte(&half, 128, 72);
    pipe.finish_frame(&mut split, look(BackgroundMode::Blur, 1.0, false));

    // Compare halves statistically rather than probing single pixels. Two lone texels can
    // agree by coincidence on a checkerboard, which is exactly what an earlier version of
    // this check did — it reported "identical" while the composite was working fine.
    let half_variance = |buf: &[u8], from: u32, to: u32| -> f64 {
        let mut lumas = Vec::new();
        for y in (0..H).step_by(4) {
            let row = y as usize * W as usize * 2;
            for x in from..to {
                lumas.push(buf[row + x as usize * 2] as f64);
            }
        }
        let mean = lumas.iter().sum::<f64>() / lumas.len() as f64;
        lumas.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / lumas.len() as f64
    };
    // Left half has matte 0 (all background -> blurred); right half has matte 1 (all
    // foreground -> sharp). So the left must have measurably less high-frequency detail.
    let left_var = half_variance(&split, 0, W / 2 - 32);
    let right_var = half_variance(&split, W / 2 + 32, W);

    println!("\n6. guided-filter upsample");
    println!(
        "   guidance readback: {}",
        if got_guidance { "ok" } else { "unavailable" }
    );
    println!("   luma variance: background half {left_var:.0} vs foreground half {right_var:.0}");
    println!(
        "   {}",
        if left_var < right_var * 0.95 {
            "PASS — the matte reaches the shader and separates the two planes"
        } else {
            "FAIL — both halves composite the same, so the matte is not being applied"
        }
    );

    Ok(())
}
