//! M0a spike 3 — can `ort` run Robust Video Matting on a vendor-neutral GPU EP?
//!
//! This is a go/no-go gate for the whole matting design, and it answers three questions
//! in order of how likely they are to kill the plan:
//!
//!   1. Does a linux-x64 ONNX Runtime build with the **WebGPU** execution provider exist
//!      at all? WebGPU is the only EP that gives GPU inference on both Nvidia and AMD
//!      without a vendor SDK. nixpkgs does not build it — its onnxruntime override args
//!      expose only coreml/cuda/nccl/openvino/rocm, and there is no Dawn in the store —
//!      so we depend on `ort`'s `download-binaries` prebuilt.
//!   2. Does RVM load through it? The upstream RVM export reuses the same symbolic
//!      height/width dimension names for the frame input and for every recurrent state
//!      tensor. TensorRT read those as equality constraints and refused to build an
//!      engine. If WebGPU's shape inference trips the same way, we rewrite the names.
//!   3. Is it fast enough?
//!
//! Failure is cheap and informative: the fallback is hand-written WGSL kernels, which
//! changes nothing else in the architecture.
//!
//! Run:  cargo run -p spike-ort-rvm -- [path-to-rvm_mobilenetv3_fp32.onnx]

use anyhow::{Result, anyhow, bail};
use ort::ep::{ExecutionProvider, WebGPU};
use ort::session::Session;
use ort::value::Tensor;
use std::time::{Duration, Instant};

/// RVM's matting resolution. The network is fully convolutional, so this is free;
/// 512x288 is the 16:9 box the reference implementation uses for HD input.
const INFER_W: usize = 512;
const INFER_H: usize = 288;

/// Enough frames to exercise the recurrent path repeatedly. A single frame would miss
/// state-shape bugs entirely — those only appear on frame 2, once r1o..r4o have been
/// fed back in as r1i..r4i.
const FRAMES: usize = 60;

/// Frames to discard before measuring. Covers lazy shader compilation and the recurrent
/// state settling from its (1,1,1,1) seed; neither is steady-state cost.
const WARMUP: usize = 10;

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/perf3ct/repos/nvidia-broadcast-linux/models/rvm_mobilenetv3_fp32.onnx".into()
    });

    println!("=== ort / RVM / WebGPU spike ===");
    println!("model: {model_path}");
    if !std::path::Path::new(&model_path).exists() {
        bail!("model not found at {model_path}");
    }

    // --- Question 1 -------------------------------------------------------------
    // Ask ONNX Runtime what it was *compiled* with. No Rust-side feature flag can
    // conjure an EP that isn't in the binary, so this is the honest answer, and it
    // prints before anything can fail so a crash below still leaves the diagnostic.
    println!("\n--- question 1: is the WebGPU EP in this build? ---");
    let webgpu_available = WebGPU::default()
        .is_available()
        .map_err(|e| anyhow!("could not query provider availability: {e}"))?;
    println!(
        "WebGPU EP compiled in: {}",
        if webgpu_available { "YES" } else { "NO" }
    );
    if !webgpu_available {
        println!(
            "\n  The downloaded prebuilt has no WebGPU support. Options:\n\
             \x20   a) build ONNX Runtime with --use_webgpu (pulls in Dawn; heavy)\n\
             \x20   b) drop ort and hand-write the RVM ops as WGSL compute shaders\n\
             \x20 (b) is the planned fallback and changes nothing else in the design."
        );
    }

    // --- Question 2 -------------------------------------------------------------
    println!("\n--- question 2: does RVM load on it? ---");

    // `.error_on_failure()` is the single most important call in this file. Without it
    // ort silently falls back to CPU when an EP can't be registered, and we would "pass"
    // this spike while measuring entirely the wrong thing. Silent CPU fallback is the
    // exact failure mode this project exists to avoid.
    let session = Session::builder()
        .map_err(|e| anyhow!("session builder: {e}"))?
        .with_execution_providers([WebGPU::default().build().error_on_failure()])
        .map_err(|e| {
            anyhow!(
                "failed to register the WebGPU execution provider: {e}\n\
                 (this is .error_on_failure() doing its job — without it we would have \
                 silently continued on CPU)"
            )
        })?
        .commit_from_file(&model_path)
        .map_err(|e| {
            anyhow!(
                "failed to load RVM: {e}\n\
                 If this mentions dimension or shape constraints, it is the reused-symbolic-dim \
                 problem: RVM's export gives the frame input and every r1i..r4i state tensor the \
                 same symbolic H/W names. Fix by rewriting each to unique symbols before loading."
            )
        })?;

    println!("loaded OK");
    println!("\ninputs:");
    for i in session.inputs() {
        println!("  {:<20} {:?}", i.name(), i.dtype());
    }
    println!("outputs:");
    for o in session.outputs() {
        println!("  {:<20} {:?}", o.name(), o.dtype());
    }

    // --- Question 3 -------------------------------------------------------------
    infer_loop(session)
}

fn infer_loop(mut session: Session) -> Result<()> {
    println!("\n--- question 3: is it fast enough? ({FRAMES} frames @ {INFER_W}x{INFER_H}) ---");

    // Recurrent state. RVM auto-shapes r1i..r4i from a (1,1,1,1) zero tensor on the
    // first frame; thereafter each frame's r*o becomes the next frame's r*i. We own
    // these buffers rather than holding the previous run's output values, so ORT's
    // allocator is free to recycle its own arena between runs.
    let mut state: Vec<(Vec<i64>, Vec<f32>)> =
        (0..4).map(|_| (vec![1, 1, 1, 1], vec![0.0f32])).collect();

    let mut frame = vec![0.0f32; 3 * INFER_H * INFER_W];
    let mut times: Vec<Duration> = Vec::with_capacity(FRAMES);
    let mut last_alpha = (0.0f32, 0.0f32);

    for f in 0..FRAMES {
        fill_test_frame(&mut frame, f);

        let src =
            Tensor::from_array((vec![1i64, 3, INFER_H as i64, INFER_W as i64], frame.clone()))
                .map_err(|e| anyhow!("src tensor: {e}"))?;
        let r1 =
            Tensor::from_array((state[0].0.clone(), state[0].1.clone())).map_err(tag("r1i"))?;
        let r2 =
            Tensor::from_array((state[1].0.clone(), state[1].1.clone())).map_err(tag("r2i"))?;
        let r3 =
            Tensor::from_array((state[2].0.clone(), state[2].1.clone())).map_err(tag("r3i"))?;
        let r4 =
            Tensor::from_array((state[3].0.clone(), state[3].1.clone())).map_err(tag("r4i"))?;
        // RVM's internal downsample ratio; 0.25 is the reference default for HD.
        let ratio = Tensor::from_array((vec![1i64], vec![0.25f32])).map_err(tag("ratio"))?;

        let t0 = Instant::now();
        let outputs = session
            .run(ort::inputs![
                "src" => src,
                "r1i" => r1,
                "r2i" => r2,
                "r3i" => r3,
                "r4i" => r4,
                "downsample_ratio" => ratio,
            ])
            .map_err(|e| anyhow!("inference failed on frame {f}: {e}"))?;
        times.push(t0.elapsed());

        for (i, name) in ["r1o", "r2o", "r3o", "r4o"].iter().enumerate() {
            let (shape, data) = outputs[*name]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("extract {name}: {e}"))?;
            state[i] = (shape.iter().copied().collect(), data.to_vec());
        }

        let (_, pha) = outputs["pha"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("extract pha: {e}"))?;
        last_alpha = (
            pha.iter().copied().fold(f32::INFINITY, f32::min),
            pha.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        );

        if f == 0 {
            println!(
                "recurrent state shapes after frame 0: {:?}",
                state.iter().map(|s| &s.0).collect::<Vec<_>>()
            );
        }
    }

    let warm = &times[WARMUP.min(times.len() - 1)..];
    let mean = warm.iter().sum::<Duration>() / warm.len() as u32;
    let worst = warm.iter().max().unwrap();
    let mean_ms = mean.as_secs_f64() * 1000.0;

    println!("\nsteady-state mean:  {mean_ms:.2} ms");
    println!("steady-state worst: {:.2} ms", worst.as_secs_f64() * 1000.0);
    println!(
        "alpha range, last frame: {:.4} .. {:.4}",
        last_alpha.0, last_alpha.1
    );

    // The degenerate-alpha guard, stated correctly. The failure mode it exists to catch
    // is a matte that is flat and *high* — RVM fed a blank frame emits a near-uniform
    // ~0.96 alpha, which composites as a one-frame "effect off" flash. A flat *low*
    // alpha is not a fault: it is the network correctly reporting no subject, which is
    // the right answer for the synthetic gradient this spike feeds it. An earlier
    // version of this check tested spread alone and cried wolf on every run.
    let spread = last_alpha.1 - last_alpha.0;
    if spread < 0.005 && last_alpha.0 > 0.5 {
        println!(
            "WARNING: flat HIGH alpha (min {:.4}, spread {spread:.6}) — the classic \
                  blank-frame signature",
            last_alpha.0
        );
    } else if spread < 0.005 {
        println!(
            "note: alpha is uniformly {:.3}. Expected here — the synthetic test frame has no \
             human in it, and RVM is trained on people. Not a fault; use a real frame to \
             validate matte quality.",
            last_alpha.0
        );
    }

    println!("\n--- verdict ---");
    println!("1080p30 gives a 33.3 ms total frame budget; matting should be well under 10 ms.");
    println!(
        "mean {mean_ms:.2} ms => {}",
        if mean_ms < 10.0 {
            "PASS"
        } else {
            "OVER BUDGET (fine for a slow-GPU conformance target)"
        }
    );

    // Teardown. Dropping an ORT session that owns a WebGPU/Dawn context segfaults on
    // this stack, reproducibly, on BOTH the Nvidia and AMD adapters, after all work has
    // completed successfully.
    //
    // This is not cosmetic for the daemon: a systemd unit with Restart=on-failure would
    // see the segfault as a failed exit and restart us in a loop, forever. So we need a
    // deliberate answer rather than ignoring it.
    //
    // CLEANROOM_SPIKE_TEARDOWN=drop  -> drop normally, reproduce the crash
    // CLEANROOM_SPIKE_TEARDOWN=leak  -> forget the session (default)
    match std::env::var("CLEANROOM_SPIKE_TEARDOWN").as_deref() {
        Ok("drop") => {
            println!("\nteardown: dropping session normally (expect a segfault)");
            drop(session);
            println!("teardown: survived the drop");
        }
        _ => {
            println!("\nteardown: leaking the session to dodge the Dawn teardown crash");
            // The process is about to exit; the OS reclaims everything. Leaking is the
            // cheap, correct move here. The daemon will need the same treatment, or a
            // shutdown path that tears the GPU context down in a controlled order.
            std::mem::forget(session);
        }
    }
    println!("teardown: clean exit");
    Ok(())
}

fn tag(what: &'static str) -> impl Fn(ort::Error) -> anyhow::Error {
    move |e| anyhow!("{what} tensor: {e}")
}

/// Deterministic moving gradient with a soft ellipse standing in for a subject.
/// Reproducible across runs so a later CPU-vs-GPU numeric comparison is meaningful,
/// and moving so the recurrent path does not trivially converge the way a constant
/// frame would.
fn fill_test_frame(buf: &mut [f32], frame_idx: usize) {
    let plane = INFER_H * INFER_W;
    let phase = frame_idx as f32 * 0.05;
    let cx = INFER_W as f32 * (0.5 + 0.15 * phase.sin());
    let cy = INFER_H as f32 * 0.55;
    let rx = INFER_W as f32 * 0.18;
    let ry = INFER_H as f32 * 0.35;

    for y in 0..INFER_H {
        for x in 0..INFER_W {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            let inside = (1.0 - (dx * dx + dy * dy)).clamp(0.0, 1.0);
            let bg = 0.2 + 0.3 * (x as f32 / INFER_W as f32);
            let v = bg + (0.75 - bg) * inside;

            let i = y * INFER_W + x;
            buf[i] = v;
            buf[plane + i] = v * 0.85;
            buf[2 * plane + i] = v * 0.7;
        }
    }
}
