//! One inference configuration per process, so a shell loop can sweep them.
//!
//! Deliberately *not* a loop inside one process. Constructing more than one ORT session in
//! a single binary is what takes this process down with SIGSEGV (see `impl Drop for
//! Matter`), so a sweep that shares a process would be measuring the crash, not the model.
//!
//! ```text
//! matte_sweep <image> <ep: webgpu|cpu> <ratio> <width> <height>
//! ```
//!
//! Prints one line: the fraction of the frame the network calls foreground, and the peak
//! alpha. For a portrait, `fg>0.5` in the region of 0.10-0.40 with `max` near 1.0 is a
//! working matte; `fg>0.5 = 0.000` with a low `max` is a network that found no subject.

use ort::ep::{CPUExecutionProvider, WebGPU};
use ort::session::Session;
use ort::value::Tensor;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 5 {
        eprintln!("usage: matte_sweep <image> <webgpu|cpu> <ratio> <w> <h>");
        std::process::exit(2);
    }
    let (path, ep, ratio) = (&a[0], &a[1], a[2].parse::<f32>().expect("ratio"));
    let (w, h) = (
        a[3].parse::<u32>().expect("width"),
        a[4].parse::<u32>().expect("height"),
    );

    let img = image::open(path)
        .expect("opening the image")
        .resize_exact(w, h, image::imageops::FilterType::Triangle)
        .to_rgba8();

    let model = cleanroom_matting::find_model().expect("locating the RVM weights");
    let builder = Session::builder().expect("session builder");
    let mut builder = match ep.as_str() {
        "cpu" => builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .expect("cpu ep"),
        _ => builder
            .with_execution_providers([WebGPU::default().build().error_on_failure()])
            .expect("webgpu ep"),
    };
    // Progress markers on stderr. Session creation and the first inference have very
    // different costs on the WebGPU provider, and a single wall-clock number cannot tell
    // "the model took ten minutes to compile" from "every frame is slow".
    eprintln!("[t] building session ({ep})...");
    let t_build = std::time::Instant::now();
    let mut session = builder.commit_from_file(&model).expect("loading the model");
    eprintln!(
        "[t] session ready in {:.1} s",
        t_build.elapsed().as_secs_f64()
    );

    let px = (w * h) as usize;
    let mut input = vec![0.0f32; 3 * px];
    for i in 0..px {
        input[i] = img.as_raw()[i * 4] as f32 / 255.0;
        input[px + i] = img.as_raw()[i * 4 + 1] as f32 / 255.0;
        input[2 * px + i] = img.as_raw()[i * 4 + 2] as f32 / 255.0;
    }

    // Same seed the daemon uses: a (1,1,1,1) zero tensor asks RVM to shape its own state.
    let mut state: Vec<(Vec<i64>, Vec<f32>)> =
        (0..4).map(|_| (vec![1, 1, 1, 1], vec![0.0f32])).collect();

    // Optional 6th argument: how many frames to run. The default settles the recurrent
    // state; a larger count is how the steady-state per-frame cost is measured, since
    // session creation is a one-off the daemon pays at startup.
    let passes: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut timings: Vec<f64> = Vec::with_capacity(passes);

    let mut alpha: Vec<f32> = Vec::new();
    // Recurrent network: let the state settle the way it would over a second of video.
    for _ in 0..passes {
        let t0 = std::time::Instant::now();
        let src = Tensor::from_array((vec![1i64, 3, h as i64, w as i64], input.clone()))
            .expect("src tensor");
        let mut rs = Vec::with_capacity(4);
        for s in &state {
            rs.push(Tensor::from_array((s.0.clone(), s.1.clone())).expect("state tensor"));
        }
        let r = Tensor::from_array((vec![1i64], vec![ratio])).expect("ratio tensor");
        let mut it = rs.into_iter();
        let out = session
            .run(ort::inputs![
                "src" => src,
                "r1i" => it.next().unwrap(),
                "r2i" => it.next().unwrap(),
                "r3i" => it.next().unwrap(),
                "r4i" => it.next().unwrap(),
                "downsample_ratio" => r,
            ])
            .expect("inference");
        for (i, name) in ["r1o", "r2o", "r3o", "r4o"].iter().enumerate() {
            let (shape, data) = out[*name].try_extract_tensor::<f32>().expect("state out");
            state[i] = (shape.iter().copied().collect(), data.to_vec());
        }
        let (_, pha) = out["pha"].try_extract_tensor::<f32>().expect("pha");
        alpha = pha.to_vec();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if timings.len() < 3 {
            eprintln!("[t] frame {} took {ms:.1} ms", timings.len());
        }
        timings.push(ms);
    }

    // Median of the second half: the first frames pay lazy allocation and shape inference,
    // which the daemon pays once at startup and never again.
    let mut steady: Vec<f64> = timings.split_off(timings.len() / 2);
    steady.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = steady[steady.len() / 2];

    let n = alpha.len() as f64;
    let fg = alpha.iter().filter(|&&v| v > 0.5).count() as f64 / n;
    let mean = alpha.iter().map(|&v| v as f64).sum::<f64>() / n;
    let max = alpha.iter().cloned().fold(f32::MIN, f32::max);
    println!(
        "ep={ep:<7} ratio={ratio:<6} src={w}x{h:<5} fg>0.5={fg:.3}  mean={mean:.3}  \
         max={max:.3}  {median:.2} ms/frame"
    );

    // The session owns a Dawn context; dropping it segfaults. Same trade as `Matter`.
    std::mem::forget(session);
}
