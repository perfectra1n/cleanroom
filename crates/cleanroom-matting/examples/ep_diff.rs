//! Run one ONNX model on one execution provider and print a fingerprint of every output.
//!
//! The point is minimal reproducers. `matte_sweep` can only drive RVM, because it hardcodes
//! that model's six inputs — so it cannot answer "is *this one operator* wrong?", which is
//! the question left after every whole-model hypothesis has been eliminated.
//!
//! ```text
//! ep_diff <model.onnx> <cpu|webgpu> <name>:<d0>x<d1>x... [more inputs...]
//! ```
//!
//! Inputs are filled with a fixed, deterministic ramp rather than random values, so two runs
//! on two providers are comparing identical bytes and any difference in the printed
//! fingerprint is the provider's doing.
//!
//! ```sh
//! # is depthwise convolution computed correctly on this provider?
//! ep_diff dw.onnx cpu    X:1x8x32x32
//! ep_diff dw.onnx webgpu X:1x8x32x32
//! ```

use ort::ep::{CPUExecutionProvider, WebGPU};
use ort::session::Session;
use ort::value::Tensor;

/// A deterministic, non-degenerate ramp.
///
/// Not zeros and not a constant: a kernel that reads the wrong element, or drops a group,
/// still produces the right answer on constant input, which is exactly the bug class being
/// hunted. Bounded to a small range so nothing saturates an activation.
fn fill(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % 17) as f32 / 17.0) - 0.5 + ((i % 5) as f32 * 0.03))
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!("usage: ep_diff <model.onnx> <cpu|webgpu> <name>:<d0>x<d1>x... [...]");
        std::process::exit(2);
    }
    let (model, ep) = (&a[0], &a[1]);

    let builder = Session::builder().expect("session builder");
    let mut builder = match ep.as_str() {
        "cpu" => builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .expect("cpu ep"),
        _ => builder
            .with_execution_providers([WebGPU::default().build().error_on_failure()])
            .expect("webgpu ep"),
    };
    let mut session = builder.commit_from_file(model).expect("loading the model");

    let mut names = Vec::new();
    let mut tensors = Vec::new();
    for spec in &a[2..] {
        let (name, rest) = spec.split_once(':').expect("input spec is name:AxBxC[=v]");
        // `=v` fills with a constant instead of the ramp. Needed for inputs that are
        // parameters rather than data — RVM's `downsample_ratio` must be positive, and a
        // ramp starting at -0.5 makes Resize refuse the model rather than compute it.
        let (dims, konst) = match rest.split_once('=') {
            Some((d, v)) => (d, Some(v.parse::<f32>().expect("constant value"))),
            None => (rest, None),
        };
        let shape: Vec<i64> = dims
            .split('x')
            .map(|d| d.parse().expect("dimension"))
            .collect();
        let n: i64 = shape.iter().product();
        let data = match konst {
            Some(v) => vec![v; n as usize],
            None => fill(n as usize),
        };
        names.push(name.to_string());
        tensors.push(Tensor::from_array((shape, data)).expect("input tensor"));
    }

    let inputs: Vec<(&str, ort::value::Value)> = names
        .iter()
        .map(|s| s.as_str())
        .zip(tensors.into_iter().map(|t| t.into_dyn()))
        .collect();

    // Scoped so `outputs` — which borrows the session — is dropped before the leak below.
    {
        let outputs = session.run(inputs).expect("inference");
        for (name, value) in outputs.iter() {
            match value.try_extract_tensor::<f32>() {
                Ok((shape, data)) => {
                    let n = data.len() as f64;
                    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
                    let min = data.iter().cloned().fold(f32::MAX, f32::min);
                    let max = data.iter().cloned().fold(f32::MIN, f32::max);
                    // A sum of |v| catches sign and ordering errors that mean alone hides.
                    let l1 = data.iter().map(|&v| v.abs() as f64).sum::<f64>();
                    println!(
                        "{ep:<7} {name:<12} shape={:?} mean={mean:+.6} min={min:+.6} max={max:+.6} L1={l1:.4}",
                        shape.as_ref()
                    );
                }
                Err(e) => println!("{ep:<7} {name:<12} (not f32: {e})"),
            }
        }
    }

    // Same trade as `Matter`: the session owns a provider context and dropping it segfaults.
    std::mem::forget(session);
}
