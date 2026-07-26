//! Robust Video Matting on a vendor-neutral GPU.
//!
//! RVM is a *recurrent* network: it carries four state tensors between frames, which is
//! why its edges are stable under motion where a per-frame segmenter shimmers. That
//! statefulness is the whole reason it was chosen, and it is also the thing most easily got
//! wrong — the state must be threaded frame to frame, and reset whenever the input geometry
//! changes.
//!
//! ## Why the weights are not vendored
//!
//! RVM is GPL-3.0, which is compatible with this project, but the mobilenetv3 export is
//! ~15 MB and the resnet50 one ~107 MB. They load from a path at runtime, resolved the same
//! way as the DeepFilterNet weights.
//!
//! ## A hazard that is real but does not bite here
//!
//! The upstream RVM ONNX export reuses the *same* symbolic height/width dimension names for
//! the frame input and for every recurrent state tensor. TensorRT reads those as equality
//! constraints and refuses to build an engine, which is why the prior art had to rewrite
//! them. ONNX Runtime's WebGPU shape inference does not, so no rewriting is needed — but
//! the model dump makes the shared names plainly visible, so it is worth knowing before
//! someone tries a different execution provider.

use ort::ep::{ExecutionProvider, WebGPU};
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

/// Matting input width. The network is fully convolutional, so this is free; 512x288 is the
/// 16:9 box the reference implementation uses for HD input, and what the spike measured.
pub const INFER_W: u32 = 512;
pub const INFER_H: u32 = 288;

/// RVM's internal downsample ratio. 0.25 is the reference default for HD.
const DOWNSAMPLE_RATIO: f32 = 0.25;

#[derive(Debug, thiserror::Error)]
pub enum MattingError {
    #[error(
        "matting model not found. Looked in: {searched}.\n\
         Download rvm_mobilenetv3_fp32.onnx from \
         https://github.com/PeterL1n/RobustVideoMatting/releases and place it at \
         {suggested}, or set CLEANROOM_RVM_MODEL."
    )]
    ModelNotFound { searched: String, suggested: String },

    #[error(
        "could not register the WebGPU execution provider: {0}\n\
         This is deliberate rather than a silent fall back to CPU: CPU matting cannot hold \
         a 33 ms frame budget, and a slow path nobody notices is worse than an error."
    )]
    NoGpu(String),

    // Field deliberately not named `source`: thiserror treats that name as an error
    // source and requires it to implement Error, which a String does not.
    #[error("could not load {path}: {detail}")]
    Load { path: PathBuf, detail: String },

    #[error("inference failed: {0}")]
    Inference(String),
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("CLEANROOM_RVM_MODEL") {
        out.push(PathBuf::from(p));
    }
    let file = "rvm_mobilenetv3_fp32.onnx";
    if let Some(d) = std::env::var_os("XDG_DATA_HOME") {
        out.push(PathBuf::from(d).join("cleanroom").join(file));
    } else if let Some(h) = std::env::var_os("HOME") {
        out.push(PathBuf::from(h).join(".local/share/cleanroom").join(file));
    }
    out.push(PathBuf::from("/usr/share/cleanroom").join(file));
    out.push(PathBuf::from("/usr/local/share/cleanroom").join(file));
    out
}

pub fn find_model() -> Result<PathBuf, MattingError> {
    let c = candidate_paths();
    for p in &c {
        if p.is_file() {
            return Ok(p.clone());
        }
    }
    Err(MattingError::ModelNotFound {
        searched: c
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        suggested: c
            .iter()
            .find(|p| p.to_string_lossy().contains(".local/share"))
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "~/.local/share/cleanroom/rvm_mobilenetv3_fp32.onnx".into()),
    })
}

/// One recurrent state tensor: its shape and contents.
type State = (Vec<i64>, Vec<f32>);

pub struct Matter {
    /// `Option` so `Drop` can take it out and deliberately leak it. See the `Drop` impl.
    session: Option<Session>,
    /// r1i..r4i, carried frame to frame. This is what makes the matte temporally stable.
    state: Vec<State>,
    /// Reusable NCHW input, so the per-frame path allocates nothing.
    input: Vec<f32>,
    /// Last alpha as 8-bit, ready to upload as an R8 texture.
    alpha: Vec<u8>,
    /// Previous alpha, kept so a degenerate frame can be replaced rather than shown.
    prev_alpha: Vec<u8>,
    /// How many frames were rejected by the degenerate-alpha guard.
    pub rejected: u64,
    frames: u64,
}

impl Matter {
    pub fn new(model: &Path) -> Result<Self, MattingError> {
        // `.error_on_failure()` is the single most important call here. Without it ort
        // silently registers nothing and runs on the CPU, and the pipeline would appear to
        // work while missing its budget by 10x — the exact silent degradation this project
        // exists to avoid. During the spike this call caught a real fault (Dawn could not
        // find libvulkan.so.1) that would otherwise have produced a fabricated GPU number.
        let session = Session::builder()
            .map_err(|e| MattingError::NoGpu(e.to_string()))?
            .with_execution_providers([WebGPU::default().build().error_on_failure()])
            .map_err(|e| MattingError::NoGpu(e.to_string()))?
            .commit_from_file(model)
            .map_err(|e| MattingError::Load {
                path: model.to_path_buf(),
                detail: e.to_string(),
            })?;

        let available = WebGPU::default().is_available().unwrap_or(false);
        tracing::info!(
            model = %model.display(),
            webgpu = available,
            "matting model loaded ({}x{})",
            INFER_W,
            INFER_H
        );

        let px = (INFER_W * INFER_H) as usize;
        Ok(Self {
            session: Some(session),
            // A (1,1,1,1) zero tensor is how RVM is told to auto-shape its state on the
            // first frame; thereafter each r*o becomes the next r*i.
            state: (0..4).map(|_| (vec![1, 1, 1, 1], vec![0.0f32])).collect(),
            input: vec![0.0; 3 * px],
            alpha: vec![0; px],
            prev_alpha: vec![255; px],
            rejected: 0,
            frames: 0,
        })
    }

    /// Reset the recurrent state.
    ///
    /// Must be called when the input geometry changes: the state tensors are shaped from
    /// the first frame, and feeding a different size afterwards is a shape error.
    pub fn reset(&mut self) {
        self.state = (0..4).map(|_| (vec![1, 1, 1, 1], vec![0.0f32])).collect();
        self.frames = 0;
    }

    /// Run one frame.
    ///
    /// `rgba` is tightly packed RGBA8 at [`INFER_W`]x[`INFER_H`] — what the GPU downscale
    /// pass produces. Returns the alpha matte as R8 at the same size, ready to upload.
    pub fn infer(&mut self, rgba: &[u8]) -> Result<&[u8], MattingError> {
        let px = (INFER_W * INFER_H) as usize;
        debug_assert_eq!(rgba.len(), px * 4);

        // Interleaved RGBA to planar NCHW float, which is what RVM's input expects.
        for i in 0..px {
            self.input[i] = rgba[i * 4] as f32 / 255.0;
            self.input[px + i] = rgba[i * 4 + 1] as f32 / 255.0;
            self.input[2 * px + i] = rgba[i * 4 + 2] as f32 / 255.0;
        }

        let src = Tensor::from_array((
            vec![1i64, 3, INFER_H as i64, INFER_W as i64],
            self.input.clone(),
        ))
        .map_err(|e| MattingError::Inference(e.to_string()))?;

        let mut rs = Vec::with_capacity(4);
        for s in &self.state {
            rs.push(
                Tensor::from_array((s.0.clone(), s.1.clone()))
                    .map_err(|e| MattingError::Inference(e.to_string()))?,
            );
        }
        let ratio = Tensor::from_array((vec![1i64], vec![DOWNSAMPLE_RATIO]))
            .map_err(|e| MattingError::Inference(e.to_string()))?;

        let mut it = rs.into_iter();
        let outputs = self
            .session
            .as_mut()
            .expect("session is only taken in Drop")
            .run(ort::inputs![
                "src" => src,
                "r1i" => it.next().unwrap(),
                "r2i" => it.next().unwrap(),
                "r3i" => it.next().unwrap(),
                "r4i" => it.next().unwrap(),
                "downsample_ratio" => ratio,
            ])
            .map_err(|e| MattingError::Inference(e.to_string()))?;

        // Carry the recurrent state forward. This is the step that makes the matte stable
        // frame to frame rather than flickering.
        for (i, name) in ["r1o", "r2o", "r3o", "r4o"].iter().enumerate() {
            let (shape, data) = outputs[*name]
                .try_extract_tensor::<f32>()
                .map_err(|e| MattingError::Inference(e.to_string()))?;
            self.state[i] = (shape.iter().copied().collect(), data.to_vec());
        }

        let (_, pha) = outputs["pha"]
            .try_extract_tensor::<f32>()
            .map_err(|e| MattingError::Inference(e.to_string()))?;

        // Degenerate-alpha guard.
        //
        // RVM fed a blank or malformed frame emits a near-uniform high alpha, which
        // composites as a one-frame "effect off" flash — sporadic, and worse under motion.
        // A flat *low* alpha is not a fault: that is the network correctly reporting no
        // subject. So the test is flat AND high, and on a hit we keep the previous matte,
        // which is invisible for one frame where committing the bad one is not.
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in pha.iter() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let degenerate = self.frames > 0 && (hi - lo) < 0.005 && lo > 0.5;

        if degenerate {
            self.rejected += 1;
            tracing::debug!(min = lo, spread = hi - lo, "rejected a degenerate matte");
            return Ok(&self.prev_alpha);
        }

        for (dst, &v) in self.alpha.iter_mut().zip(pha.iter()) {
            *dst = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
        self.prev_alpha.copy_from_slice(&self.alpha);
        self.frames += 1;
        Ok(&self.alpha)
    }
}

impl Drop for Matter {
    fn drop(&mut self) {
        // Deliberately leak the ONNX Runtime session.
        //
        // Dropping a session that owns a WebGPU/Dawn context segfaults, reproducibly, on
        // both the Nvidia and AMD adapters, *after* all inference has completed
        // successfully. This was found during the M0a spike and confirmed again here: two
        // Matter instances in one test binary took the process down with SIGSEGV.
        //
        // This is not cosmetic. The daemon's unit specifies Restart=on-failure, and a
        // segfault at exit is a failed exit — so without this, every clean shutdown would
        // be read as a crash and systemd would restart us forever.
        //
        // Leaking is the correct trade here: a Matter lives for the life of a pipeline, the
        // memory is reclaimed by the OS at exit, and the alternative is a crash loop.
        if let Some(session) = self.session.take() {
            std::mem::forget(session);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_error_is_actionable() {
        let e = MattingError::ModelNotFound {
            searched: "/a".into(),
            suggested: "/b".into(),
        };
        let m = e.to_string();
        assert!(m.contains("rvm_mobilenetv3"), "must name the file");
        assert!(m.contains("github.com"), "must say where to get it");
        assert!(
            m.contains("CLEANROOM_RVM_MODEL"),
            "must mention the override"
        );
    }

    #[test]
    fn no_gpu_error_explains_why_there_is_no_cpu_fallback() {
        let m = MattingError::NoGpu("dawn sad".into()).to_string();
        assert!(m.contains("deliberate"), "must not read like a bug: {m}");
    }

    /// Both model-dependent checks live in ONE test on purpose.
    ///
    /// Cargo runs tests on parallel threads, and creating two ONNX Runtime sessions that
    /// each own a WebGPU/Dawn context *concurrently* aborts the process (SIGABRT). Creating
    /// them sequentially is fine — verified by `examples/two_sessions.rs` — and the daemon
    /// only ever holds one at a time, so the constraint is on concurrent construction, not
    /// on lifetime. Splitting this back into two tests reintroduces the crash.
    #[test]
    fn matting_a_real_frame_produces_a_usable_matte_and_resets() {
        let Ok(model) = find_model() else {
            eprintln!("RVM weights not installed; skipping");
            return;
        };
        let mut m = match Matter::new(&model) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("could not load the model ({e}); skipping");
                return;
            }
        };

        let px = (INFER_W * INFER_H) as usize;
        let mut frame = vec![0u8; px * 4];
        // A soft ellipse on a gradient — not a person, so a low alpha is the *correct*
        // answer here. This checks shape, range and state hand-off, not matte quality.
        for y in 0..INFER_H as usize {
            for x in 0..INFER_W as usize {
                let i = (y * INFER_W as usize + x) * 4;
                let dx = (x as f32 - INFER_W as f32 * 0.5) / (INFER_W as f32 * 0.2);
                let dy = (y as f32 - INFER_H as f32 * 0.55) / (INFER_H as f32 * 0.35);
                let inside = (1.0 - (dx * dx + dy * dy)).clamp(0.0, 1.0);
                let v = (0.25 + 0.5 * inside) * 255.0;
                frame[i] = v as u8;
                frame[i + 1] = (v * 0.9) as u8;
                frame[i + 2] = (v * 0.8) as u8;
                frame[i + 3] = 255;
            }
        }

        // Several frames: the network is recurrent, so frame 1 is not representative, and
        // more than one is the only way to exercise the state hand-off at all.
        let mut last_len = 0;
        for _ in 0..8 {
            last_len = m.infer(&frame).expect("inference must succeed").len();
        }
        assert_eq!(
            last_len, px,
            "matte must be one byte per pixel at infer size"
        );

        // The state must have been shaped away from its (1,1,1,1) seed.
        assert!(
            m.state
                .iter()
                .all(|(shape, _)| shape.len() == 4 && shape[1] > 1),
            "recurrent state was never shaped: {:?}",
            m.state.iter().map(|s| &s.0).collect::<Vec<_>>()
        );
        eprintln!(
            "state shapes: {:?}, rejected {}",
            m.state.iter().map(|s| &s.0).collect::<Vec<_>>(),
            m.rejected
        );

        // Reset must restore the seed, or a resolution change becomes a shape error.
        m.reset();
        assert!(
            m.state.iter().all(|(shape, _)| shape == &vec![1, 1, 1, 1]),
            "reset must restore the auto-shaping seed"
        );
    }
}
