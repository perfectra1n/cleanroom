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

use ort::ep::{CPUExecutionProvider, WebGPU};
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

/// Matting input width. The network is fully convolutional, so this is free; 512x288 is the
/// 16:9 box the reference implementation uses for HD input, and what the spike measured.
///
/// These remain the defaults, but the working resolution is now a runtime value: the CPU
/// provider costs 39 ms/frame here and 10.6 ms at 256x144, and the difference between those
/// two is the difference between holding 30 fps and not. See [`Backend`].
pub const INFER_W: u32 = 512;
pub const INFER_H: u32 = 288;

/// Which execution provider runs the network.
///
/// This is not a performance knob. The WebGPU provider in ONNX Runtime 1.24.2 returns a
/// **wrong** matte for this model — see [`Matter::infer`]'s cross-check — and the whole
/// reason this enum exists is that "fast" and "correct" turned out to be different
/// backends on the reference machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Try the GPU, then prove it against the CPU on the first usable frame and switch if
    /// the GPU is wrong. The only setting that cannot silently produce a dead matte.
    Auto,
    /// Force the WebGPU provider and trust it. Fast; verify it yourself.
    Gpu,
    /// Force the CPU provider. Slower, and correct everywhere it has been measured.
    Cpu,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Backend::Auto => "auto",
            Backend::Gpu => "gpu",
            Backend::Cpu => "cpu",
        })
    }
}

impl std::str::FromStr for Backend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Backend::Auto),
            "gpu" | "webgpu" => Ok(Backend::Gpu),
            "cpu" => Ok(Backend::Cpu),
            other => Err(format!(
                "unknown matting backend `{other}` (expected auto, gpu or cpu)"
            )),
        }
    }
}

/// Alpha above which a pixel counts as subject.
const SUBJECT_ALPHA: f32 = 0.5;

/// Fraction of the frame that has to read as subject before the matte is "working".
///
/// Deliberately a *coverage* test rather than a peak one. Peak alpha is a single pixel and
/// therefore noise: the broken WebGPU path still produced the occasional stray texel above
/// 0.5, which was enough to make a peak-based check declare it healthy while every frame
/// came out uniformly blurred. Coverage cannot be faked that way — measured on the same
/// frame, the CPU provider called 22.7% of pixels subject and the WebGPU provider 0.0%.
///
/// 2% of a 16:9 frame is a person occupying about a seventh of the width at full height:
/// far smaller than anyone sits from a webcam, and far larger than any amount of speckle.
const MIN_SUBJECT_COVERAGE: f32 = 0.02;

/// Consecutive subject-less frames that trigger a cross-check against the CPU.
///
/// Verification is continuous rather than one-shot, and the reason is the exact shape of
/// the WebGPU defect. Measured on one static frame run repeatedly through the provider, the
/// mean alpha decays 0.039 -> 0.027 -> 0.018 -> 0.005 -> 0.002 over ten frames: the
/// recurrent state degrades on every hand-off until the matte is gone. The CPU provider
/// holds 0.227 indefinitely on the same input.
///
/// A one-shot check at startup passes cleanly and then latches, because early frames still
/// carry signal — and the failure appears later, precisely when the subject stops moving.
/// Which is to say: it works in testing and breaks the moment somebody sits still in a
/// meeting. So the matte is watched for as long as the GPU is in use.
///
/// 45 frames is 1.5 s at 30 fps: longer than any blink of the matte, shorter than anyone
/// would tolerate watching their own face blurred.
const COLLAPSE_STREAK: u32 = 45;

/// Ceiling for the backoff between inconclusive cross-checks.
///
/// A check is inconclusive when *neither* provider finds a subject, which is what an empty
/// room looks like — and an empty room can last all day. Doubling the wait each time keeps
/// the answer prompt when someone sits down in the first few seconds without paying 40 ms
/// of CPU inference every second forever when nobody does.
const VALIDATION_MAX_WAIT: u32 = 900;

/// RVM's internal downsample ratio, derived rather than hardcoded.
///
/// This is the scale RVM applies *internally*, before its encoder; the deep guided filter
/// then refines the result back up to `src`'s resolution. Upstream's rule is to choose it
/// so the **downsampled** width lands between 256 and 512 px.
///
/// The widely-quoted 0.25 is the value for a **1920**-wide `src`. Ours is already
/// `INFER_W` wide, so reusing 0.25 here ran the encoder at 128x72 — a quarter of the
/// intended linear resolution, which is exactly the "soft, unstable edge" failure. Writing
/// it as a formula means the correct value follows automatically if `INFER_W` ever moves.
///
/// Measured in the daemon at 1920x1080/30, RTX 5090, as `matting_ms` (which also covers
/// the matte readback, so it is not comparable to the spike's bare inference number):
///
/// | ratio | encoder input | matting_ms | fps |
/// |-------|---------------|------------|-----|
/// | 0.25  | 128x72        | 7.60       | 30.0 |
/// | 1.00  | 512x288       | 9.03-10.13 | 30.0 |
///
/// ~2 ms for 4x the segmentation resolution, inside a 33 ms budget. Worth it.
const DOWNSAMPLE_RATIO: f32 = if INFER_W > 512 {
    512.0 / INFER_W as f32
} else {
    1.0
};

/// Weight given to a *rising* alpha sample when the matte is otherwise stable.
///
/// Higher than `ALPHA_FALL` because gaining subject early is invisible, where losing it
/// early punches a hole in a moving limb.
const ALPHA_RISE: f32 = 0.55;

/// Weight given to a *falling* alpha sample when the matte is otherwise stable.
const ALPHA_FALL: f32 = 0.22;

/// The per-pixel alpha change at which temporal damping is fully released.
///
/// Below this, a change is treated as noise and averaged; at or above it, the new sample is
/// taken essentially whole, because a jump that large is the network reporting real motion
/// and averaging it is what produces ghost trails.
const MOTION_FULL: f32 = 0.25;

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
         `video.matting_backend = auto` falls back to the CPU provider here and says so; \
         this error means the GPU was asked for explicitly, and silently running 4x slower \
         would be worse than failing."
    )]
    NoGpu(String),

    // Field deliberately not named `source`: thiserror treats that name as an error
    // source and requires it to implement Error, which a String does not.
    #[error("could not load {path}: {detail}")]
    Load { path: PathBuf, detail: String },

    #[error("inference failed: {0}")]
    Inference(String),
}

/// The channel-padded export, preferred over the stock one wherever both exist.
///
/// Not an optimisation — a correctness fix. ONNX Runtime's WebGPU provider computes `Conv`
/// wrongly when the input channel count is divisible by 3 and not by 4, and RVM's decoder
/// hits that in exactly one node (`Conv_200`, 171 input channels) because it concatenates
/// the 3-channel source image onto a skip connection. Padding that conv to 172 channels is
/// an exact identity — the added weights are zero — and it is the difference between a
/// correct matte on the GPU at 7.3 ms and a dead one.
///
/// Produced by `tools/onnx/pad_conv_channels.py`. See `docs/pitfalls.md`.
pub const PADDED_MODEL: &str = "rvm_mobilenetv3_fp32.padded.onnx";

/// The stock upstream export.
pub const STOCK_MODEL: &str = "rvm_mobilenetv3_fp32.onnx";

fn candidate_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("CLEANROOM_RVM_MODEL") {
        out.push(PathBuf::from(p));
    }
    // Padded first in every directory, so a machine that has both silently gets the one
    // that is correct on the GPU rather than the one that happens to be named upstream.
    let dirs: Vec<PathBuf> = {
        let mut d = Vec::new();
        if let Some(x) = std::env::var_os("XDG_DATA_HOME") {
            d.push(PathBuf::from(x).join("cleanroom"));
        } else if let Some(h) = std::env::var_os("HOME") {
            d.push(PathBuf::from(h).join(".local/share/cleanroom"));
        }
        d.push(PathBuf::from("/usr/share/cleanroom"));
        d.push(PathBuf::from("/usr/local/share/cleanroom"));
        d
    };
    for file in [PADDED_MODEL, STOCK_MODEL] {
        for dir in &dirs {
            out.push(dir.join(file));
        }
    }
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
    /// Temporally smoothed alpha in f32, carried between frames. Kept at full precision
    /// rather than reusing the quantised `prev_alpha`: an EMA that reads back its own 8-bit
    /// output cannot resolve changes below 1/255 and stalls short of its target.
    smoothed: Vec<f32>,
    /// How many frames were rejected by the degenerate-alpha guard.
    pub rejected: u64,
    frames: u64,

    /// Inference geometry. Runtime rather than `const` because the viable resolution
    /// depends on which provider ended up running.
    width: u32,
    height: u32,
    /// The provider actually in use. Never `Auto` — that is a request, not a result.
    backend: Backend,
    /// Set while the GPU is in use under `Auto` and therefore still on probation — which is
    /// for as long as it runs, not just at startup. Holds the model path, because proving
    /// the GPU wrong means building a CPU session to compare against.
    unproven: Option<PathBuf>,
    /// Consecutive frames the GPU has reported no subject, against [`COLLAPSE_STREAK`].
    validation_frames: u32,
    /// Streak length required for the next check; doubles on each inconclusive result so an
    /// empty room does not pay for a CPU inference over and over.
    validation_wait: u32,
    /// Why the backend ended up where it did, for `status` and `doctor` to report. Empty
    /// when nothing surprising happened.
    pub note: String,
}

/// Serialises session construction, process-wide.
///
/// Creating two ONNX Runtime sessions that each own a WebGPU/Dawn context *concurrently*
/// aborts the process with SIGABRT — not an error, not a panic, an abort with no unwinding
/// and nothing to catch. Sequential construction is fine, proven by `examples/two_sessions.rs`.
///
/// This was previously guarded only by a comment and by the two model-dependent tests being
/// deliberately merged into one, because cargo runs tests on parallel threads. That works
/// right up until somebody splits the test back apart or adds a second caller, at which
/// point the failure is a hard abort in CI with no stack. A mutex makes the constraint
/// something the program enforces rather than something a reader has to know.
///
/// It is held across construction only, never across inference.
static SESSION_INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// How many intra-op threads the CPU provider may use.
///
/// Not "as many as there are cores". This runs beside a GPU pipeline on the same thread,
/// and past a handful of threads the extra parallelism buys less than the scheduling
/// pressure costs — see `build_session`. `CLEANROOM_MATTING_THREADS` overrides it for
/// anyone measuring on very different hardware.
fn matting_threads() -> usize {
    if let Some(n) = std::env::var("CLEANROOM_MATTING_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| (n.get() / 4).clamp(2, 6))
        .unwrap_or(4)
}

/// Build one session on one provider.
///
/// `.error_on_failure()` on the GPU path is the single most important call here. Without it
/// ort silently registers nothing and runs on the CPU, and the pipeline would appear to work
/// while missing its budget — the exact silent degradation this project exists to avoid.
/// During the spike it caught a real fault (Dawn could not find libvulkan.so.1) that would
/// otherwise have produced a fabricated GPU number.
fn build_session(model: &Path, backend: Backend) -> Result<Session, MattingError> {
    // Poisoning is irrelevant here: the guard protects an external library's global
    // initialisation, not any state of ours, so a previous panic leaves nothing invalid.
    let _serialise = SESSION_INIT.lock().unwrap_or_else(|e| e.into_inner());

    let builder = Session::builder().map_err(|e| MattingError::NoGpu(e.to_string()))?;
    let mut builder = match backend {
        Backend::Cpu => {
            let b = builder
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .map_err(|e| MattingError::NoGpu(e.to_string()))?;

            // Bound the thread pool, and stop it spinning.
            //
            // Not a throughput optimisation — measured on a 24-core machine this is worth
            // nothing either way, 30 fps and ~11 ms/frame with 6 threads or with 24. It is
            // about being a reasonable thing to have running in the background: ONNX
            // Runtime's CPU provider otherwise takes one intra-op thread per core and
            // *busy-waits* between operators, so an idle-ish video call would sit there
            // spinning two dozen cores for a 320x180 network. A webcam effect has no
            // business being the largest consumer of a laptop's battery.
            let threads = matting_threads();
            let b = b
                .with_intra_threads(threads)
                .map_err(|e| MattingError::NoGpu(e.to_string()))?;
            b.with_config_entry("session.intra_op.allow_spinning", "0")
                .map_err(|e| MattingError::NoGpu(e.to_string()))?
        }
        _ => builder
            .with_execution_providers([WebGPU::default().build().error_on_failure()])
            .map_err(|e| MattingError::NoGpu(e.to_string()))?,
    };
    builder
        .commit_from_file(model)
        .map_err(|e| MattingError::Load {
            path: model.to_path_buf(),
            detail: e.to_string(),
        })
}

/// Fraction of the matte that reads as subject.
///
/// The one statistic that distinguishes a working matte from a dead one, and the reason it
/// is a fraction rather than a maximum: see [`MIN_SUBJECT_COVERAGE`].
fn subject_coverage(pha: &[f32]) -> f32 {
    if pha.is_empty() {
        return 0.0;
    }
    pha.iter().filter(|&&v| v > SUBJECT_ALPHA).count() as f32 / pha.len() as f32
}

/// One forward pass: threads the recurrent state in and out, returns the alpha plane.
///
/// A free function rather than a method so the cross-check can drive a *second*, temporary
/// session with the same code. Returning an owned `Vec` costs one 590 kB copy per frame at
/// 512x288 (~60 us, against 7-40 ms of inference) and buys the caller a result that does not
/// borrow the session — which is what lets `infer` keep using `&mut self` afterwards.
fn run_frame(
    session: &mut Session,
    input: &[f32],
    state: &mut [State],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, MattingError> {
    let src = Tensor::from_array((vec![1i64, 3, height as i64, width as i64], input.to_vec()))
        .map_err(|e| MattingError::Inference(e.to_string()))?;

    let mut rs = Vec::with_capacity(4);
    for s in state.iter() {
        rs.push(
            Tensor::from_array((s.0.clone(), s.1.clone()))
                .map_err(|e| MattingError::Inference(e.to_string()))?,
        );
    }
    let ratio = Tensor::from_array((vec![1i64], vec![DOWNSAMPLE_RATIO]))
        .map_err(|e| MattingError::Inference(e.to_string()))?;

    let mut it = rs.into_iter();
    let outputs = session
        .run(ort::inputs![
            "src" => src,
            "r1i" => it.next().unwrap(),
            "r2i" => it.next().unwrap(),
            "r3i" => it.next().unwrap(),
            "r4i" => it.next().unwrap(),
            "downsample_ratio" => ratio,
        ])
        .map_err(|e| MattingError::Inference(e.to_string()))?;

    // Carry the recurrent state forward. This is the step that makes the matte stable frame
    // to frame rather than flickering.
    for (i, name) in ["r1o", "r2o", "r3o", "r4o"].iter().enumerate() {
        let (shape, data) = outputs[*name]
            .try_extract_tensor::<f32>()
            .map_err(|e| MattingError::Inference(e.to_string()))?;
        state[i] = (shape.iter().copied().collect(), data.to_vec());
    }

    let (_, pha) = outputs["pha"]
        .try_extract_tensor::<f32>()
        .map_err(|e| MattingError::Inference(e.to_string()))?;
    Ok(pha.to_vec())
}

impl Matter {
    /// Load the model at `width`x`height` on `requested`.
    ///
    /// `Backend::Auto` prefers the GPU but does not trust it: the first frame containing a
    /// subject is run through a CPU session as well, and the GPU is kept only if it agrees.
    /// See [`Matter::infer`].
    pub fn new(
        model: &Path,
        requested: Backend,
        width: u32,
        height: u32,
    ) -> Result<Self, MattingError> {
        let mut note = String::new();
        let (session, backend, unproven) = match requested {
            Backend::Cpu => (build_session(model, Backend::Cpu)?, Backend::Cpu, None),
            Backend::Gpu => (build_session(model, Backend::Gpu)?, Backend::Gpu, None),
            Backend::Auto => match build_session(model, Backend::Gpu) {
                Ok(s) => (s, Backend::Gpu, Some(model.to_path_buf())),
                Err(e) => {
                    // No GPU provider at all is a legitimate machine, not a failure. Say so
                    // rather than refusing to start: a correct slow matte beats none.
                    note = format!("no GPU provider ({e}); using the CPU provider");
                    tracing::warn!(error = %e, "no WebGPU provider; falling back to CPU matting");
                    (build_session(model, Backend::Cpu)?, Backend::Cpu, None)
                }
            },
        };

        tracing::info!(
            model = %model.display(),
            backend = %backend,
            unproven = unproven.is_some(),
            "matting model loaded ({width}x{height})"
        );

        let px = (width * height) as usize;
        Ok(Self {
            session: Some(session),
            // A (1,1,1,1) zero tensor is how RVM is told to auto-shape its state on the
            // first frame; thereafter each r*o becomes the next r*i.
            state: (0..4).map(|_| (vec![1, 1, 1, 1], vec![0.0f32])).collect(),
            input: vec![0.0; 3 * px],
            alpha: vec![0; px],
            prev_alpha: vec![255; px],
            smoothed: Vec::new(),
            rejected: 0,
            frames: 0,
            width,
            height,
            backend,
            unproven,
            validation_frames: 0,
            validation_wait: COLLAPSE_STREAK,
            note,
        })
    }

    /// The provider actually running the network.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Inference width, which the caller must match when sizing the downscale.
    pub fn infer_w(&self) -> u32 {
        self.width
    }

    /// Inference height.
    pub fn infer_h(&self) -> u32 {
        self.height
    }

    /// Whether the GPU provider is in use and still on probation.
    ///
    /// Stays true for as long as the GPU runs under `Auto`. The defect being watched for is
    /// a matte that decays over seconds, so there is no point at which the provider can be
    /// declared good and stop being checked.
    pub fn is_unproven(&self) -> bool {
        self.unproven.is_some()
    }

    /// Reset the recurrent state, as if no frame had ever been seen.
    ///
    /// Must be called when the input geometry changes: the state tensors are shaped from
    /// the first frame, and feeding a different size afterwards is a shape error. It is
    /// also the right thing to do after any gap in the frame stream — power save releasing
    /// the camera, a suspend, a run of inference errors — because the recurrent state
    /// encodes "what the scene looked like a frame ago", and after an arbitrary gap that is
    /// a statement about a scene that no longer exists.
    ///
    /// `prev_alpha` is reset too. It is what the degenerate-alpha guard substitutes for a
    /// rejected frame, so leaving a stale matte behind means the first bad frame after a
    /// reset composites the *old* scene's silhouette onto the new one.
    pub fn reset(&mut self) {
        self.state = (0..4).map(|_| (vec![1, 1, 1, 1], vec![0.0f32])).collect();
        self.prev_alpha.fill(255);
        // Drop the smoothing history too: it is an average of frames from a scene that is
        // no longer in front of the camera, so blending the next frame into it would fade
        // the old subject out rather than showing the new one.
        self.smoothed.clear();
        self.frames = 0;
    }

    /// Run one frame.
    ///
    /// `rgba` is tightly packed RGBA8 at [`Matter::infer_w`]x[`Matter::infer_h`] — what the
    /// GPU downscale pass produces. Returns the alpha matte as R8 at the same size, ready to
    /// upload.
    pub fn infer(&mut self, rgba: &[u8]) -> Result<&[u8], MattingError> {
        let px = (self.width * self.height) as usize;
        debug_assert_eq!(rgba.len(), px * 4);

        // Interleaved RGBA to planar NCHW float, which is what RVM's input expects.
        for i in 0..px {
            self.input[i] = rgba[i * 4] as f32 / 255.0;
            self.input[px + i] = rgba[i * 4 + 1] as f32 / 255.0;
            self.input[2 * px + i] = rgba[i * 4 + 2] as f32 / 255.0;
        }

        let session = self
            .session
            .as_mut()
            .expect("session is only taken in Drop");
        let pha = run_frame(
            session,
            &self.input,
            &mut self.state,
            self.width,
            self.height,
        )?;

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

        // Prove the GPU before trusting it — see `cross_check`.
        if self.unproven.is_some() {
            let coverage = subject_coverage(&pha);
            self.cross_check(coverage);
        }

        Self::smooth_into(&mut self.smoothed, self.frames, &pha);
        for (dst, &v) in self.alpha.iter_mut().zip(self.smoothed.iter()) {
            *dst = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
        self.prev_alpha.copy_from_slice(&self.alpha);
        self.frames += 1;
        Ok(&self.alpha)
    }

    /// Decide whether the GPU provider is telling the truth, using the CPU as the oracle.
    ///
    /// The motivating defect: ONNX Runtime 1.24.2's WebGPU provider runs this model roughly
    /// four times faster than the CPU provider and returns an alpha matte that is zero
    /// everywhere. Nothing about that is detectable from inside a single provider — the
    /// session builds, inference succeeds, the timings look excellent, and the composite
    /// dutifully treats the entire frame as background and blurs the subject along with the
    /// room. `.error_on_failure()` cannot catch it, because nothing fails.
    ///
    /// So the only honest check is a second opinion. When the GPU reports a frame with no
    /// subject in it, run the *same* pixels through a CPU session: if the CPU finds a
    /// subject where the GPU found none, the GPU is wrong and we switch permanently.
    ///
    /// Frames where the GPU already sees a subject prove it works, and cost nothing.
    fn cross_check(&mut self, gpu_coverage: f32) {
        // The GPU is finding a subject right now, so there is nothing to investigate. The
        // streak resets rather than the provider being marked proven for good: this same
        // provider produces a healthy matte and then lets it decay away over the following
        // seconds, so "it worked a moment ago" is not evidence that it works now.
        if gpu_coverage >= MIN_SUBJECT_COVERAGE {
            self.validation_frames = 0;
            return;
        }

        self.validation_frames += 1;

        // An empty room also produces no subject, and that is not a fault. Only a sustained
        // absence is worth spending a CPU inference to explain.
        if self.validation_frames < self.validation_wait {
            return;
        }

        let Some(model) = self.unproven.take() else {
            return;
        };

        // One CPU session, one frame, on the pixels the GPU just called empty.
        let verdict = build_session(&model, Backend::Cpu).and_then(|mut cpu| {
            let mut fresh: Vec<State> = (0..4).map(|_| (vec![1, 1, 1, 1], vec![0.0f32])).collect();
            run_frame(&mut cpu, &self.input, &mut fresh, self.width, self.height).inspect(|_| {
                // Same trade as `Drop`: this session owns a provider context, and dropping
                // it is what segfaults. It is one session for the life of the process.
                std::mem::forget(cpu);
            })
        });

        match verdict {
            Ok(cpu_pha) => {
                let cpu_coverage = subject_coverage(&cpu_pha);
                if cpu_coverage < MIN_SUBJECT_COVERAGE {
                    // Both blank: nobody is in front of the camera. Inconclusive, so leave
                    // the GPU alone and re-arm rather than switching on no evidence.
                    self.unproven = Some(model);
                    self.validation_frames = 0;
                    self.validation_wait = (self.validation_wait * 2).min(VALIDATION_MAX_WAIT);
                    tracing::debug!(
                        cpu_coverage,
                        gpu_coverage,
                        next_check_in = self.validation_wait,
                        "matting cross-check inconclusive; no subject on either provider"
                    );
                    return;
                }

                // The CPU found a subject in pixels the GPU called empty. The GPU is wrong.
                self.note = format!(
                    "the GPU provider found {:.1}% of the frame to be subject where the CPU \
                     found {:.1}%; switched to the CPU provider",
                    gpu_coverage * 100.0,
                    cpu_coverage * 100.0
                );
                tracing::warn!(gpu_coverage, cpu_coverage, "{}", self.note);
                match build_session(&model, Backend::Cpu) {
                    Ok(s) => {
                        if let Some(old) = self.session.replace(s) {
                            std::mem::forget(old);
                        }
                        self.backend = Backend::Cpu;
                        self.reset();
                    }
                    Err(e) => {
                        self.note =
                            format!("the GPU matte looks wrong but the CPU provider failed: {e}");
                        tracing::error!(error = %e, "could not switch to the CPU provider");
                    }
                }
            }
            Err(e) => {
                self.note = format!("could not verify the GPU matte against the CPU: {e}");
                tracing::warn!(error = %e, "matting cross-check failed");
            }
        }
    }

    /// Motion-adaptive, asymmetric temporal smoothing of the alpha matte.
    ///
    /// Two things are going on, and both are asymmetries.
    ///
    /// **Motion-adaptive.** A per-pixel exponential average kills the frame-to-frame
    /// shimmer that makes an edge look like it is boiling, but applied uniformly it also
    /// smears anything that moves. So the blend weight is driven by how much that pixel
    /// actually changed: where the matte is stable, smooth hard; where it jumped, trust the
    /// new value and barely smooth at all. A large jump is the network reporting real
    /// motion, and averaging it with history is precisely how you get a limb dragging a
    /// ghost behind it.
    ///
    /// **Asymmetric in direction.** Falling alpha — foreground becoming background — is
    /// damped harder than rising. Getting this wrong is very visible: when someone moves,
    /// the trailing edge of an arm turns to background a frame or two before the network is
    /// confident, and the hole punched there shows the *blurred* room through the middle of
    /// their sleeve. Rising alpha has no equivalent failure; the worst case is a few
    /// milliseconds of extra subject, which nobody sees.
    fn smooth_into(smoothed: &mut Vec<f32>, frames: u64, pha: &[f32]) {
        // First real frame: nothing to blend against, so take it as-is. Blending against
        // the seeded matte would fade the subject in over the first few frames.
        if frames == 0 || smoothed.len() != pha.len() {
            smoothed.clear();
            smoothed.extend(pha.iter().map(|v| v.clamp(0.0, 1.0)));
            return;
        }

        for (prev, &now) in smoothed.iter_mut().zip(pha.iter()) {
            let now = now.clamp(0.0, 1.0);
            let delta = now - *prev;

            // Base weight on the new sample, by direction.
            let base = if delta < 0.0 { ALPHA_FALL } else { ALPHA_RISE };

            // Release the damping in proportion to how big the change is, so a genuine
            // movement is followed immediately while noise around zero is averaged away.
            // At |delta| >= MOTION_FULL the new sample is taken essentially whole.
            let motion = (delta.abs() / MOTION_FULL).min(1.0);
            let w = base + (1.0 - base) * motion;

            *prev += delta * w;
        }
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
        //
        // `CLEANROOM_DROP_ORT_SESSION=1` takes the honest path instead, so the defect can be
        // re-tested on a new driver, a new ort, or a new Dawn without editing this file.
        // `examples/teardown_check.rs` is the harness. If it ever exits 0, this leak can go —
        // see docs/pitfalls.md for what has already been tried.
        if let Some(session) = self.session.take() {
            if std::env::var_os("CLEANROOM_DROP_ORT_SESSION").is_some() {
                tracing::warn!("dropping the ORT session deliberately; this may segfault");
                drop(session);
            } else {
                std::mem::forget(session);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regression this whole cross-check exists for.
    ///
    /// A peak-based test called the broken WebGPU provider healthy, because a matte that is
    /// zero everywhere still throws the odd stray texel over 0.5. Coverage is what actually
    /// separates the two, and these are the measured numbers from the reference machine on
    /// one identical frame: 22.7% of pixels for the CPU provider, 0.0% for WebGPU.
    #[test]
    fn coverage_separates_a_working_matte_from_a_dead_one() {
        let px = 512 * 288;

        // A dead matte with speckle: nothing but noise, plus a handful of hot pixels.
        let mut dead = vec![0.01f32; px];
        for i in 0..20 {
            dead[i * 997] = 0.9;
        }
        assert!(
            subject_coverage(&dead) < MIN_SUBJECT_COVERAGE,
            "speckle must not read as a subject, got {}",
            subject_coverage(&dead)
        );
        assert!(
            dead.iter().cloned().fold(f32::MIN, f32::max) > SUBJECT_ALPHA,
            "this matte must be one a peak-based check would have wrongly passed"
        );

        // A working matte: a subject occupying a fifth of the frame.
        let mut live = vec![0.0f32; px];
        live[..px / 5].fill(1.0);
        assert!(
            subject_coverage(&live) >= MIN_SUBJECT_COVERAGE,
            "a fifth of the frame is plainly a subject"
        );
    }

    /// An empty room is not a broken GPU. Both providers legitimately return nothing, and
    /// switching on that would be a false positive that costs 4x the inference time.
    #[test]
    fn an_empty_frame_reads_as_no_subject_on_any_provider() {
        assert_eq!(subject_coverage(&vec![0.0f32; 1000]), 0.0);
        assert_eq!(subject_coverage(&[]), 0.0);
    }

    #[test]
    fn backend_parses_the_names_a_user_would_type() {
        use std::str::FromStr;
        assert_eq!(Backend::from_str("auto").unwrap(), Backend::Auto);
        assert_eq!(Backend::from_str("CPU").unwrap(), Backend::Cpu);
        assert_eq!(Backend::from_str(" gpu ").unwrap(), Backend::Gpu);
        // `webgpu` is what the provider is called in ort and in every log line, so someone
        // reading a log and typing what they saw must not get an error.
        assert_eq!(Backend::from_str("webgpu").unwrap(), Backend::Gpu);
        assert!(Backend::from_str("cuda").is_err());
    }

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

    /// The error is only reachable when the GPU was demanded explicitly, so it has to say
    /// which setting relaxes it. There *is* a CPU fallback now — the failure mode this
    /// project cares about turned out to be a GPU that runs and returns nonsense, not a GPU
    /// that refuses to start — so an error implying no alternative exists would send someone
    /// hunting for a Vulkan problem instead of typing one word.
    #[test]
    fn no_gpu_error_names_the_setting_that_falls_back() {
        let m = MattingError::NoGpu("dawn sad".into()).to_string();
        assert!(
            m.contains("dawn sad"),
            "must keep the underlying cause: {m}"
        );
        assert!(
            m.contains("matting_backend"),
            "must name the setting that falls back: {m}"
        );
    }

    /// The asymmetry is the point, so it gets a test rather than a comment.
    ///
    /// Alpha falling means foreground turning into background. If that is allowed to happen
    /// as readily as the reverse, the trailing edge of a moving arm becomes background a
    /// frame early and the blurred room shows through the middle of a sleeve. Gaining
    /// subject early has no equivalent cost, so rising is allowed to move faster.
    #[test]
    fn falling_alpha_is_damped_harder_than_rising() {
        let step = 0.1; // well below MOTION_FULL, so the direction weights dominate

        let mut rising = vec![0.5f32];
        Self_smooth(&mut rising, &[0.5 + step]);
        let gained = rising[0] - 0.5;

        let mut falling = vec![0.5f32];
        Self_smooth(&mut falling, &[0.5 - step]);
        let lost = 0.5 - falling[0];

        assert!(
            lost < gained,
            "falling must move less than rising: fell {lost}, rose {gained}"
        );
    }

    /// Without this, the smoothing that removes shimmer also produces ghost trails: a real
    /// movement gets averaged with where the subject used to be.
    #[test]
    fn a_large_change_is_followed_almost_immediately() {
        let mut s = vec![0.0f32];
        Self_smooth(&mut s, &[1.0]);
        assert!(
            s[0] > 0.95,
            "a full-range jump must be taken nearly whole, got {}",
            s[0]
        );
    }

    /// Small changes are noise on a stationary subject, and averaging them is the entire
    /// reason this filter exists.
    #[test]
    fn small_changes_are_averaged_away() {
        let mut s = vec![0.5f32];
        Self_smooth(&mut s, &[0.52]);
        assert!(s[0] < 0.515, "a 0.02 wobble should be damped, got {}", s[0]);
    }

    /// Helper: run one smoothing step against an established history.
    #[allow(non_snake_case)]
    fn Self_smooth(state: &mut Vec<f32>, next: &[f32]) {
        // frames > 0 so the first-frame passthrough does not apply.
        Matter::smooth_into(state, 1, next);
    }

    /// Guards a bug that produced a *plausible* matte rather than an error.
    ///
    /// `DOWNSAMPLE_RATIO` was 0.25 — correct for a 1920-wide `src`, but ours is `INFER_W`
    /// wide, so RVM's encoder ran at 128x72 and the deep guided filter upsampled that back
    /// to 512x288. Nothing failed; the matte was simply built from a quarter of the linear
    /// detail, which reads as a soft, unstable edge rather than as a fault.
    ///
    /// Upstream's rule is that the *downsampled* side should land between 256 and 512.
    #[test]
    fn downsample_ratio_keeps_the_encoder_in_rvms_recommended_band() {
        let effective = INFER_W as f32 * DOWNSAMPLE_RATIO;
        assert!(
            (256.0..=512.0).contains(&effective),
            "encoder would run at {effective}px wide; RVM wants 256..=512 \
             (INFER_W = {INFER_W}, ratio = {DOWNSAMPLE_RATIO})"
        );
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
        // The CPU provider, deliberately. This test asserts shape, range and state hand-off,
        // and it must do so against the provider that is known to compute them correctly —
        // a GPU run here would be testing the machine's driver stack rather than this code.
        let mut m = match Matter::new(&model, Backend::Cpu, INFER_W, INFER_H) {
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
        // …and must not leave the previous scene's matte behind. `prev_alpha` is what the
        // degenerate-alpha guard substitutes for a rejected frame, so a stale one composites
        // the *old* silhouette onto the new scene — visible only as a brief wrong cut-out,
        // which is exactly the kind of fault nobody manages to reproduce on demand.
        assert!(
            m.prev_alpha.iter().all(|&v| v == 255),
            "reset must clear the fallback matte too"
        );
    }
}
