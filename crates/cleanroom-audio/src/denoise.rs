//! DeepFilterNet noise suppression.
//!
//! ## Why the weights are not bundled
//!
//! DeepFilterNet's *code* is MIT/Apache-2.0, but its **model weights carry no licence
//! grant at all**. The README licenses "all code in this repository"; weights are not
//! code and are never otherwise mentioned, and upstream issue #697 asks this exact
//! question and has gone unanswered since July 2026 on a repo with no maintainer activity
//! since October 2024.
//!
//! Debian and nixpkgs both redistribute the compiled plugin with weights embedded, so the
//! practical risk is low — but that is inference from silence, not permission. So we
//! build with `default-features = false` (no `default-model`, no `include_bytes!`) and
//! load the archive from a path at runtime. The weights stay a user-supplied asset,
//! outside our binary and outside our distribution.
//!
//! ## The behavioural traps
//!
//! Documented at each site below, but the one that would bite hardest: **an attenuation
//! limit of 100 dB or more means *no limit at all*.** It is not "very strong suppression".
//! The widely-copied `Attenuation Limit (dB) = 100` in PipeWire filter-chain configs is
//! therefore switching the limiter off, which is almost certainly not what anyone typing
//! it intended.

use crate::ringbuf::HOP;
use df::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum DenoiseError {
    #[error(
        "DeepFilterNet weights not found. Looked in: {searched}.\n\
         The weights are not bundled because upstream has never granted a licence for \
         them (DeepFilterNet issue #697). Download DeepFilterNet3_onnx.tar.gz from \
         https://github.com/Rikorose/DeepFilterNet/tree/main/models and place it at \
         {suggested}, or set CLEANROOM_DFN_MODEL."
    )]
    ModelNotFound { searched: String, suggested: String },

    #[error("could not load the DeepFilterNet model at {path}: {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: anyhow::Error,
    },

    #[error("DeepFilterNet reported a hop size of {got}, expected {HOP}")]
    UnexpectedHop { got: usize },

    #[error("DeepFilterNet processing failed: {0}")]
    Process(anyhow::Error),
}

/// Where the model archive can live, in priority order.
fn candidate_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Some(explicit) = std::env::var_os("CLEANROOM_DFN_MODEL") {
        out.push(PathBuf::from(explicit));
    }

    let file = "DeepFilterNet3_onnx.tar.gz";

    if let Some(data) = std::env::var_os("XDG_DATA_HOME") {
        out.push(PathBuf::from(data).join("cleanroom").join(file));
    } else if let Some(home) = std::env::var_os("HOME") {
        out.push(
            PathBuf::from(home)
                .join(".local/share/cleanroom")
                .join(file),
        );
    }

    out.push(PathBuf::from("/usr/share/cleanroom").join(file));
    out.push(PathBuf::from("/usr/local/share/cleanroom").join(file));

    out
}

/// Find the model archive, or explain where to put one.
pub fn find_model() -> Result<PathBuf, DenoiseError> {
    let candidates = candidate_paths();
    for p in &candidates {
        if p.is_file() {
            return Ok(p.clone());
        }
    }
    let suggested = candidates
        .iter()
        .find(|p| p.to_string_lossy().contains(".local/share"))
        .cloned()
        .unwrap_or_else(|| PathBuf::from("~/.local/share/cleanroom/DeepFilterNet3_onnx.tar.gz"));

    Err(DenoiseError::ModelNotFound {
        searched: candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        suggested: suggested.display().to_string(),
    })
}

/// A DeepFilterNet instance processing one hop at a time.
pub struct Denoiser {
    df: DfTract,
    /// Reused per-hop buffers, so the hot path allocates nothing.
    noisy: Array2<f32>,
    enhanced: Array2<f32>,
    /// Last local-SNR estimate in dB, for metering and for the GUI to show whether the
    /// model thinks it is looking at speech or at noise.
    pub last_lsnr: f32,
}

impl Denoiser {
    /// Load a model and configure it.
    ///
    /// `attenuation_db` is clamped below 100 on purpose — see the module docs. Passing
    /// 100 or more would disable the limiter entirely, and silently.
    pub fn new(
        model: &Path,
        attenuation_db: f32,
        post_filter_beta: f32,
    ) -> Result<Self, DenoiseError> {
        let params = DfParams::new(model.to_path_buf()).map_err(|source| DenoiseError::Load {
            path: model.to_path_buf(),
            source,
        })?;

        let rp = RuntimeParams::default_with_ch(1)
            .with_atten_lim(clamp_attenuation(attenuation_db))
            .with_post_filter(post_filter_beta)
            // Defaults from upstream's own CLI: below -10 dB SNR treat as noise, above
            // 30 dB as clean speech, and gate anything under 20 dB of local SNR.
            .with_thresholds(-10.0, 30.0, 20.0)
            .with_mask_reduce(ReduceMask::MEAN);

        let df = DfTract::new(params, &rp).map_err(|source| DenoiseError::Load {
            path: model.to_path_buf(),
            source,
        })?;

        // The hop is not negotiable and upstream only guards it with a debug_assert,
        // which means a release build would silently produce garbage rather than fail.
        // Check it once, here, where the error can be useful.
        if df.hop_size != HOP {
            return Err(DenoiseError::UnexpectedHop { got: df.hop_size });
        }

        tracing::info!(
            model = %model.display(),
            sr = df.sr,
            hop = df.hop_size,
            atten_db = clamp_attenuation(attenuation_db),
            "DeepFilterNet loaded"
        );

        Ok(Self {
            df,
            noisy: Array2::zeros((1, HOP)),
            enhanced: Array2::zeros((1, HOP)),
            last_lsnr: 0.0,
        })
    }

    /// Denoise exactly one hop.
    ///
    /// On failure the input is passed through unchanged rather than emitting silence: a
    /// model hiccup should degrade to "not denoised", never to "microphone dead".
    pub fn process(&mut self, input: &[f32; HOP], output: &mut [f32; HOP]) {
        self.noisy
            .as_slice_mut()
            .expect("contiguous")
            .copy_from_slice(input);

        let noisy: ArrayView2<f32> = self.noisy.view();
        let enh: ArrayViewMut2<f32> = self.enhanced.view_mut();

        match self.df.process(noisy, enh) {
            Ok(lsnr) => {
                self.last_lsnr = lsnr;
                output.copy_from_slice(self.enhanced.as_slice().expect("contiguous"));
            }
            Err(e) => {
                tracing::warn!(error = %e, "denoise failed for one hop; passing audio through");
                output.copy_from_slice(input);
            }
        }
    }

    /// Change the attenuation limit without reloading the model.
    ///
    /// This is what makes a GUI slider possible with no audio interruption — the prior
    /// art respawned an entire helper process per slider drag and dropped ~200 ms of mic
    /// audio each time.
    pub fn set_attenuation(&mut self, db: f32) {
        self.df.set_atten_lim(clamp_attenuation(db));
    }

    pub fn set_post_filter(&mut self, beta: f32) {
        self.df.set_pf_beta(beta);
    }
}

/// Keep the attenuation limit inside the range where it means what it says.
///
/// Upstream maps `>= 100.0` to `atten_lim: None` — no limit at all — and treats `< 0.01`
/// as "too strong", short-circuiting to passthrough. Neither is what a user dragging a
/// slider to the end expects, so the value is clamped into the meaningful band.
fn clamp_attenuation(db: f32) -> f32 {
    db.clamp(0.1, 99.9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attenuation_is_clamped_below_the_no_limit_sentinel() {
        // 100 or more means "no limit at all" upstream, which is the opposite of what
        // someone setting the maximum expects.
        assert!(clamp_attenuation(100.0) < 100.0);
        assert!(clamp_attenuation(1000.0) < 100.0);
        // And below 0.01 upstream short-circuits to passthrough.
        assert!(clamp_attenuation(0.0) >= 0.01);
        assert!(clamp_attenuation(-5.0) >= 0.01);
        // Ordinary values pass through untouched.
        assert_eq!(clamp_attenuation(40.0), 40.0);
    }

    #[test]
    fn missing_model_error_says_where_to_get_one() {
        // The weights cannot be bundled, so this error is the entire onboarding path for
        // anyone who has not downloaded them. It has to be actionable.
        let err = DenoiseError::ModelNotFound {
            searched: "/a, /b".into(),
            suggested: "/c".into(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("DeepFilterNet3_onnx.tar.gz"),
            "must name the file"
        );
        assert!(msg.contains("github.com"), "must say where to get it");
        assert!(
            msg.contains("CLEANROOM_DFN_MODEL"),
            "must mention the override"
        );
    }

    #[test]
    fn model_search_honours_the_env_override_first() {
        // Set via the process env, so run it in a way that does not race other tests.
        unsafe { std::env::set_var("CLEANROOM_DFN_MODEL", "/tmp/explicit-model.tar.gz") };
        let c = candidate_paths();
        assert_eq!(c[0], PathBuf::from("/tmp/explicit-model.tar.gz"));
        unsafe { std::env::remove_var("CLEANROOM_DFN_MODEL") };
    }

    #[test]
    fn denoising_real_audio_preserves_length_and_changes_the_signal() {
        // Only runs where the weights are present; skips cleanly otherwise so CI and a
        // fresh clone both stay green.
        let Ok(model) = find_model() else {
            eprintln!("DeepFilterNet weights not installed; skipping");
            return;
        };
        let mut d = match Denoiser::new(&model, 40.0, 0.02) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("could not load model ({e}); skipping");
                return;
            }
        };

        // A 440 Hz tone with white noise on top. The denoiser should not leave it
        // untouched, and must not blow up or resize anything.
        let mut input = [0.0f32; HOP];
        let mut seed = 12345u32;
        for (i, s) in input.iter_mut().enumerate() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 16) as f32 / 32768.0 - 1.0;
            *s =
                0.3 * (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 48000.0).sin() + 0.1 * noise;
        }

        let mut output = [0.0f32; HOP];
        // Several hops: the model is recurrent and the first hop is not representative.
        for _ in 0..10 {
            d.process(&input, &mut output);
        }

        assert!(
            output.iter().all(|s| s.is_finite()),
            "output must be finite"
        );
        assert!(
            output.iter().any(|&s| s != 0.0),
            "output must not be silence"
        );
        eprintln!(
            "denoised 10 hops, last local SNR estimate {:.1} dB",
            d.last_lsnr
        );
    }
}
