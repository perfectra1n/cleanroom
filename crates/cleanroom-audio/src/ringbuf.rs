//! The ring buffer between PipeWire and the denoiser.
//!
//! These two have incompatible, non-negotiable block sizes:
//!
//! * **PipeWire's quantum is variable.** On the reference machine it is 1024, but it is
//!   negotiated across the whole graph and changes when any other client asks for
//!   something different. You cannot pin it and must not assume it.
//! * **DeepFilterNet's hop is exactly 480 samples** at 48 kHz — 10 ms. Feeding it
//!   anything else is not "slightly wrong": upstream guards it with a `debug_assert`,
//!   which means a *release* build silently produces garbage rather than failing.
//!
//! So every block that arrives is chopped into 480-sample hops, whatever length it came
//! in at, and whatever is left over waits for the next block.
//!
//! ## Why it is primed with silence
//!
//! Without priming, the very first `collect` would come up short: 1024 samples in yields
//! two hops and 64 left over, so only 960 are available where 1024 were asked for. Every
//! shortfall is an audible click.
//!
//! Priming the output with one hop of silence bounds the problem permanently. The deficit
//! at any moment is `samples_in mod 480`, which is strictly less than one hop, so one hop
//! of slack is exactly enough — for any quantum, forever. The cost is 10 ms of latency,
//! which is inaudible and far cheaper than dropping or stretching.

/// DeepFilterNet's hop size at 48 kHz. Not a tunable.
pub const HOP: usize = 480;

/// A single-producer, single-consumer sample queue with a fixed capacity.
///
/// Deliberately not `VecDeque`: this is touched from the PipeWire realtime callback, and
/// a growable container can reallocate, which is exactly the kind of unbounded operation
/// an RT thread must never perform. Capacity is fixed and overrun is explicit.
pub struct SampleRing {
    buf: Vec<f32>,
    read: usize,
    write: usize,
    len: usize,
}

impl SampleRing {
    /// Create a ring holding `hops` hops' worth of samples.
    pub fn with_hops(hops: usize) -> Self {
        Self {
            buf: vec![0.0; hops * HOP],
            read: 0,
            write: 0,
            len: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of whole hops available.
    pub fn hops_available(&self) -> usize {
        self.len / HOP
    }

    /// Append samples. Returns how many were dropped because the ring was full.
    ///
    /// Dropping the *newest* samples on overrun rather than the oldest is deliberate: a
    /// full ring means the consumer has stalled, and the older audio is what is about to
    /// be played. Discarding that would leave a gap; discarding the newest just means a
    /// shorter delay once the consumer catches up.
    pub fn push(&mut self, samples: &[f32]) -> usize {
        let space = self.capacity() - self.len;
        let take = samples.len().min(space);
        for &s in &samples[..take] {
            self.buf[self.write] = s;
            self.write = (self.write + 1) % self.buf.len();
        }
        self.len += take;
        samples.len() - take
    }

    /// Fill `out` from the ring. Returns how many samples were written; any shortfall is
    /// left untouched so the caller decides what to do about it.
    pub fn pop(&mut self, out: &mut [f32]) -> usize {
        let take = out.len().min(self.len);
        for slot in out.iter_mut().take(take) {
            *slot = self.buf[self.read];
            self.read = (self.read + 1) % self.buf.len();
        }
        self.len -= take;
        take
    }

    /// Remove exactly one hop, or return false if a whole hop is not ready.
    pub fn pop_hop(&mut self, out: &mut [f32; HOP]) -> bool {
        if self.len < HOP {
            return false;
        }
        self.pop(out.as_mut_slice());
        true
    }

    /// Append `n` samples of silence.
    pub fn push_silence(&mut self, n: usize) {
        let silence = [0.0f32; HOP];
        let mut remaining = n;
        while remaining > 0 {
            let chunk = remaining.min(HOP);
            self.push(&silence[..chunk]);
            remaining -= chunk;
        }
    }

    pub fn clear(&mut self) {
        self.read = 0;
        self.write = 0;
        self.len = 0;
    }
}

/// Input and output rings plus scratch, sized and primed correctly.
pub struct HopBridge {
    pub input: SampleRing,
    pub output: SampleRing,
    /// Total samples dropped to overrun. Non-zero means the denoiser could not keep up,
    /// which the daemon reports rather than hides.
    pub overruns: u64,
}

impl HopBridge {
    /// `capacity_hops` bounds how much jitter can be absorbed. 32 hops is 320 ms —
    /// enough to ride out a scheduling hiccup, small enough that the latency ceiling
    /// stays sane if something goes badly wrong.
    pub fn new(capacity_hops: usize) -> Self {
        let mut output = SampleRing::with_hops(capacity_hops);

        // The priming that makes every subsequent collect succeed. See the module docs:
        // the deficit is bounded by one hop, so one hop of slack covers it for any quantum.
        output.push_silence(HOP);

        Self {
            input: SampleRing::with_hops(capacity_hops),
            output,
            overruns: 0,
        }
    }

    /// Feed captured audio in.
    pub fn submit(&mut self, samples: &[f32]) {
        let dropped = self.input.push(samples);
        if dropped > 0 {
            self.overruns += dropped as u64;
        }
    }

    /// Run `process` over every whole hop currently available.
    ///
    /// `process` receives one hop and writes one hop. It runs on a worker thread, never
    /// in the PipeWire realtime callback — DeepFilterNet evaluates a neural network per
    /// hop, and doing that in an RT callback is how you get xruns.
    pub fn drain<F>(&mut self, mut process: F)
    where
        F: FnMut(&[f32; HOP], &mut [f32; HOP]),
    {
        let mut inp = [0.0f32; HOP];
        let mut outp = [0.0f32; HOP];
        while self.input.pop_hop(&mut inp) {
            outp.fill(0.0);
            process(&inp, &mut outp);
            let dropped = self.output.push(&outp);
            if dropped > 0 {
                self.overruns += dropped as u64;
            }
        }
    }

    /// Take processed audio out, filling any shortfall with silence.
    ///
    /// A shortfall should be impossible given the priming, but silence is a far better
    /// failure mode than leaving the caller's buffer untouched — that would replay the
    /// previous block, which is a very audible stutter.
    pub fn collect(&mut self, out: &mut [f32]) -> usize {
        let got = self.output.pop(out);
        if got < out.len() {
            out[got..].fill(0.0);
        }
        got
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_size_is_the_deepfilternet_constant() {
        // Guard with teeth: upstream only debug_asserts this, so a wrong value produces
        // silent garbage in a release build rather than a panic.
        assert_eq!(
            HOP, 480,
            "DeepFilterNet's hop at 48kHz is exactly 480 samples"
        );
    }

    #[test]
    fn push_and_pop_round_trip() {
        let mut r = SampleRing::with_hops(4);
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_eq!(r.push(&input), 0);
        assert_eq!(r.len(), 100);

        let mut out = vec![0.0; 100];
        assert_eq!(r.pop(&mut out), 100);
        assert_eq!(out, input);
        assert!(r.is_empty());
    }

    #[test]
    fn wraps_around_without_losing_order() {
        // The classic ring bug: samples come back reordered or duplicated after a wrap.
        let mut r = SampleRing::with_hops(1);
        let mut seen = Vec::new();
        let mut out = vec![0.0; 300];

        for round in 0..5 {
            let chunk: Vec<f32> = (0..300).map(|i| (round * 300 + i) as f32).collect();
            r.push(&chunk);
            let n = r.pop(&mut out);
            seen.extend_from_slice(&out[..n]);
        }
        for w in seen.windows(2) {
            assert!(w[1] > w[0], "samples came back out of order: {w:?}");
        }
    }

    #[test]
    fn overrun_is_reported_rather_than_silently_swallowed() {
        let mut r = SampleRing::with_hops(1);
        let dropped = r.push(&vec![1.0f32; 1000]);
        assert_eq!(dropped, 1000 - 480, "must report exactly what did not fit");
        assert_eq!(r.len(), 480);
    }

    #[test]
    fn the_bridge_is_primed_so_the_first_block_never_runs_short() {
        // The whole reason priming exists. Without it the first output block is 960
        // samples where 1024 were asked for, and that shortfall is an audible click.
        let mut b = HopBridge::new(32);
        assert_eq!(
            b.output.len(),
            HOP,
            "must start with exactly one hop of slack"
        );

        b.submit(&vec![0.5f32; 1024]);
        b.drain(|inp, outp| outp.copy_from_slice(inp));

        let mut out = vec![0.0f32; 1024];
        assert_eq!(
            b.collect(&mut out),
            1024,
            "the first block must fill completely"
        );
    }

    #[test]
    fn survives_a_quantum_that_is_not_a_multiple_of_the_hop() {
        // 1024 is not a multiple of 480, which is the entire difficulty. Run enough
        // blocks that the remainder cycles through every phase.
        let mut b = HopBridge::new(32);
        let block = vec![0.25f32; 1024];
        let mut out = vec![0.0f32; 1024];

        for i in 0..200 {
            b.submit(&block);
            b.drain(|inp, outp| outp.copy_from_slice(inp));
            assert_eq!(
                b.collect(&mut out),
                1024,
                "block {i} came up short — a click"
            );
        }
        assert_eq!(b.overruns, 0, "no overruns should occur in steady state");
    }

    #[test]
    fn survives_a_quantum_that_changes_mid_stream() {
        // PipeWire renegotiates the quantum whenever another client joins the graph, so
        // a fixed-quantum assumption breaks the moment someone opens a browser tab.
        let mut b = HopBridge::new(32);
        for quantum in [1024usize, 512, 2048, 256, 480, 96, 1024] {
            let block = vec![0.1f32; quantum];
            let mut out = vec![0.0f32; quantum];
            for _ in 0..30 {
                b.submit(&block);
                b.drain(|inp, outp| outp.copy_from_slice(inp));
                assert_eq!(
                    b.collect(&mut out),
                    quantum,
                    "short block after switching to quantum {quantum}"
                );
            }
        }
        assert_eq!(b.overruns, 0);
    }

    #[test]
    fn drain_always_hands_the_denoiser_exactly_one_hop() {
        // Non-negotiable: DeepFilterNet only debug_asserts this, so a release build
        // would silently misbehave rather than fail.
        let mut b = HopBridge::new(32);
        b.submit(&vec![1.0f32; 3000]);
        let mut calls = 0;
        b.drain(|inp, outp| {
            assert_eq!(inp.len(), HOP);
            assert_eq!(outp.len(), HOP);
            calls += 1;
            outp.copy_from_slice(inp);
        });
        assert_eq!(calls, 3000 / HOP, "every whole hop, and no partial one");
    }

    #[test]
    fn a_stalled_consumer_produces_overruns_not_a_panic() {
        // If the denoiser thread wedges, the RT callback must keep working regardless.
        let mut b = HopBridge::new(2);
        for _ in 0..50 {
            b.submit(&vec![1.0f32; 1024]);
        }
        assert!(
            b.overruns > 0,
            "a stalled consumer must be visible as overruns"
        );
    }

    #[test]
    fn collect_pads_with_silence_rather_than_stale_audio() {
        // A shortfall must not leave the caller's previous contents in place — that
        // replays the last block, which is a very audible stutter.
        let mut b = HopBridge::new(32);
        let mut out = vec![7.0f32; 1024];
        let got = b.collect(&mut out);
        assert_eq!(got, HOP, "only the primed hop is available yet");
        assert!(
            out[got..].iter().all(|&s| s == 0.0),
            "the shortfall must be silence, not whatever was in the buffer"
        );
    }

    #[test]
    fn audio_is_preserved_exactly_through_a_passthrough_bridge() {
        // End-to-end sample fidelity: what goes in must come out, offset only by the
        // one hop of priming latency.
        let mut b = HopBridge::new(64);
        let signal: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut collected = Vec::new();
        let mut out = vec![0.0f32; 1024];

        for chunk in signal.chunks(1024) {
            b.submit(chunk);
            b.drain(|inp, outp| outp.copy_from_slice(inp));
            let n = b.collect(&mut out);
            collected.extend_from_slice(&out[..n]);
        }

        // Skip the primed hop of silence; the rest must match the input sample for sample.
        let recovered = &collected[HOP..];
        let compare = recovered.len().min(signal.len());
        for i in 0..compare {
            assert!(
                (recovered[i] - signal[i]).abs() < 1e-6,
                "sample {i} changed: {} != {}",
                recovered[i],
                signal[i]
            );
        }
    }
}
