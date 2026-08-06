//! The lock-free bridge between PipeWire's realtime callbacks and the denoise worker.
//!
//! Three parties, three block sizes, and a hard rule about who may wait:
//!
//! * **PipeWire's quantum is variable.** On the reference machine it is 1024, but it is
//!   negotiated across the whole graph and changes when any other client asks for
//!   something different. You cannot pin it and must not assume it.
//! * **DeepFilterNet's hop is exactly 480 samples** at 48 kHz — 10 ms. Feeding it
//!   anything else is not "slightly wrong": upstream guards it with a `debug_assert`,
//!   which means a *release* build silently produces garbage rather than failing.
//! * **The RT callbacks may not block, allocate, or run inference.** An earlier shape of
//!   this module documented a worker thread and then never spawned one: the denoiser ran
//!   inside the playback stream's `RT_PROCESS` callback, behind a `Mutex`, with a config
//!   snapshot (a lock *and* a heap allocation) taken per hop. Every cycle where the
//!   forward pass overran the quantum budget became silence padding — audibly choppy,
//!   only with denoise enabled, and invisible to any "is the audio silent" test because
//!   the gaps are sub-hop and scattered.
//!
//! So the shape is now actually built: two wait-free SPSC rings ([`rtrb`]) with a
//! normal-priority worker between them. The capture callback pushes raw samples and wakes
//! the worker; the worker chops them into hops, runs the denoiser, and pushes the result;
//! the playback callback pops. Nothing on the RT path does more than a bounded memcpy and
//! an atomic.
//!
//! ## The intra-cycle race, and why the output pops whole cycles or nothing
//!
//! Within one graph cycle the capture callback and the playback callback run
//! microseconds apart on the same RT thread. A normal-priority worker cannot win that
//! race, must not be asked to, and so the playback side can never count on *this*
//! cycle's audio — only on a standing lead of processed samples from previous cycles.
//!
//! That lead has to be built, and partial pops would prevent it: popping whatever is
//! available and padding the rest yields a sub-hop shortfall every cycle — precisely the
//! choppiness this module exists to end, delivered by other means. Instead
//! [`CycleReader`] fills whole cycles or answers with pure silence while the ring keeps
//! accumulating, and holds delivery back until a full quantum-plus-hop lead exists (see
//! its docs for why the hysteresis is load-bearing). The deficit is repaid in a couple
//! of settling cycles at the start of the stream and after a quantum increase, rather
//! than bled out forever. The steady-state cost is one quantum plus up to two hops of
//! latency: around 30 ms at a 1024 quantum, unremarkable next to the latency every
//! denoising mic adds anyway.
//!
//! A quantum *decrease* leaves that lead oversized, which is pure latency;
//! [`CycleReader`] trims it back by discarding the excess once — a single skip-ahead
//! blip at the moment the graph renegotiates, in exchange for not wearing the old
//! quantum's latency forever.

use crate::node::SharedAudio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// DeepFilterNet's hop size at 48 kHz. Not a tunable.
pub const HOP: usize = 480;

/// Ring capacity, in hops. 320 ms — enough to ride out a scheduling hiccup, small
/// enough that the latency ceiling stays sane if something goes badly wrong.
pub const RING_HOPS: usize = 32;

/// How much processed lead beyond the current quantum is tolerated before
/// [`CycleReader`] trims it, in samples. Two hops: one is the legitimate remainder bound
/// (the deficit at any moment is `samples mod HOP`), the second is slack so the trim
/// only fires on a genuine quantum decrease, never oscillates against normal jitter,
/// and always leaves more than the hysteresis target so trimming cannot re-prime.
const TRIM_SLACK: usize = 2 * HOP;

/// Append samples, dropping the *newest* on overflow and counting what was lost.
///
/// Dropping the newest rather than the oldest is deliberate: a full ring means the
/// consumer has stalled, and the older audio is what is about to be played. Discarding
/// that would leave a gap; discarding the newest just means a shorter delay once the
/// consumer catches up. The count lands in an atomic so the RT side never waits to
/// report it.
pub fn push_or_drop(tx: &mut rtrb::Producer<f32>, samples: &[f32], overruns: &AtomicU64) {
    let take = samples.len().min(tx.slots());
    if let Ok(chunk) = tx.write_chunk_uninit(take) {
        chunk.fill_from_iter(samples[..take].iter().copied());
    }
    if take < samples.len() {
        overruns.fetch_add((samples.len() - take) as u64, Ordering::Relaxed);
    }
}

/// The playback callback's end of the processed ring: fills whole cycles or answers
/// with silence, never partially — and rebuilds a deliberate lead after any shortfall.
///
/// The hysteresis is the load-bearing part. Delivery does not resume the moment the
/// ring merely covers one quantum; after a shortfall it waits for a full
/// `quantum + HOP`. Without that cushion, resuming at the bare minimum leaves the
/// remainder walk (`floor(k·quantum / HOP)` versus `k·quantum`) free to dip below zero
/// a few cycles later — delivered, delivered, pad, in a loop. With it, delivery resumes
/// only when the worker's long-run rate can never be pierced again, so silence is
/// confined to the settling cycles after a stream start or a quantum increase.
pub struct CycleReader {
    rx: rtrb::Consumer<f32>,
    /// Whether we are rebuilding the lead. Starts true: a fresh stream has no lead.
    priming: bool,
    /// The previous cycle's size. A quantum *increase* must trigger priming even when
    /// the old lead still technically covers the new quantum — an entering lead without
    /// the `+ HOP` cushion delivers for a few cycles and then starves, which is the
    /// deliver-deliver-pad oscillation the hysteresis exists to forbid. Observed as a
    /// pad on cycle 3 after a 96 → 1024 renegotiation, exactly the walk predicted.
    last_want: usize,
}

impl CycleReader {
    pub fn new(rx: rtrb::Consumer<f32>) -> Self {
        Self {
            rx,
            priming: true,
            last_want: 0,
        }
    }

    /// Samples currently buffered — the lead. For tests and diagnostics.
    pub fn available(&self) -> usize {
        self.rx.slots()
    }

    /// Fill `out` completely from the ring, or fill it with silence and keep building.
    ///
    /// Returns whether real audio was delivered. After a successful pop, a lead
    /// exceeding the quantum by more than [`TRIM_SLACK`] (a quantum decrease just
    /// happened) is discarded to cap latency — a single skip-ahead blip against
    /// carrying the old quantum's delay forever. The trim leaves more than the
    /// hysteresis target, so trimming can never itself trigger re-priming.
    pub fn pop_cycle(&mut self, out: &mut [f32]) -> bool {
        if out.len() > self.last_want {
            self.priming = true;
        }
        self.last_want = out.len();

        let target = if self.priming {
            out.len() + HOP
        } else {
            out.len()
        };
        if self.rx.slots() < target {
            self.priming = true;
            out.fill(0.0);
            return false;
        }
        self.priming = false;

        if let Ok(chunk) = self.rx.read_chunk(out.len()) {
            let (a, b) = chunk.as_slices();
            out[..a.len()].copy_from_slice(a);
            out[a.len()..].copy_from_slice(b);
            chunk.commit_all();
        }
        let excess = self.rx.slots().saturating_sub(out.len() + TRIM_SLACK);
        if excess > 0
            && let Ok(chunk) = self.rx.read_chunk(excess)
        {
            chunk.commit_all();
        }
        true
    }
}

/// Run `process` over every whole hop waiting in `input`. Returns whether any hop was
/// processed, so the worker knows whether to park.
///
/// `process` receives exactly one hop and writes exactly one hop — non-negotiable, see
/// the module docs on DeepFilterNet's `debug_assert`.
pub fn drain_hops<F>(
    input: &mut rtrb::Consumer<f32>,
    output: &mut rtrb::Producer<f32>,
    process: &mut F,
    overruns: &AtomicU64,
) -> bool
where
    F: FnMut(&[f32; HOP], &mut [f32; HOP]),
{
    let mut inp = [0.0f32; HOP];
    let mut outp = [0.0f32; HOP];
    let mut any = false;
    while input.slots() >= HOP {
        let Ok(chunk) = input.read_chunk(HOP) else {
            break;
        };
        let (a, b) = chunk.as_slices();
        inp[..a.len()].copy_from_slice(a);
        inp[a.len()..].copy_from_slice(b);
        chunk.commit_all();

        outp.fill(0.0);
        process(&inp, &mut outp);
        push_or_drop(output, &outp, overruns);
        any = true;
    }
    any
}

/// The denoise worker: the normal-priority thread the RT callbacks hand their work to.
///
/// Parked rather than polled: the capture callback unparks it after every push (an
/// `unpark` is a futex wake — RT-safe), and the timeout is only a safety net so shutdown
/// and a silently-stopped capture stream cannot leave it parked forever. Inference,
/// config snapshots and their allocations all happen here, where a lock or a slow hop
/// costs latency the rings absorb rather than a missed RT deadline.
pub fn run_worker<F>(
    mut input: rtrb::Consumer<f32>,
    mut output: rtrb::Producer<f32>,
    mut process: F,
    stop: Arc<AtomicBool>,
    shared: Arc<SharedAudio>,
) where
    F: FnMut(&[f32; HOP], &mut [f32; HOP]),
{
    while !stop.load(Ordering::Relaxed) {
        if !drain_hops(&mut input, &mut output, &mut process, &shared.overruns) {
            std::thread::park_timeout(Duration::from_millis(10));
        }
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

    fn rings() -> (
        rtrb::Producer<f32>,
        rtrb::Consumer<f32>,
        rtrb::Producer<f32>,
        CycleReader,
        AtomicU64,
    ) {
        let (cap_tx, cap_rx) = rtrb::RingBuffer::new(RING_HOPS * HOP);
        let (out_tx, out_rx) = rtrb::RingBuffer::new(RING_HOPS * HOP);
        (
            cap_tx,
            cap_rx,
            out_tx,
            CycleReader::new(out_rx),
            AtomicU64::new(0),
        )
    }

    /// One simulated graph cycle in the WORST-CASE event order: capture push, playback
    /// pop, and only then the worker running. Within a real cycle the two callbacks are
    /// microseconds apart on the RT thread, so a normal-priority worker genuinely
    /// cannot land its output in between — a test using the friendly order would pass
    /// against an implementation that chops in production.
    fn cycle(
        cap_tx: &mut rtrb::Producer<f32>,
        cap_rx: &mut rtrb::Consumer<f32>,
        out_tx: &mut rtrb::Producer<f32>,
        reader: &mut CycleReader,
        overruns: &AtomicU64,
        block: &[f32],
        out: &mut [f32],
    ) -> bool {
        push_or_drop(cap_tx, block, overruns);
        let delivered = reader.pop_cycle(out);
        drain_hops(cap_rx, out_tx, &mut |i, o| o.copy_from_slice(i), overruns);
        delivered
    }

    #[test]
    fn the_lead_is_paid_for_at_startup_not_bled_out_every_cycle() {
        // The whole design in one assertion: with the worker always a full cycle late,
        // the lead costs at most two padded cycles at the start (two, not one, because
        // the worker's first cycle yields only floor(Q/HOP) hops — the remainder arrives
        // a cycle later) and then never pads again. 1024 is not a multiple of 480, so
        // 200 cycles walk the remainder through every phase.
        let (mut ct, mut cr, mut ot, mut or_, ov) = rings();
        let block = vec![0.25f32; 1024];
        let mut out = vec![0.0f32; 1024];

        let mut silent = 0;
        for i in 0..200 {
            if !cycle(&mut ct, &mut cr, &mut ot, &mut or_, &ov, &block, &mut out) {
                silent += 1;
                assert!(
                    i < 2,
                    "a silent cycle after the lead is built is a chop: cycle {i}"
                );
            }
        }
        assert!(
            silent <= 2,
            "the lead must cost at most two startup cycles, cost {silent}"
        );
        assert_eq!(
            ov.load(Ordering::Relaxed),
            0,
            "steady state must not overrun"
        );
    }

    #[test]
    fn a_failed_pop_is_pure_silence_and_leaves_the_ring_building() {
        let (_ct, _cr, mut ot, mut or_, ov) = rings();
        push_or_drop(&mut ot, &[1.0f32; 100], &ov);

        let mut out = vec![7.0f32; 1024];
        assert!(!or_.pop_cycle(&mut out), "100 samples cannot fill 1024");
        assert!(
            out.iter().all(|&s| s == 0.0),
            "a shortfall must be silence, not stale buffer contents"
        );
        assert_eq!(
            or_.available(),
            100,
            "the ring must keep accumulating toward the lead, not be part-drained"
        );
    }

    #[test]
    fn survives_a_quantum_that_changes_mid_stream() {
        // PipeWire renegotiates the quantum whenever another client joins the graph. An
        // increase legitimately needs the lead rebuilt (a couple of settling cycles);
        // after those, every block must be delivered again — the hysteresis exists
        // precisely so settling cannot smear into deliver-pad oscillation.
        let (mut ct, mut cr, mut ot, mut or_, ov) = rings();
        for quantum in [1024usize, 512, 2048, 256, 480, 96, 1024] {
            let block = vec![0.1f32; quantum];
            let mut out = vec![0.0f32; quantum];
            for i in 0..30 {
                let delivered = cycle(&mut ct, &mut cr, &mut ot, &mut or_, &ov, &block, &mut out);
                assert!(
                    delivered || i < 2,
                    "only the settling cycles after a quantum change may pad (quantum \
                     {quantum}, cycle {i})"
                );
            }
        }
        assert_eq!(ov.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_quantum_decrease_does_not_leave_its_latency_behind() {
        // The lead built for a 2048 quantum is pure delay once the graph drops to 256.
        // pop_cycle must trim it instead of carrying ~40 ms of stale latency forever.
        let (mut ct, mut cr, mut ot, mut or_, ov) = rings();
        let big = vec![0.1f32; 2048];
        let mut out_big = vec![0.0f32; 2048];
        for _ in 0..10 {
            cycle(&mut ct, &mut cr, &mut ot, &mut or_, &ov, &big, &mut out_big);
        }

        let small = vec![0.1f32; 256];
        let mut out_small = vec![0.0f32; 256];
        for _ in 0..5 {
            cycle(
                &mut ct,
                &mut cr,
                &mut ot,
                &mut or_,
                &ov,
                &small,
                &mut out_small,
            );
        }
        // The bound: the trim leaves at most want + TRIM_SLACK, and up to one more hop
        // can land between the trim and this observation.
        assert!(
            or_.available() <= 256 + TRIM_SLACK + HOP,
            "the lead must be trimmed to the new quantum's scale, found {} samples",
            or_.available()
        );
    }

    #[test]
    fn drain_always_hands_the_denoiser_exactly_one_hop() {
        // Non-negotiable: DeepFilterNet only debug_asserts this, so a release build
        // would silently misbehave rather than fail.
        let (mut ct, mut cr, mut ot, _or, ov) = rings();
        push_or_drop(&mut ct, &vec![1.0f32; 3000], &ov);
        let mut calls = 0;
        drain_hops(
            &mut cr,
            &mut ot,
            &mut |inp, outp| {
                assert_eq!(inp.len(), HOP);
                assert_eq!(outp.len(), HOP);
                calls += 1;
                outp.copy_from_slice(inp);
            },
            &ov,
        );
        assert_eq!(calls, 3000 / HOP, "every whole hop, and no partial one");
    }

    #[test]
    fn a_stalled_worker_produces_overruns_not_a_panic() {
        // If the worker wedges, the RT callback must keep working and the loss must be
        // visible rather than silent.
        let (mut ct, _cr, _ot, _or, ov) = rings();
        for _ in 0..50 {
            push_or_drop(&mut ct, &vec![1.0f32; 1024], &ov);
        }
        assert!(
            ov.load(Ordering::Relaxed) > 0,
            "a stalled consumer must be visible as overruns"
        );
    }

    #[test]
    fn audio_is_preserved_exactly_through_a_passthrough_bridge() {
        // End-to-end sample fidelity: what goes in must come out — delayed by the lead,
        // never reordered, resampled or dropped. The signal starts at 1.0 so the
        // boundary between lead silence and real audio is unambiguous.
        let (mut ct, mut cr, mut ot, mut or_, ov) = rings();
        let signal: Vec<f32> = (0..8192).map(|i| 1.0 + (i as f32 * 0.01).sin()).collect();
        let mut collected = Vec::new();
        let mut out = vec![0.0f32; 1024];

        for chunk in signal.chunks(1024) {
            cycle(&mut ct, &mut cr, &mut ot, &mut or_, &ov, chunk, &mut out);
            collected.extend_from_slice(&out);
        }

        let start = collected
            .iter()
            .position(|&s| s != 0.0)
            .expect("real audio must eventually arrive");
        for (i, (got, want)) in collected[start..].iter().zip(signal.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "sample {i} after the lead changed: {got} != {want}"
            );
        }
    }

    #[test]
    fn the_worker_thread_delivers_audio_and_stops_on_request() {
        // The one threaded test: the worker must move samples end to end and must exit
        // promptly when told, or the daemon hangs on shutdown.
        let (mut ct, cr, ot, mut or_, _ov) = rings();
        let shared = SharedAudio::new();
        let stop = Arc::new(AtomicBool::new(false));
        let worker = std::thread::spawn({
            let stop = stop.clone();
            let shared = shared.clone();
            move || run_worker(cr, ot, |i, o| o.copy_from_slice(i), stop, shared)
        });

        // Two hops: the reader's hysteresis holds back `out.len() + HOP`, so one hop
        // alone would (correctly) never be released.
        push_or_drop(&mut ct, &vec![0.5f32; 2 * HOP], &shared.overruns);
        worker.thread().unpark();

        let mut out = vec![0.0f32; HOP];
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !or_.pop_cycle(&mut out) {
            assert!(
                std::time::Instant::now() < deadline,
                "the worker never delivered a hop"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(out.iter().all(|&s| s == 0.5));

        stop.store(true, Ordering::Relaxed);
        worker.thread().unpark();
        worker.join().expect("the worker must exit cleanly");
    }
}
