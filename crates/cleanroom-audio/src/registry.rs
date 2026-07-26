//! Watching the PipeWire registry: who is listening, and what else could we capture.
//!
//! Two jobs, one listener, because they need the same events and the same main loop.
//!
//! **Releasing the microphone when idle.** `node.passive = true` was tried first and gave a
//! permanently silent virtual mic. The reason is structural: the capture and source streams
//! are two *independent* nodes joined only by a userspace ring buffer, not by a PipeWire
//! link, so nothing else was ever going to drive the graph on our behalf. A passive node
//! waits for a driver that does not exist. The property is not the mechanism — watching our
//! own source node's links is.
//!
//! **Enumerating microphones.** `ListMicrophones` returned an empty vec, so the CLI printed
//! "(none)" and a GUI picker had nothing to show. The registry already carries every
//! `Audio/Source` node; noticing them costs one more match in the same callback.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A microphone the user could choose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    /// PipeWire `node.name`. Stable across restarts, which `object.id` is not, so this is
    /// what goes in the config file.
    pub name: String,
    /// `node.description`, falling back to the name when a node does not set one.
    pub description: String,
}

/// What the registry watcher publishes for the rest of the daemon to read.
#[derive(Debug, Default)]
pub struct RegistryView {
    sources: Mutex<Vec<Source>>,
    /// How many links currently terminate on our source node.
    listeners: Mutex<u32>,
}

impl RegistryView {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Microphones currently present, sorted by description for a stable picker order.
    pub fn sources(&self) -> Vec<Source> {
        self.sources.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Whether anything is currently consuming the virtual microphone.
    pub fn has_listeners(&self) -> bool {
        self.listeners.lock().map(|g| *g > 0).unwrap_or(true)
    }

    pub fn listener_count(&self) -> u32 {
        self.listeners.lock().map(|g| *g).unwrap_or(0)
    }

    /// Called from the PipeWire thread once per tick.
    pub fn publish(&self, sources: Vec<Source>, listeners: u32) {
        if let Ok(mut g) = self.sources.lock() {
            *g = sources;
        }
        if let Ok(mut g) = self.listeners.lock() {
            *g = listeners;
        }
    }
}

/// Tracks registry globals and decides whether the capture stream should be running.
///
/// Kept separate from the PipeWire plumbing so the policy — which is where the subtlety is —
/// can be tested without a running graph.
pub struct LinkTracker {
    /// Our source node's id, once the stream reports one.
    node_id: Option<u32>,
    /// Link global id -> the node ids at each end.
    ///
    /// The endpoints are stored rather than a precomputed "is it ours" flag, because our
    /// own node id arrives *after* some links already exist. Keeping the raw ends means
    /// they are re-evaluated once we know who we are, instead of staying permanently
    /// misfiled as somebody else's.
    links: HashMap<u32, (Option<u32>, Option<u32>)>,
    /// Node global id -> the source it described, for removal.
    nodes: HashMap<u32, Source>,
}

impl Default for LinkTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LinkTracker {
    pub fn new() -> Self {
        Self {
            node_id: None,
            links: HashMap::new(),
            nodes: HashMap::new(),
        }
    }

    /// Record our source node's id, ignoring PipeWire's "not assigned yet" sentinel.
    ///
    /// `Stream::node_id()` returns `SPA_ID_INVALID` (`u32::MAX`) until the stream has
    /// actually been connected and negotiated on the main loop, which has *not* happened
    /// when `connect()` returns. Storing that sentinel as if it were an id means no link
    /// ever matches, the listener count sits at zero forever, and the microphone is
    /// released and never resumed — a silent virtual mic, which is the same symptom
    /// `node.passive = true` produced and just as hard to attribute.
    pub fn set_node_id(&mut self, id: u32) {
        if id == u32::MAX {
            return;
        }
        self.node_id = Some(id);
    }

    /// Whether our own node id is known yet.
    pub fn node_known(&self) -> bool {
        self.node_id.is_some()
    }

    /// Record a link global. `output`/`input` are the node ids at each end.
    ///
    /// Both ends are checked, not just the output. We are a source, so in the normal case
    /// our node is the *output* of a link into somebody's stream — but a loopback or a
    /// monitor can put us on the other side, and a link we fail to recognise is a listener
    /// we would cut off mid-sentence.
    pub fn add_link(&mut self, id: u32, output: Option<u32>, input: Option<u32>) {
        self.links.insert(id, (output, input));
    }

    pub fn remove_global(&mut self, id: u32) {
        self.links.remove(&id);
        self.nodes.remove(&id);
    }

    /// How many links currently touch our node, or `None` while we do not yet know which
    /// node is ours.
    ///
    /// `None` is not zero, and the difference matters: zero means "nobody is listening, the
    /// microphone may be released", where `None` means "no idea", and releasing on no idea
    /// is how you ship a silent microphone.
    pub fn listeners(&self) -> Option<u32> {
        let me = self.node_id?;
        Some(
            self.links
                .values()
                .filter(|(o, i)| *o == Some(me) || *i == Some(me))
                .count() as u32,
        )
    }

    /// Record an `Audio/Source` node, if that is what this global is.
    ///
    /// Our own nodes are excluded: offering the virtual microphone as an input to itself is
    /// a feedback loop, and `is_owned_node` already knows about PipeWire's `.N` suffix for
    /// republished names.
    pub fn add_node(&mut self, id: u32, media_class: &str, name: &str, description: Option<&str>) {
        if !media_class.starts_with("Audio/Source") {
            return;
        }
        if cleanroom_core::node::is_owned_node(name) {
            return;
        }
        self.nodes.insert(
            id,
            Source {
                name: name.to_string(),
                description: description.unwrap_or(name).to_string(),
            },
        );
    }

    /// Current sources, sorted so a picker does not reshuffle itself between polls.
    pub fn sources(&self) -> Vec<Source> {
        let mut v: Vec<Source> = self.nodes.values().cloned().collect();
        v.sort_by(|a, b| a.description.cmp(&b.description).then(a.name.cmp(&b.name)));
        v
    }
}

/// Decides when the capture stream may be released, with hysteresis on the falling edge.
///
/// The rising edge is immediate — somebody just started listening and silence would be the
/// bug. The falling edge waits, because apps drop and remake links constantly during format
/// renegotiation, and cycling the hardware microphone on every transient is worse than
/// simply holding it: a device that takes a moment to resume turns a routine reconnect into
/// clipped audio at the start of someone's sentence.
pub struct IdlePolicy {
    grace: std::time::Duration,
    idle_since: Option<std::time::Instant>,
}

impl IdlePolicy {
    pub fn new(grace: std::time::Duration) -> Self {
        Self {
            grace,
            idle_since: None,
        }
    }

    /// Given the current listener count, should the capture stream be active?
    pub fn should_capture(&mut self, listeners: u32, now: std::time::Instant) -> bool {
        if listeners > 0 {
            self.idle_since = None;
            return true;
        }
        match self.idle_since {
            None => {
                self.idle_since = Some(now);
                true
            }
            Some(since) => now.duration_since(since) < self.grace,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn a_link_on_either_end_of_our_node_counts() {
        let mut t = LinkTracker::new();
        t.set_node_id(42);
        t.add_link(1, Some(42), Some(7)); // we are the output: the normal case
        t.add_link(2, Some(9), Some(42)); // we are the input: monitors and loopbacks
        t.add_link(3, Some(8), Some(9)); // nothing to do with us
        assert_eq!(t.listeners(), Some(2));
    }

    /// Before the stream reports its node id, every link is somebody else's. Counting them
    /// as ours would keep the microphone open forever; counting none of them is right,
    /// because the grace period covers the startup window anyway.
    #[test]
    fn links_seen_before_we_know_our_node_id_are_counted_once_we_do() {
        let mut t = LinkTracker::new();
        // A consumer that links during startup, before the stream has been assigned an id.
        t.add_link(1, Some(42), Some(7));
        assert_eq!(t.listeners(), None, "unknown, not zero");
        assert!(!t.node_known());

        t.set_node_id(42);
        assert_eq!(
            t.listeners(),
            Some(1),
            "a link seen before we knew our id must still count"
        );
    }

    /// The bug that made the virtual microphone silent. `Stream::node_id()` returns
    /// SPA_ID_INVALID until the stream is negotiated on the main loop, which has not
    /// happened when `connect()` returns. Storing it means no link ever matches, the count
    /// sits at zero, and the capture stream is released and never resumed.
    #[test]
    fn the_not_assigned_yet_sentinel_is_not_a_node_id() {
        let mut t = LinkTracker::new();
        t.set_node_id(u32::MAX);
        assert!(!t.node_known(), "SPA_ID_INVALID must not read as an id");
        assert_eq!(t.listeners(), None, "and so the count stays unknown");
    }

    #[test]
    fn removing_a_link_drops_the_listener() {
        let mut t = LinkTracker::new();
        t.set_node_id(42);
        t.add_link(1, Some(42), Some(7));
        assert_eq!(t.listeners(), Some(1));
        t.remove_global(1);
        assert_eq!(t.listeners(), Some(0));
    }

    /// Offering our own virtual microphone as a capture target is a feedback loop, and the
    /// config layer already refuses it — but it should never be offered in the first place.
    #[test]
    fn our_own_nodes_are_never_offered_as_sources() {
        let mut t = LinkTracker::new();
        t.add_node(
            1,
            "Audio/Source",
            cleanroom_core::node::VIRTUAL_MIC_NODE,
            None,
        );
        t.add_node(
            2,
            "Audio/Source",
            "alsa_input.usb-Focusrite",
            Some("Scarlett"),
        );
        let s = t.sources();
        assert_eq!(s.len(), 1, "only the real microphone: {s:?}");
        assert_eq!(s[0].description, "Scarlett");
    }

    #[test]
    fn only_audio_sources_are_offered() {
        let mut t = LinkTracker::new();
        t.add_node(1, "Audio/Sink", "speakers", Some("Speakers"));
        t.add_node(2, "Video/Source", "cam", Some("Camera"));
        t.add_node(3, "Stream/Output/Audio", "firefox", Some("Firefox"));
        assert!(t.sources().is_empty(), "{:?}", t.sources());
    }

    /// A node with no description must still be selectable, or a perfectly good microphone
    /// becomes invisible because of a missing property.
    #[test]
    fn a_source_without_a_description_falls_back_to_its_name() {
        let mut t = LinkTracker::new();
        t.add_node(1, "Audio/Source", "alsa_input.thing", None);
        assert_eq!(t.sources()[0].description, "alsa_input.thing");
    }

    #[test]
    fn a_listener_appearing_resumes_capture_immediately() {
        let mut p = IdlePolicy::new(Duration::from_secs(2));
        let t0 = Instant::now();
        assert!(p.should_capture(0, t0));
        assert!(
            !p.should_capture(0, t0 + Duration::from_secs(3)),
            "released"
        );
        assert!(
            p.should_capture(1, t0 + Duration::from_secs(4)),
            "resumed at once"
        );
    }

    /// The whole point of the grace period: a renegotiating client drops its link for a few
    /// milliseconds, and cycling the hardware microphone through that clips the start of
    /// whatever they say next.
    #[test]
    fn a_brief_gap_in_links_does_not_release_the_microphone() {
        let mut p = IdlePolicy::new(Duration::from_secs(2));
        let t0 = Instant::now();
        p.should_capture(1, t0);
        assert!(p.should_capture(0, t0 + Duration::from_millis(50)));
        assert!(p.should_capture(1, t0 + Duration::from_millis(100)));
        assert!(
            p.should_capture(0, t0 + Duration::from_secs(10)),
            "the clock restarts when the link came back"
        );
    }

    #[test]
    fn sustained_idleness_does_release_it() {
        let mut p = IdlePolicy::new(Duration::from_secs(2));
        let t0 = Instant::now();
        p.should_capture(1, t0);
        assert!(p.should_capture(0, t0 + Duration::from_millis(10)));
        assert!(!p.should_capture(0, t0 + Duration::from_secs(5)));
    }
}
