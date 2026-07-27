//! The ledger, and the two directions it can fail in.
//!
//! A high-water mark only fails when a number gets worse. A *ratchet* also fails when a
//! number gets better and nobody wrote it down — because slack that is not recorded is
//! slack the next change spends without anyone noticing they spent it. Both directions
//! resolve with the same one command, and the message says so.
//!
//! # `record` cannot bless anything
//!
//! This is the load-bearing rule of the whole design. `record` may only *lower* a ceiling,
//! drop an entry that no longer applies, and refresh the totals. It cannot raise a ceiling
//! and it cannot create an entry. Taking on debt goes through `except`, which demands a
//! written `--note`.
//!
//! If `record` could bless a regression it would become `--bless`, `--bless` becomes muscle
//! memory inside a fortnight, and by month six the ledger is a landfill that everyone
//! scrolls past. Splitting the two commands is what keeps the file meaningful.
//!
//! # Slack bands
//!
//! Requiring exact equality would mean every edit to `run_once` demands a re-record, which
//! is the fastest possible route to the tool being resented. Each ratcheted number gets an
//! integer band: an improvement inside the band passes quietly, one beyond it asks to be
//! recorded. Regressions are never banded — any increase past the ceiling fails.
//!
//! Small counters (allows, unsafe blocks, unwraps, panics) have a band of 1, which is to
//! say no band at all. Deleting one `unwrap()` should lower the floor permanently, and
//! that is exactly the request.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::metrics::FnMetrics;

/// Bumped only when the file's shape changes incompatibly.
pub const SCHEMA: u32 = 1;

/// Recorded so a dependency bump reads as "the analyser changed" rather than as a mystery
/// regression in code nobody touched. This is the drift vector most tools forget.
pub const SYN_VERSION: &str = "2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Metric {
    Cognitive,
    Cyclomatic,
    Nesting,
    Lines,
}

impl Metric {
    pub const ALL: [Metric; 4] = [
        Metric::Cognitive,
        Metric::Cyclomatic,
        Metric::Nesting,
        Metric::Lines,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Metric::Cognitive => "cognitive",
            Metric::Cyclomatic => "cyclomatic",
            Metric::Nesting => "nesting",
            Metric::Lines => "lines",
        }
    }

    pub fn of(self, m: &FnMetrics) -> u32 {
        match self {
            Metric::Cognitive => m.cognitive,
            Metric::Cyclomatic => m.cyclomatic,
            Metric::Nesting => m.nesting,
            Metric::Lines => m.lines,
        }
    }

    /// `max(absolute, budget * percent / 100)`, integer arithmetic throughout — there is
    /// not a single float anywhere in this tool, so there is nothing to round differently
    /// on a different machine.
    pub fn slack(self, budget: u32) -> u32 {
        let (abs, pct) = match self {
            // Nesting is a single-digit integer. Band it and it would never tighten.
            Metric::Nesting => (1, 0),
            Metric::Cognitive => (3, 15),
            Metric::Cyclomatic => (3, 15),
            Metric::Lines => (5, 15),
        };
        abs.max(budget * pct / 100)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Limits {
    /// The primary gate. Nesting-weighted, so 235 flat lines of `create_shader_module`
    /// score near zero while 95 dense ones score high — see the metrics module for why
    /// that ordering is the whole point.
    pub cognitive: u32,
    /// Deliberately loose. `?` counts here, and idiomatic error handling should not read
    /// as spaghetti.
    pub cyclomatic: u32,
    pub nesting: u32,
    /// Effective (non-blank, non-comment) lines. Loose, because length turned out to be
    /// the weakest signal in this codebase.
    pub lines: u32,
    /// Effective lines per file, production code only.
    pub file_lines: u32,
}

impl Limits {
    pub fn get(&self, m: Metric) -> u32 {
        match m {
            Metric::Cognitive => self.cognitive,
            Metric::Cyclomatic => self.cyclomatic,
            Metric::Nesting => self.nesting,
            Metric::Lines => self.lines,
        }
    }

    /// A file's only metric is length, and it is checked against `file_lines` rather than
    /// the per-function `lines` budget.
    pub fn get_for(&self, m: Metric, is_file: bool) -> u32 {
        if is_file {
            self.file_lines
        } else {
            self.get(m)
        }
    }

    pub fn exceeded_by(&self, m: &FnMetrics) -> Vec<Metric> {
        Metric::ALL
            .into_iter()
            .filter(|k| k.of(m) > self.get(*k))
            .collect()
    }
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Totals {
    pub allow_lints: u32,
    pub unsafe_blocks: u32,
    pub unwraps: u32,
    pub panics: u32,
}

impl Totals {
    /// `(label, value, slack)`. All exact — no slack at all. These are small integers
    /// where every unit is a decision somebody made, and deleting one `unwrap()` should
    /// lower the floor permanently.
    ///
    /// Total line count is deliberately **not** here. Ratcheting it would mean the
    /// codebase may never grow, so the first feature after adoption fails CI as a
    /// "regression". Size is not debt; these four are.
    pub fn rows(&self) -> [(&'static str, u32, u32); 4] {
        [
            ("allow_lints", self.allow_lints, 1),
            ("unsafe_blocks", self.unsafe_blocks, 1),
            ("unwraps", self.unwraps, 1),
            ("panics", self.panics, 1),
        ]
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Exception {
    pub key: String,
    #[serde(default)]
    pub file: String,
    /// Mandatory, and enforced on the way in. An exception with no written reason is
    /// indistinguishable from a rubber stamp six months later.
    pub note: String,
    #[serde(default)]
    pub cognitive: Option<u32>,
    #[serde(default)]
    pub cyclomatic: Option<u32>,
    #[serde(default)]
    pub nesting: Option<u32>,
    #[serde(default)]
    pub lines: Option<u32>,
}

impl Exception {
    pub fn ceiling(&self, m: Metric) -> Option<u32> {
        match m {
            Metric::Cognitive => self.cognitive,
            Metric::Cyclomatic => self.cyclomatic,
            Metric::Nesting => self.nesting,
            Metric::Lines => self.lines,
        }
    }

    pub fn set(&mut self, m: Metric, v: Option<u32>) {
        match m {
            Metric::Cognitive => self.cognitive = v,
            Metric::Cyclomatic => self.cyclomatic = v,
            Metric::Nesting => self.nesting = v,
            Metric::Lines => self.lines = v,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct UnmeasuredEntry {
    pub file: String,
    #[serde(rename = "macro")]
    pub mac: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub schema: u32,
    #[serde(default)]
    pub syn: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Baseline {
    pub meta: Meta,
    pub limits: Limits,
    #[serde(default)]
    pub totals: Totals,
    #[serde(default, rename = "over")]
    pub over: Vec<Exception>,
    #[serde(default, rename = "unmeasured")]
    pub unmeasured: Vec<UnmeasuredEntry>,
}

impl Baseline {
    pub fn parse(text: &str) -> Result<Self> {
        let b: Baseline = toml::from_str(text).context("parsing ratchet.toml")?;
        if b.meta.schema != SCHEMA {
            bail!(
                "ratchet.toml is schema {} but this tool speaks schema {SCHEMA}",
                b.meta.schema
            );
        }
        for e in &b.over {
            if e.note.trim().is_empty() {
                bail!(
                    "the exception for `{}` has an empty note. Every exception needs a \
                     written reason — without one, nobody reading this file in six months \
                     can tell deliberate debt from an accident.",
                    e.key
                );
            }
        }
        Ok(b)
    }

    pub fn exception(&self, key: &str) -> Option<&Exception> {
        self.over.iter().find(|e| e.key == key)
    }
}

// ---------------------------------------------------------------------------------------
// verdicts
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finding {
    /// A recorded ceiling was exceeded.
    Regressed {
        key: String,
        file: String,
        line: u32,
        metric: &'static str,
        was: u32,
        now: u32,
    },
    /// Over a threshold with no exception at all — it crossed the line in this change.
    Uncovered {
        key: String,
        file: String,
        line: u32,
        metric: &'static str,
        limit: u32,
        now: u32,
    },
    /// Improved past the slack band without the floor being moved down with it.
    Stale {
        what: String,
        metric: &'static str,
        was: u32,
        now: u32,
    },
    /// An entry naming a function that no longer exists.
    Orphan { key: String },
}

impl Finding {
    pub fn is_regression(&self) -> bool {
        matches!(self, Finding::Regressed { .. } | Finding::Uncovered { .. })
    }
}

/// One thing the ratchet has an opinion about: a function, or a whole file.
///
/// Files and functions are ratcheted identically — a limit, an optional recorded ceiling,
/// the same three verdicts — so they share one code path rather than two that drift. A
/// file's key is `file:<path>`, which cannot collide with a symbol path.
pub struct Subject {
    pub key: String,
    pub file: String,
    pub line: u32,
    /// `(metric, measured, limit)`.
    pub vals: Vec<(Metric, u32, u32)>,
}

impl Subject {
    pub fn from_fn(f: &FnMetrics, limits: &Limits) -> Self {
        Subject {
            key: f.key.clone(),
            file: f.file.clone(),
            line: f.line,
            vals: Metric::ALL
                .into_iter()
                .map(|m| (m, m.of(f), limits.get(m)))
                .collect(),
        }
    }

    /// A file carries only a length, checked against `file_lines` rather than the
    /// per-function `lines` limit.
    pub fn from_file(path: &str, lines: u32, limits: &Limits) -> Self {
        Subject {
            key: format!("file:{path}"),
            file: path.to_string(),
            line: 1,
            vals: vec![(Metric::Lines, lines, limits.file_lines)],
        }
    }

    pub fn is_file(&self) -> bool {
        self.key.starts_with("file:")
    }

    pub fn over(&self) -> Vec<(Metric, u32)> {
        self.vals
            .iter()
            .filter(|(_, actual, limit)| actual > limit)
            .map(|(m, a, _)| (*m, *a))
            .collect()
    }
}

/// Compare a fresh census against the ledger.
///
/// Order matters for readability: regressions first (they block the change), then
/// uncovered violations, then the bookkeeping. Within each group, sorted, so the output is
/// stable between runs.
pub fn compare(
    base: &Baseline,
    subjects: &[Subject],
    totals: &Totals,
    unmeasured: &BTreeSet<(String, String)>,
) -> Vec<Finding> {
    let mut regressions = Vec::new();
    let mut uncovered = Vec::new();
    let mut stale = Vec::new();
    let mut orphans = Vec::new();

    let live: BTreeSet<&str> = subjects.iter().map(|s| s.key.as_str()).collect();

    for s in subjects {
        let exc = base.exception(&s.key);
        for (metric, actual, limit) in &s.vals {
            let (actual, limit) = (*actual, *limit);
            match exc.and_then(|e| e.ceiling(*metric)) {
                Some(budget) => {
                    if actual > budget {
                        regressions.push(Finding::Regressed {
                            key: s.key.clone(),
                            file: s.file.clone(),
                            line: s.line,
                            metric: metric.name(),
                            was: budget,
                            now: actual,
                        });
                    } else if actual + metric.slack(budget) <= budget {
                        stale.push(Finding::Stale {
                            what: s.key.clone(),
                            metric: metric.name(),
                            was: budget,
                            now: actual,
                        });
                    }
                }
                None if actual > limit => uncovered.push(Finding::Uncovered {
                    key: s.key.clone(),
                    file: s.file.clone(),
                    line: s.line,
                    metric: metric.name(),
                    limit,
                    now: actual,
                }),
                None => {}
            }
        }
    }

    for e in &base.over {
        if !live.contains(e.key.as_str()) {
            orphans.push(Finding::Orphan { key: e.key.clone() });
        }
    }

    for ((label, now, slack), (_, was, _)) in totals.rows().into_iter().zip(base.totals.rows()) {
        if now > was {
            regressions.push(Finding::Regressed {
                key: format!("totals.{label}"),
                file: String::new(),
                line: 0,
                metric: label,
                was,
                now,
            });
        } else if now + slack <= was {
            stale.push(Finding::Stale {
                what: format!("totals.{label}"),
                metric: label,
                was,
                now,
            });
        }
    }

    let recorded: BTreeSet<(String, String)> = base
        .unmeasured
        .iter()
        .map(|u| (u.file.clone(), u.mac.clone()))
        .collect();
    let new_hiding = unmeasured.difference(&recorded).count() as u32;
    let gone = recorded.difference(unmeasured).count() as u32;
    if new_hiding > 0 {
        regressions.push(Finding::Regressed {
            key: "unmeasured".into(),
            file: String::new(),
            line: 0,
            metric: "macro-hidden code",
            was: recorded.len() as u32,
            now: unmeasured.len() as u32,
        });
    } else if gone > 0 {
        stale.push(Finding::Stale {
            what: "unmeasured".into(),
            metric: "macro-hidden code",
            was: recorded.len() as u32,
            now: unmeasured.len() as u32,
        });
    }

    regressions.extend(uncovered);
    regressions.extend(stale);
    regressions.extend(orphans);
    regressions
}

// ---------------------------------------------------------------------------------------
// emitting
// ---------------------------------------------------------------------------------------

const BANNER: &str = "\
# Cleanroom complexity ratchet — the floor only moves down.
#
# Rewritten by `mise run ratchet:record`. Two ways this file fails CI, and they mean
# opposite things:
#
#   a number went UP past its ceiling   -> split the function, or take the debt on
#                                          purpose with `ratchet except <key> --note '<why>'`
#   a number went DOWN, unrecorded      -> `mise run ratchet:record`
#
# `record` can only ever lower a ceiling or drop an entry. It cannot raise one and it
# cannot add one — that is deliberate, and it is what stops this file becoming a landfill
# of blessed regressions.
#
# Nothing under the limits appears here: this is a ledger of exceptions, not a census.
# Cognitive complexity is the gate; length is loose on purpose. Measured on this tree,
# the second-longest function in the workspace (FramePipeline::new, 235 lines) is a flat
# sequence of constructor calls, while doctor::check_gpu is dense at 95 — so ranking by
# length gets them backwards. See xtask/src/metrics.rs for the exact rules.
#
# `lines` counts effective lines: blank and comment-only lines are excluded, and a
# function's doc comment does not count against it. Writing down *why* is free.
";

/// Hand-rolled rather than `toml::to_string`, so the serializer's own version can never
/// reformat the file underneath us and produce a diff nobody asked for.
pub fn render(
    limits: &Limits,
    totals: &Totals,
    over: &[Exception],
    unmeasured: &BTreeSet<(String, String)>,
) -> String {
    let mut s = String::from(BANNER);

    let _ = write!(
        s,
        "\n[meta]\nschema = {SCHEMA}\nsyn = \"{SYN_VERSION}\"\n\n\
         # Ceilings for code that is not grandfathered below.\n[limits]\n\
         cognitive  = {}\ncyclomatic = {}\nnesting    = {}\nlines      = {}\nfile_lines = {}\n\n\
         # Ratcheted totals across production `src/`. These may only fall.\n[totals]\n\
         allow_lints   = {}\nunsafe_blocks = {}\nunwraps       = {}\npanics        = {}\n",
        limits.cognitive,
        limits.cyclomatic,
        limits.nesting,
        limits.lines,
        limits.file_lines,
        totals.allow_lints,
        totals.unsafe_blocks,
        totals.unwraps,
        totals.panics,
    );

    if !over.is_empty() {
        s.push_str(
            "\n# Grandfathered. Each number is that function's personal ceiling.\n\
             # Fix the code and re-record; do not edit a ceiling by hand.\n",
        );
    }
    let mut sorted: Vec<&Exception> = over.iter().collect();
    sorted.sort_by(|a, b| (&a.file, &a.key).cmp(&(&b.file, &b.key)));
    for e in sorted {
        let _ = write!(
            s,
            "\n[[over]]\nkey  = \"{}\"\nfile = \"{}\"\n",
            e.key, e.file
        );
        for m in Metric::ALL {
            if let Some(v) = e.ceiling(m) {
                let _ = writeln!(s, "{:<10} = {v}", m.name());
            }
        }
        let _ = writeln!(s, "note = {}", toml_string(&e.note));
    }

    if !unmeasured.is_empty() {
        s.push_str(
            "\n# Code this tool cannot see into: syn keeps macro bodies as raw tokens.\n\
             # Recorded rather than skipped, so \"not measured\" is a visible number.\n\
             # No line numbers, so this churns only when a new macro appears in a new file.\n",
        );
        for (file, mac) in unmeasured {
            let _ = write!(
                s,
                "\n[[unmeasured]]\nfile  = \"{file}\"\nmacro = \"{mac}\"\n"
            );
        }
    }

    s
}

/// Multi-line notes are emitted as a TOML basic string with escapes, which round-trips
/// through any parser and never depends on how the text happens to be wrapped.
fn toml_string(v: &str) -> String {
    let mut out = String::with_capacity(v.len() + 2);
    out.push('"');
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => {}
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subj(key: &str, cognitive: u32) -> Subject {
        Subject::from_fn(&fnm(key, cognitive), &limits())
    }

    fn fnm(key: &str, cognitive: u32) -> FnMetrics {
        FnMetrics {
            key: key.into(),
            file: "a.rs".into(),
            line: 1,
            cognitive,
            cyclomatic: 1,
            nesting: 0,
            lines: 1,
            params: 0,
        }
    }

    fn limits() -> Limits {
        Limits {
            cognitive: 25,
            cyclomatic: 45,
            nesting: 5,
            lines: 120,
            file_lines: 900,
        }
    }

    fn base(over: Vec<Exception>) -> Baseline {
        Baseline {
            meta: Meta {
                schema: SCHEMA,
                syn: SYN_VERSION.into(),
            },
            limits: limits(),
            totals: Totals::default(),
            over,
            unmeasured: vec![],
        }
    }

    fn exc(key: &str, cognitive: u32) -> Exception {
        Exception {
            key: key.into(),
            file: "a.rs".into(),
            note: "because".into(),
            cognitive: Some(cognitive),
            cyclomatic: None,
            nesting: None,
            lines: None,
        }
    }

    #[test]
    fn growing_past_a_ceiling_is_a_regression() {
        let f = compare(
            &base(vec![exc("k", 100)]),
            &[subj("k", 101)],
            &Totals::default(),
            &BTreeSet::new(),
        );
        assert!(matches!(
            f[0],
            Finding::Regressed {
                now: 101,
                was: 100,
                ..
            }
        ));
    }

    #[test]
    fn a_small_improvement_passes_quietly_a_large_one_asks_to_be_recorded() {
        // budget 100 -> slack max(3, 15) = 15. 90 is inside the band; 85 is not.
        let quiet = compare(
            &base(vec![exc("k", 100)]),
            &[subj("k", 90)],
            &Totals::default(),
            &BTreeSet::new(),
        );
        assert!(quiet.is_empty(), "got {quiet:?}");

        let loud = compare(
            &base(vec![exc("k", 100)]),
            &[subj("k", 85)],
            &Totals::default(),
            &BTreeSet::new(),
        );
        assert!(matches!(loud[0], Finding::Stale { now: 85, .. }));
    }

    #[test]
    fn crossing_a_limit_with_no_exception_is_uncovered_not_a_regression() {
        let f = compare(
            &base(vec![]),
            &[subj("k", 26)],
            &Totals::default(),
            &BTreeSet::new(),
        );
        assert!(matches!(
            f[0],
            Finding::Uncovered {
                limit: 25,
                now: 26,
                ..
            }
        ));
    }

    #[test]
    fn an_entry_for_a_vanished_function_is_an_orphan() {
        let f = compare(
            &base(vec![exc("gone", 100)]),
            &[],
            &Totals::default(),
            &BTreeSet::new(),
        );
        assert!(matches!(&f[0], Finding::Orphan { key } if key == "gone"));
    }

    #[test]
    fn small_counters_have_no_slack_at_all() {
        let recorded = Totals {
            unwraps: 12,
            ..Default::default()
        };
        let now = Totals {
            unwraps: 11,
            ..Default::default()
        };

        let mut b = base(vec![]);
        b.totals = recorded;
        let f = compare(&b, &[], &now, &BTreeSet::new());
        assert!(
            matches!(
                &f[0],
                Finding::Stale {
                    metric: "unwraps",
                    ..
                }
            ),
            "deleting one unwrap must lower the floor permanently, got {f:?}"
        );
    }

    #[test]
    fn a_new_macro_hiding_place_is_a_regression() {
        let mut seen = BTreeSet::new();
        seen.insert(("a.rs".to_string(), "slint::include_modules".to_string()));
        let f = compare(&base(vec![]), &[], &Totals::default(), &seen);
        assert!(matches!(&f[0], Finding::Regressed { key, .. } if key == "unmeasured"));
    }

    #[test]
    fn an_exception_without_a_note_is_refused_at_parse_time() {
        let text = format!(
            "[meta]\nschema = {SCHEMA}\n[limits]\ncognitive=1\ncyclomatic=1\nnesting=1\n\
             lines=1\nfile_lines=1\n[[over]]\nkey=\"k\"\nnote=\"\"\ncognitive=9\n"
        );
        let err = Baseline::parse(&text).unwrap_err().to_string();
        assert!(err.contains("written reason"), "got {err}");
    }

    #[test]
    fn rendering_is_stable_and_round_trips() {
        let over = vec![exc("b::k", 100), exc("a::k", 50)];
        let mut un = BTreeSet::new();
        un.insert(("z.rs".into(), "m".into()));
        let once = render(&limits(), &Totals::default(), &over, &un);
        let twice = render(&limits(), &Totals::default(), &over, &un);
        assert_eq!(once, twice);
        Baseline::parse(&once).expect("what we emit, we can read");
    }

    #[test]
    fn a_note_with_quotes_and_newlines_survives_the_round_trip() {
        let mut e = exc("k", 10);
        e.note = "it \"has\" quotes\nand a newline".into();
        let text = render(&limits(), &Totals::default(), &[e], &BTreeSet::new());
        let parsed = Baseline::parse(&text).unwrap();
        assert_eq!(parsed.over[0].note, "it \"has\" quotes\nand a newline");
    }
}
