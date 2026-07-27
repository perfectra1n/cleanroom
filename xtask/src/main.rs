//! Repository chores that are easier to get right in Rust than in shell.
//!
//! Currently one: the complexity ratchet. `mise run ratchet` locally, a CI job in anger.
//!
//! Argument parsing is a `match` on `args()` rather than clap. Five subcommands do not
//! justify pulling a derive macro into the one tool in this workspace whose build time is
//! on the critical path of every CI run.

mod baseline;
mod metrics;
mod scan;

use std::collections::BTreeMap;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};

use baseline::{Baseline, Exception, Finding, Limits, Metric, Subject};

/// Everything under a limit is invisible to the ratchet, so these decide how long the
/// ledger is — and a ledger nobody reads is a ledger nobody enforces.
///
/// Picked from `ratchet report`'s calibration table rather than from round numbers that
/// sound strict. Over this tree they grandfather roughly a dozen functions, which fits on
/// one screen. For reference, the distribution they were chosen against:
///
/// ```text
/// cognitive   10->18  15->8   20->4   25->3   30->2
/// cyclomatic  15->7   20->4   30->2   40->1   50->1
/// nesting      3->6    4->1    5->1    6->0
/// lines       80->11 100->8  120->5  150->4  200->3
/// ```
const DEFAULT_LIMITS: Limits = Limits {
    // The gate. Nesting-weighted, so it ranks dense code above merely long code.
    cognitive: 15,
    // Loose on purpose: `?` counts here, and this repo propagates errors idiomatically.
    cyclomatic: 25,
    // Three levels is where a reader stops holding the enclosing conditions in their head.
    nesting: 3,
    // Effective lines. The weakest signal here, so it trails rather than leads.
    lines: 100,
    file_lines: 900,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();

    let result = match argv.as_slice() {
        ["ratchet", "check", rest @ ..] => cmd_check(rest),
        ["ratchet", "report", rest @ ..] => cmd_report(rest),
        ["ratchet", "record"] => cmd_record(false),
        ["ratchet", "record", "--init"] => cmd_record(true),
        ["ratchet", "except", key, rest @ ..] => cmd_except(key, rest),
        _ => {
            eprintln!(
                "usage:
  xtask ratchet check            fail on a regression, or on an unrecorded improvement
  xtask ratchet report [--top N] rank every function, worst first; ignores the ledger
  xtask ratchet record           lower stale floors. Cannot raise a ceiling or add one.
  xtask ratchet record --init    write the ledger for the first time
  xtask ratchet except <key> --note '<why>'
                                 take on debt deliberately. The note is mandatory."
            );
            return ExitCode::from(64);
        }
    };

    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("xtask: {e:#}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------------------

fn cmd_report(args: &[&str]) -> Result<ExitCode> {
    let top: usize = match args {
        ["--top", n] => n.parse().context("--top wants a number")?,
        [] => 25,
        _ => bail!("report takes an optional --top N"),
    };

    let root = scan::repo_root()?;
    let census = scan::scan(&root)?;

    let mut fns = census.fns.clone();
    fns.sort_by(|a, b| {
        b.cognitive
            .cmp(&a.cognitive)
            .then(b.nesting.cmp(&a.nesting))
            .then(a.key.cmp(&b.key))
    });

    println!(
        "{} functions across {} files. Worst {} by cognitive complexity:\n",
        census.fns.len(),
        census.files.len(),
        top.min(fns.len())
    );
    println!(
        "{:>5} {:>5} {:>5} {:>6}  function",
        "cog", "cyc", "nest", "lines"
    );
    for f in fns.iter().take(top) {
        println!(
            "{:>5} {:>5} {:>5} {:>6}  {}\n{:>30}{}:{}",
            f.cognitive, f.cyclomatic, f.nesting, f.lines, f.key, "", f.file, f.line
        );
    }

    let totals = census.totals();
    println!(
        "\ntotals: {} effective lines, {} allow-lints, {} unsafe, {} unwrap, {} panic",
        census.files.iter().map(|f| f.lines).sum::<u32>(),
        totals.allow_lints,
        totals.unsafe_blocks,
        totals.unwraps,
        totals.panics
    );

    // How many functions each candidate limit would grandfather. This is the number that
    // decides whether the ledger is readable or theatre, so pick limits from here rather
    // than from a round number that sounds strict.
    println!("\nfunctions over each candidate limit:");
    let candidates: [(Metric, &[u32]); 4] = [
        (Metric::Cognitive, &[10, 15, 20, 25, 30]),
        (Metric::Cyclomatic, &[15, 20, 30, 40, 50]),
        (Metric::Nesting, &[3, 4, 5, 6]),
        (Metric::Lines, &[80, 100, 120, 150, 200]),
    ];
    for (metric, values) in candidates {
        let cells: Vec<String> = values
            .iter()
            .map(|v| {
                let n = census.fns.iter().filter(|f| metric.of(f) > *v).count();
                format!("{v}->{n}")
            })
            .collect();
        println!("  {:<11} {}", metric.name(), cells.join("  "));
    }
    let union = |lim: &Limits| {
        census
            .fns
            .iter()
            .filter(|f| !lim.exceeded_by(f).is_empty())
            .count()
    };
    println!(
        "\n  ledger size with the tool's current defaults: {}",
        union(&DEFAULT_LIMITS)
    );

    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------------------

fn cmd_check(args: &[&str]) -> Result<ExitCode> {
    let allow_stale = args.contains(&"--allow-stale");
    let root = scan::repo_root()?;
    let path = root.join("ratchet.toml");
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "no ledger at {}. Run `xtask ratchet record --init`.",
            path.display()
        )
    })?;
    let base = Baseline::parse(&text)?;

    if !base.meta.syn.is_empty() && base.meta.syn != baseline::SYN_VERSION {
        println!(
            "the analyser changed (syn {} -> {}). Metric values can shift for reasons that\n\
             have nothing to do with your change. Re-record to adopt the new analyser:\n\n    \
             mise run ratchet:record\n",
            base.meta.syn,
            baseline::SYN_VERSION
        );
        return Ok(ExitCode::from(3));
    }

    let census = scan::scan(&root)?;
    let findings = baseline::compare(
        &base,
        &census.subjects(&base.limits),
        &census.totals(),
        &census.unmeasured_set(),
    );

    if findings.is_empty() {
        println!(
            "complexity: {} entries, all holding ({} functions measured).",
            base.over.len(),
            census.fns.len()
        );
        return Ok(ExitCode::SUCCESS);
    }

    let gha = std::env::var_os("GITHUB_ACTIONS").is_some();
    let regressions: Vec<&Finding> = findings.iter().filter(|f| f.is_regression()).collect();
    let bookkeeping: Vec<&Finding> = findings.iter().filter(|f| !f.is_regression()).collect();

    for f in &regressions {
        report_regression(f, gha);
    }
    if !bookkeeping.is_empty() {
        report_bookkeeping(&bookkeeping, gha);
    }

    if !regressions.is_empty() {
        Ok(ExitCode::FAILURE)
    } else if allow_stale {
        println!("(--allow-stale: not failing. CI never passes this flag.)");
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(2))
    }
}

fn report_regression(f: &Finding, gha: bool) {
    match f {
        Finding::Regressed {
            key,
            file,
            line,
            metric,
            was,
            now,
        } => {
            if gha && !file.is_empty() {
                println!(
                    "::error file={file},line={line},title=complexity::{key}: {metric} {was} -> {now}"
                );
            }
            // Totals have no per-entry exception to reach for — there is nothing to
            // grandfather, only debt to not take on. Pointing at `except` here would send
            // someone after a key that does not exist.
            if let Some(label) = key.strip_prefix("totals.") {
                if gha {
                    println!("::error title=complexity::{label} rose from {was} to {now}");
                }
                println!(
                    "\ncomplexity: workspace {label} rose\n\n    {label:<14} {was} -> {now}\n\n\
                     These four totals are debt, and debt only goes down. There is no\n\
                     exception list for them by design — a suppression you can grandfather\n\
                     is a suppression that never gets removed.\n\n  \
                     If you reached for `#[allow(...)]` to get past another complexity\n  \
                     failure: that is exactly the escape hatch this counter exists to close.\n  \
                     Fix the shape instead, or record the debt on the function itself with\n  \
                     `ratchet except`."
                );
                return;
            }
            println!(
                "\ncomplexity: {key} got harder\n  {file}:{line}\n\n    {metric:<11} {was} -> {now}   (ceiling {was})\n\n\
                 The ledger is a ratchet: a recorded number may fall, never rise. If this\n\
                 genuinely had to grow, take the debt on purpose rather than silently —\n\n    \
                 mise run ratchet:except -- {key} --note '<why>'\n\n\
                 `record` will not do it for you. That is the point."
            );
        }
        Finding::Uncovered {
            key,
            file,
            line,
            metric,
            limit,
            now,
        } => {
            if gha {
                println!(
                    "::error file={file},line={line},title=complexity::{key}: {metric} {now} exceeds the limit of {limit}"
                );
            }
            println!(
                "\ncomplexity: {key} is over budget and is not grandfathered\n  {file}:{line}\n\n    \
                 {metric:<11} {now}   (limit {limit})\n\n\
                 Everything already over budget when the ratchet landed is in ratchet.toml.\n\
                 This is not, so it crossed the line in this change. Either simplify it, or\n\
                 record the debt deliberately —\n\n    \
                 mise run ratchet:except -- {key} --note '<why>'"
            );
        }
        _ => {}
    }
}

fn report_bookkeeping(items: &[&Finding], gha: bool) {
    let stale: Vec<&&Finding> = items
        .iter()
        .filter(|f| matches!(f, Finding::Stale { .. }))
        .collect();
    let orphans: Vec<&&Finding> = items
        .iter()
        .filter(|f| matches!(f, Finding::Orphan { .. }))
        .collect();

    if !stale.is_empty() {
        if gha {
            println!(
                "::error title=complexity::{} numbers improved but ratchet.toml still records the old ones",
                stale.len()
            );
        }
        println!(
            "\ncomplexity: {} number(s) improved and the floor was not moved down with them\n",
            stale.len()
        );
        for f in &stale {
            if let Finding::Stale {
                what,
                metric,
                was,
                now,
            } = f
            {
                println!("    {what:<52} {metric:<11} {was} -> {now}");
            }
        }
        println!(
            "\n  This is not a complaint about your change. A floor that can drift back up\n  \
             is not a floor.\n\n    mise run ratchet:record\n\n  \
             Then commit ratchet.toml alongside the change that earned it."
        );
    }

    if !orphans.is_empty() {
        println!(
            "\ncomplexity: {} entr(ies) name a function that no longer exists\n",
            orphans.len()
        );
        for f in &orphans {
            if let Finding::Orphan { key } = f {
                println!("    {key}   (renamed, moved or deleted)");
            }
        }
        println!("\n  Same fix, same reason:\n\n    mise run ratchet:record");
    }
}

// ---------------------------------------------------------------------------------------

fn cmd_record(init: bool) -> Result<ExitCode> {
    let root = scan::repo_root()?;
    let path = root.join("ratchet.toml");
    let census = scan::scan(&root)?;

    let (limits, existing) = if init {
        if path.exists() {
            bail!(
                "{} already exists. `record --init` is for bootstrapping only; plain \
                 `record` updates an existing ledger.",
                path.display()
            );
        }
        (DEFAULT_LIMITS, Vec::new())
    } else {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("no ledger at {}", path.display()))?;
        let base = Baseline::parse(&text)?;
        (base.limits.clone(), base.over.clone())
    };

    let subjects = census.subjects(&limits);
    let by_key: BTreeMap<&str, &Subject> = subjects.iter().map(|s| (s.key.as_str(), s)).collect();
    let mut over: Vec<Exception> = Vec::new();

    if init {
        // Bootstrap. Every note is a TODO the human is expected to replace; `check` refuses
        // an empty one, so the placeholder cannot quietly become permanent without at
        // least being read.
        for s in &subjects {
            let breached = s.over();
            if breached.is_empty() {
                continue;
            }
            let mut e = Exception {
                key: s.key.clone(),
                file: s.file.clone(),
                note: format!(
                    "TODO: say why this is allowed to be this shape, and what it would take \
                     to fix. Recorded at adoption from {}:{}.",
                    s.file, s.line
                ),
                cognitive: None,
                cyclomatic: None,
                nesting: None,
                lines: None,
            };
            for (m, v) in breached {
                e.set(m, Some(v));
            }
            over.push(e);
        }
    } else {
        // Refuse to bless anything. This is the rule the whole design rests on: if `record`
        // could wave a regression through it would be `--bless`, `--bless` becomes muscle
        // memory, and the ledger is a landfill inside six months.
        let mut blocked = Vec::new();
        for e in &existing {
            let Some(s) = by_key.get(e.key.as_str()) else {
                continue;
            };
            for (m, actual, _) in &s.vals {
                if let Some(budget) = e.ceiling(*m)
                    && *actual > budget
                {
                    blocked.push(format!(
                        "    {:<48} {:<11} {budget} -> {actual}",
                        e.key,
                        m.name()
                    ));
                }
            }
        }
        for s in &subjects {
            if existing.iter().any(|e| e.key == s.key) {
                continue;
            }
            for (m, v) in s.over() {
                blocked.push(format!(
                    "    {:<48} {:<11} {v} (limit {})",
                    s.key,
                    m.name(),
                    limits.get_for(m, s.is_file())
                ));
            }
        }
        if !blocked.is_empty() {
            bail!(
                "`record` cannot raise a ceiling or create an entry, and these need one:\n\n{}\n\n\
                 Fix the code, or take the debt on purpose with a written reason:\n\n    \
                 mise run ratchet:except -- <key> --note '<why>'",
                blocked.join("\n")
            );
        }

        for mut e in existing {
            let Some(s) = by_key.get(e.key.as_str()) else {
                println!("dropping {} — it no longer exists", e.key);
                continue;
            };
            e.file = s.file.clone();
            let breached = s.over();
            let mut any = false;
            for m in Metric::ALL {
                if e.ceiling(m).is_none() {
                    continue;
                }
                match breached.iter().find(|(bm, _)| *bm == m) {
                    Some((_, v)) => {
                        e.set(m, Some(*v));
                        any = true;
                    }
                    None => {
                        println!("dropping {}.{} — back under the limit", e.key, m.name());
                        e.set(m, None);
                    }
                }
            }
            if any {
                over.push(e);
            } else {
                println!("dropping {} — no longer over any limit", e.key);
            }
        }
    }

    let unmeasured = census.unmeasured_set();
    let rendered = baseline::render(&limits, &census.totals(), &over, &unmeasured);
    std::fs::write(&path, &rendered).with_context(|| format!("writing {}", path.display()))?;
    println!(
        "wrote {} — {} entries, {} unmeasured macro sites",
        path.display(),
        over.len(),
        unmeasured.len()
    );
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------------------

fn cmd_except(key: &str, args: &[&str]) -> Result<ExitCode> {
    let note = match args {
        ["--note", n] => (*n).to_string(),
        _ => bail!(
            "a note is mandatory: `ratchet except <key> --note '<why>'`.\n\
             An exception with no written reason is indistinguishable from a rubber stamp \
             six months later, and this file is meant to be read."
        ),
    };
    if note.trim().is_empty() {
        bail!("the note is empty");
    }

    let root = scan::repo_root()?;
    let path = root.join("ratchet.toml");
    let text = std::fs::read_to_string(&path)?;
    let base = Baseline::parse(&text)?;
    let census = scan::scan(&root)?;
    let subjects = census.subjects(&base.limits);

    // Accept any unambiguous suffix, so you can type `run_once` rather than the full path.
    let matches: Vec<&Subject> = subjects
        .iter()
        .filter(|s| s.key == key || s.key.ends_with(&format!("::{key}")))
        .collect();
    let s = match matches.as_slice() {
        [one] => *one,
        [] => bail!("no function or file `{key}` in the measured tree"),
        many => bail!(
            "`{key}` is ambiguous:\n  {}",
            many.iter()
                .map(|s| s.key.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        ),
    };

    let breached = s.over();
    if breached.is_empty() {
        bail!("`{}` is within every limit — it needs no exception.", s.key);
    }

    let mut over = base.over.clone();
    let mut entry = over
        .iter()
        .position(|e| e.key == s.key)
        .map(|i| over.remove(i))
        .unwrap_or(Exception {
            key: s.key.clone(),
            file: s.file.clone(),
            note: String::new(),
            cognitive: None,
            cyclomatic: None,
            nesting: None,
            lines: None,
        });
    entry.file = s.file.clone();
    entry.note = note;
    for (m, v) in breached {
        entry.set(m, Some(v));
    }
    over.push(entry);

    let rendered = baseline::render(
        &base.limits,
        &census.totals(),
        &over,
        &census.unmeasured_set(),
    );
    std::fs::write(&path, &rendered)?;
    println!("recorded an exception for {} in {}", s.key, path.display());
    Ok(ExitCode::SUCCESS)
}
