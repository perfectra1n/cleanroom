//! What "complex" means here, and why it does not mean "long".
//!
//! The obvious metric is lines per function, and on this codebase it is the *worst* of the
//! ones available. Measured before any of this was written:
//!
//! | function | code lines | `if` | `match` | max indent |
//! |---|---|---|---|---|
//! | `FramePipeline::new` | 235 | 4 | 0 | 24 |
//! | `run_once` | 298 | 28 | 9 | 41 |
//! | `check_gpu` | 95 | 6 | 3 | 25 |
//!
//! `FramePipeline::new` is the second-longest function in the workspace and is a flat
//! sequence of `create_shader_module` calls — there is nothing to hold in your head.
//! `check_gpu` is a third its length and genuinely dense. A length gate ranks them the
//! wrong way round, so **cognitive complexity leads and length follows**. Length keeps a
//! loose ceiling because a 500-line function is a problem whatever its shape, but it is
//! not what the tool argues about first.
//!
//! # The rules, so a reviewer can reproduce a number by eye
//!
//! **Cognitive** — `1 + current nesting` for `if`, `if let`, `match`, `while`, `for`,
//! `loop` and `let … else`; a flat `+1` for `else`, for each `else if` rung, for each
//! guarded match arm, for each labelled `break`/`continue`, and for each *sequence* of
//! `&&`/`||` (so `a && b && c` is 1, and `a && b || c` is 2). Closures increase nesting but
//! cost nothing themselves. `matches!` costs 1 — it is a match wearing a macro's clothes.
//!
//! **A `match` costs 1 regardless of arm count.** This is the single most important
//! Rust-specific rule here. `cleanroom-ctl`'s 10-arm clap dispatch is a lookup table, not
//! a thicket; charging it 10 would make the tool's worst offender a function nobody has
//! ever had trouble reading.
//!
//! **`?` costs nothing cognitively.** It is counted in cyclomatic, where a path counter
//! belongs, and surfaced separately so a high cyclomatic can be explained at a glance.
//! Charging it would be actively harmful in this repository: `run_frame` in
//! `cleanroom-matting` is eight `.map_err(…)?` chains and is trivially readable, and the
//! cheapest way to lower a `?`-taxed score is to write `.unwrap()` instead — in a project
//! whose first design commitment is that nothing fails silently.
//!
//! **Cyclomatic** is the literal path count: `1 +` each `if`/`else if`, each match arm past
//! the first, each guard, each loop, each `&&`/`||` operator, and each `?`. Deliberately
//! the noisier number, so it gets a loose ceiling and exists mostly to be reported.
//!
//! # Length is measured from `fn`, not from the first attribute
//!
//! A function's doc comment does not count against it. This repo's rationale-dense
//! comments are its best feature and a naive line count taxes exactly them; you can write
//! forty lines of *why* above a function for free. Comments *inside* a body still count,
//! which is right — 300 lines is 300 lines to scroll no matter what is in them. Blank and
//! comment-only lines within the body are excluded too ("effective lines"): measured on
//! this tree, effective runs ≈0.62× raw, so a threshold set against raw numbers would
//! silently be half as strict as it looks.
//!
//! Block comments are not recognised, only `//`. There is exactly one `/*` in the
//! workspace and it does not begin a line; handling them properly means tracking string
//! literals, and getting *that* subtly wrong is how a scanner reports a plausible number
//! for the wrong reason.

use std::collections::BTreeMap;

use proc_macro2::TokenTree;
use syn::visit::{self, Visit};
use syn::{
    Attribute, BinOp, Block, Expr, ExprIf, FnArg, ImplItemFn, Item, ItemFn, ItemImpl, ItemMod,
    ItemTrait, Signature, Stmt, TraitItemFn, UseTree,
};

/// Everything the ratchet knows about one function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnMetrics {
    /// Stable identity: `crate::module::Type::method`. Not a line number — line numbers
    /// churn on every edit above the function, a symbol path does not.
    pub key: String,
    pub file: String,
    pub line: u32,
    pub cognitive: u32,
    pub cyclomatic: u32,
    pub nesting: u32,
    /// Effective (non-blank, non-comment) lines from `fn` to the closing brace.
    pub lines: u32,
    /// `self` excluded.
    pub params: u32,
}

/// Per-file counters. These ratchet as totals rather than per-function, because the
/// question they answer is "how much of this is there" and not "which function".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileMetrics {
    pub path: String,
    pub lines: u32,
    pub unwraps: u32,
    pub panics: u32,
    pub unsafe_blocks: u32,
    pub allow_lints: u32,
}

/// Code this tool cannot see into, recorded rather than skipped.
///
/// `syn` keeps a macro invocation's contents as an unparsed token stream, so
/// `slint::include_modules!()` and `zbus::proxy` expand to code no visitor here will ever
/// walk. Silently scoring that as zero would be the exact "quietly lands on the CPU"
/// failure the README says this project exists to avoid, so it goes in the baseline and is
/// ratcheted: introduce a new kind of hiding place and CI says so.
///
/// No line number, deliberately — keyed on `(file, macro)` it churns only when a new macro
/// appears in a new file, not when code moves around.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Unmeasured {
    pub file: String,
    pub mac: String,
}

/// Macros known to hide no control flow. They are not laid in the unmeasured ledger,
/// which would otherwise fill with `format!` and drown the entries that matter.
const TRANSPARENT_MACROS: &[&str] = &[
    "format",
    "print",
    "println",
    "eprint",
    "eprintln",
    "write",
    "writeln",
    "vec",
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
    "panic",
    "todo",
    "unimplemented",
    "unreachable",
    "matches",
    "trace",
    "debug",
    "info",
    "warn",
    "error",
    "include_str",
    "include_bytes",
    "concat",
    "stringify",
    "env",
];

const PANIC_MACROS: &[&str] = &["panic", "todo", "unimplemented"];

// ---------------------------------------------------------------------------------------
// source lines
// ---------------------------------------------------------------------------------------

/// Per-line "is this code?" bitmap, computed once per file.
pub struct Source {
    is_code: Vec<bool>,
}

impl Source {
    pub fn new(text: &str) -> Self {
        let is_code = text
            .lines()
            .map(|l| {
                let t = l.trim();
                !t.is_empty() && !t.starts_with("//")
            })
            .collect();
        Self { is_code }
    }

    pub fn total(&self) -> u32 {
        self.is_code.iter().filter(|c| **c).count() as u32
    }

    /// Effective code lines outside every excluded range — used for the whole-file count,
    /// where `#[cfg(test)]` modules must not inflate the number.
    ///
    /// Without this, `cleanroom-gpu/src/frame.rs` measures 1280 rather than its true
    /// production 909, because 537 of its lines are an inline test module. The file budget
    /// would then be enforcing a number that is 40% test code — which is the same mistake
    /// as counting `.unwrap()` in tests.
    pub fn effective_outside(&self, excluded: &[(usize, usize)]) -> u32 {
        (1..=self.is_code.len())
            .filter(|n| self.is_code[n - 1])
            .filter(|n| !excluded.iter().any(|(a, b)| n >= a && n <= b))
            .count() as u32
    }

    /// Effective lines in the inclusive 1-based range.
    fn effective(&self, start: usize, end: usize) -> u32 {
        if start == 0 || start > end {
            return 0;
        }
        let lo = start - 1;
        let hi = end.min(self.is_code.len());
        self.is_code[lo..hi].iter().filter(|c| **c).count() as u32
    }
}

// ---------------------------------------------------------------------------------------
// attribute helpers
// ---------------------------------------------------------------------------------------

/// True for `#[cfg(test)]`, `#[cfg(all(test, unix))]` and so on — but **false** for
/// `#[cfg(not(test))]`, which marks production code.
///
/// A substring search for the ident `test` gets `not(test)` backwards and silently drops
/// production code out of the debt count. That is the same class of mistake
/// `packaging_invariants.rs` records for `media.class`: a naive match that fails on a
/// correct file, and fails quietly.
fn is_cfg_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("cfg")
            && a.parse_args::<syn::Meta>()
                .map(|m| meta_mentions_test(&m, false))
                .unwrap_or(false)
    })
}

fn meta_mentions_test(meta: &syn::Meta, negated: bool) -> bool {
    match meta {
        syn::Meta::Path(p) => !negated && p.is_ident("test"),
        syn::Meta::List(l) => {
            let negate_here = l.path.is_ident("not");
            let inner_negated = negated ^ negate_here;
            l.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            .map(|items| items.iter().any(|m| meta_mentions_test(m, inner_negated)))
            .unwrap_or(false)
        }
        syn::Meta::NameValue(_) => false,
    }
}

fn is_test_fn(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        let segs: Vec<String> = a
            .path()
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();
        matches!(segs.last().map(String::as_str), Some("test" | "bench"))
    })
}

/// One count per *lint named*, so `#[allow(dead_code, unused)]` is 2. `#[expect(...)]` is
/// counted too: it is the better habit, but it is still a suppression and the ratchet
/// should notice it arriving.
fn count_allows(attrs: &[Attribute]) -> u32 {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("allow") || a.path().is_ident("expect"))
        .map(|a| {
            a.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            .map(|items| items.len() as u32)
            .unwrap_or(1)
        })
        .sum()
}

// ---------------------------------------------------------------------------------------
// the file-level walker
// ---------------------------------------------------------------------------------------

pub struct FileWalker<'a> {
    src: &'a Source,
    path: String,
    scope: Vec<String>,
    pub fns: Vec<FnMetrics>,
    pub file: FileMetrics,
    pub unmeasured: Vec<Unmeasured>,
    /// Line ranges that are not production code, subtracted from the file's own length.
    pub excluded: Vec<(usize, usize)>,
    seen_keys: BTreeMap<String, u32>,
}

impl<'a> FileWalker<'a> {
    pub fn new(src: &'a Source, path: String, crate_ident: &str, module: &str) -> Self {
        let mut scope = vec![crate_ident.to_string()];
        if !module.is_empty() {
            scope.extend(module.split("::").map(str::to_string));
        }
        Self {
            src,
            file: FileMetrics {
                path: path.clone(),
                lines: src.total(),
                ..Default::default()
            },
            path,
            scope,
            fns: Vec::new(),
            unmeasured: Vec::new(),
            excluded: Vec::new(),
            seen_keys: BTreeMap::new(),
        }
    }

    fn key_for(&mut self, name: &str) -> String {
        let base = format!("{}::{name}", self.scope.join("::"));
        // Same name, same scope, different `#[cfg]`: syn evaluates no cfg, so both are
        // parsed. Rare, and a silent merge would track the wrong function.
        let n = self.seen_keys.entry(base.clone()).or_insert(0);
        *n += 1;
        if *n == 1 { base } else { format!("{base}#{n}") }
    }

    fn measure(&mut self, attrs: &[Attribute], sig: &Signature, block: &Block) {
        self.file.allow_lints += count_allows(attrs);

        let mut body = Body::new(&self.path);
        body.visit_block(block);

        self.file.unwraps += body.unwraps;
        self.file.panics += body.panics;
        self.file.unsafe_blocks += body.unsafe_blocks;
        self.file.allow_lints += body.allow_lints;
        self.unmeasured.append(&mut body.unmeasured);

        let start = sig.fn_token.span.start().line;
        let end = block.brace_token.span.close().end().line;
        let key = self.key_for(&sig.ident.to_string());

        self.fns.push(FnMetrics {
            key,
            file: self.path.clone(),
            line: start as u32,
            cognitive: body.cognitive,
            cyclomatic: body.cyclomatic,
            nesting: body.max_nesting,
            lines: self.src.effective(start, end),
            params: sig
                .inputs
                .iter()
                .filter(|a| matches!(a, FnArg::Typed(_)))
                .count() as u32,
        });

        // A `fn` defined inside a `fn` gets its own entry rather than being folded into
        // its parent's score. `Body` stops at item boundaries, so the two halves partition
        // the tree exactly once.
        self.walk_nested_items(block);
    }

    fn walk_nested_items(&mut self, block: &Block) {
        for stmt in &block.stmts {
            if let Stmt::Item(item) = stmt {
                self.visit_item(item);
            }
        }
    }
}

impl<'ast> Visit<'ast> for FileWalker<'_> {
    /// File-level *inner* attributes — `#![allow(dead_code)]` at the top of
    /// `cleanroomd/src/state.rs` is the only broad suppression in the workspace, and it
    /// lives here rather than on any item. Counting only outer attributes reports zero
    /// suppressions for a repo that has one, which is worse than not counting at all.
    fn visit_file(&mut self, f: &'ast syn::File) {
        self.file.allow_lints += count_allows(&f.attrs);
        visit::visit_file(self, f);
    }

    fn visit_item_mod(&mut self, m: &'ast ItemMod) {
        if is_cfg_test(&m.attrs) {
            // The whole subtree, gone — and its lines come off the file's own length too.
            if let Some((brace, _)) = &m.content {
                self.excluded
                    .push((m.mod_token.span.start().line, brace.span.close().end().line));
            }
            return;
        }
        self.file.allow_lints += count_allows(&m.attrs);
        self.scope.push(m.ident.to_string());
        visit::visit_item_mod(self, m);
        self.scope.pop();
    }

    fn visit_item_impl(&mut self, i: &'ast ItemImpl) {
        if is_cfg_test(&i.attrs) {
            return;
        }
        // `unsafe impl` is real debt. `unsafe extern "C" { }` is not — edition 2024 makes
        // that keyword mandatory, and counting it would record debt the language forces on
        // you. Foreign mods are a different syn node, so they never reach here.
        if i.unsafety.is_some() {
            self.file.unsafe_blocks += 1;
        }
        self.file.allow_lints += count_allows(&i.attrs);
        self.scope.push(impl_scope_name(i));
        visit::visit_item_impl(self, i);
        self.scope.pop();
    }

    fn visit_item_trait(&mut self, t: &'ast ItemTrait) {
        if is_cfg_test(&t.attrs) {
            return;
        }
        self.file.allow_lints += count_allows(&t.attrs);
        self.scope.push(t.ident.to_string());
        visit::visit_item_trait(self, t);
        self.scope.pop();
    }

    fn visit_item_fn(&mut self, f: &'ast ItemFn) {
        if is_cfg_test(&f.attrs) || is_test_fn(&f.attrs) {
            self.excluded.push((
                f.sig.fn_token.span.start().line,
                f.block.brace_token.span.close().end().line,
            ));
            return;
        }
        if f.sig.unsafety.is_some() {
            self.file.unsafe_blocks += 1;
        }
        self.measure(&f.attrs, &f.sig, &f.block);
    }

    fn visit_impl_item_fn(&mut self, f: &'ast ImplItemFn) {
        if is_cfg_test(&f.attrs) || is_test_fn(&f.attrs) {
            self.excluded.push((
                f.sig.fn_token.span.start().line,
                f.block.brace_token.span.close().end().line,
            ));
            return;
        }
        if f.sig.unsafety.is_some() {
            self.file.unsafe_blocks += 1;
        }
        self.measure(&f.attrs, &f.sig, &f.block);
    }

    fn visit_trait_item_fn(&mut self, f: &'ast TraitItemFn) {
        if is_cfg_test(&f.attrs) || is_test_fn(&f.attrs) {
            return;
        }
        if let Some(block) = &f.default {
            self.measure(&f.attrs, &f.sig, block);
        }
    }

    fn visit_item(&mut self, i: &'ast Item) {
        // Attributes on items the specific hooks above do not cover (statics, consts,
        // structs, enums) still carry suppressions worth counting.
        if let Item::Struct(_) | Item::Enum(_) | Item::Const(_) | Item::Static(_) = i
            && let Some(attrs) = item_attrs(i)
        {
            if is_cfg_test(attrs) {
                return;
            }
            self.file.allow_lints += count_allows(attrs);
        }
        visit::visit_item(self, i);
    }

    fn visit_use_tree(&mut self, _: &'ast UseTree) {
        // Nothing to measure, and descending wastes time on every file.
    }

    /// Item-position macros — `slint::include_modules!();` sits at the top of
    /// `cleanroom-gui/src/main.rs` and expands to the entire generated UI binding. Function
    /// bodies are walked by `Body`, so this only ever sees the ones outside them.
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        let name = m
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        if !TRANSPARENT_MACROS.contains(&name.as_str()) {
            self.unmeasured.push(Unmeasured {
                file: self.path.clone(),
                mac: path_string(&m.path),
            });
        }
    }
}

fn item_attrs(i: &Item) -> Option<&[Attribute]> {
    Some(match i {
        Item::Struct(s) => &s.attrs,
        Item::Enum(e) => &e.attrs,
        Item::Const(c) => &c.attrs,
        Item::Static(s) => &s.attrs,
        _ => return None,
    })
}

/// `Foo` for an inherent impl, `<Foo as Bar>` for a trait impl.
///
/// The disambiguation is not optional: this workspace has five `fn fmt` and three
/// `fn drop`, and without the trait in the key they collide into one entry.
fn impl_scope_name(i: &ItemImpl) -> String {
    let ty = type_name(&i.self_ty);
    match &i.trait_ {
        Some((_, path, _)) => {
            let tr = path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_else(|| "?".into());
            format!("<{ty} as {tr}>")
        }
        None => ty,
    }
}

fn type_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p) => p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_else(|| "?".into()),
        syn::Type::Reference(r) => type_name(&r.elem),
        _ => "?".into(),
    }
}

// ---------------------------------------------------------------------------------------
// the function-body walker
// ---------------------------------------------------------------------------------------

struct Body {
    file: String,
    depth: u32,
    max_nesting: u32,
    cognitive: u32,
    cyclomatic: u32,
    unwraps: u32,
    panics: u32,
    unsafe_blocks: u32,
    allow_lints: u32,
    unmeasured: Vec<Unmeasured>,
    /// Which logical operator we are currently inside, so `a && b && c` charges once but
    /// `a && b || c` charges twice.
    logical_run: Option<bool>, // Some(true) = &&, Some(false) = ||
}

impl Body {
    fn new(file: &str) -> Self {
        Self {
            file: file.to_string(),
            depth: 0,
            max_nesting: 0,
            cognitive: 0,
            cyclomatic: 1,
            unwraps: 0,
            panics: 0,
            unsafe_blocks: 0,
            allow_lints: 0,
            unmeasured: Vec::new(),
            logical_run: None,
        }
    }

    /// The one place nesting is entered, so the cognitive weight and the depth high-water
    /// mark cannot drift apart.
    fn nested(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        self.max_nesting = self.max_nesting.max(self.depth);
        f(self);
        self.depth -= 1;
    }

    /// `else if` is a ladder rung, not a new structure.
    ///
    /// syn represents `if a {} else if b {} else if c {}` as an `Expr::If` whose
    /// `else_branch` is another `Expr::If`, three deep. A visitor that just recurses
    /// charges the third rung `1 + 2` and reports nesting depth 3, so every dispatch chain
    /// in `doctor.rs` and `format.rs` scores as though it were deeply nested. It is not:
    /// a reader scans a ladder, they do not descend it.
    fn walk_if(&mut self, e: &ExprIf, is_rung: bool) {
        self.cognitive += if is_rung { 1 } else { 1 + self.depth };
        self.cyclomatic += 1;

        // The condition is evaluated at the *current* depth: `&&`/`||` inside it are flat.
        let saved = self.logical_run.take();
        self.visit_expr(&e.cond);
        self.logical_run = saved;

        self.nested(|s| s.visit_block(&e.then_branch));

        match e.else_branch.as_ref().map(|(_, b)| b.as_ref()) {
            Some(Expr::If(inner)) => self.walk_if(inner, true),
            Some(Expr::Block(b)) => {
                self.cognitive += 1;
                self.nested(|s| s.visit_block(&b.block));
            }
            _ => {}
        }
    }

    fn scan_macro_tokens(&mut self, tokens: proc_macro2::TokenStream) {
        // `tracing::warn!("{}", x.unwrap())` would otherwise be a free hiding place for
        // panic debt: syn keeps macro contents as raw tokens, so no visitor reaches it.
        // Still immune to the false positive that sinks a regex — `format!("unwrap")` is a
        // Literal, not an Ident.
        let mut prev_dot = false;
        for tt in tokens {
            match tt {
                TokenTree::Punct(p) => prev_dot = p.as_char() == '.',
                TokenTree::Ident(i) => {
                    if prev_dot {
                        match i.to_string().as_str() {
                            "unwrap" | "unwrap_unchecked" => self.unwraps += 1,
                            _ => {}
                        }
                    }
                    prev_dot = false;
                }
                TokenTree::Group(g) => {
                    self.scan_macro_tokens(g.stream());
                    prev_dot = false;
                }
                TokenTree::Literal(_) => prev_dot = false,
            }
        }
    }

    fn handle_macro(&mut self, mac: &syn::Macro) {
        let name = mac
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();

        if PANIC_MACROS.contains(&name.as_str()) {
            self.panics += 1;
        }
        // `matches!(x, P)` is a match wearing a macro's clothes.
        if name == "matches" {
            self.cognitive += 1 + self.depth;
            self.cyclomatic += 1;
        }
        if !TRANSPARENT_MACROS.contains(&name.as_str()) {
            self.unmeasured.push(Unmeasured {
                file: self.file.clone(),
                mac: path_string(&mac.path),
            });
        }
        self.scan_macro_tokens(mac.tokens.clone());
    }
}

fn path_string(p: &syn::Path) -> String {
    p.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

impl<'ast> Visit<'ast> for Body {
    // Nested items get their own entry; stop here so nothing is counted twice.
    fn visit_item(&mut self, _: &'ast Item) {}

    fn visit_stmt(&mut self, s: &'ast Stmt) {
        if let Stmt::Local(local) = s {
            self.allow_lints += count_allows(&local.attrs);
            if let Some(init) = &local.init {
                let saved = self.logical_run.take();
                self.visit_expr(&init.expr);
                self.logical_run = saved;
                // `let Some(x) = … else { return }` is one flow break. Its else block does
                // not charge again — the `let else` already did.
                if let Some(div) = &init.diverge {
                    self.cognitive += 1 + self.depth;
                    self.cyclomatic += 1;
                    self.nested(|b| b.visit_expr(&div.1));
                }
            }
            return;
        }
        visit::visit_stmt(self, s);
    }

    fn visit_expr(&mut self, e: &'ast Expr) {
        match e {
            Expr::If(i) => {
                self.walk_if(i, false);
                return;
            }
            Expr::Match(m) => {
                // One increment for the whole match, not one per arm. A 10-arm dispatch on
                // an enum is a lookup table.
                self.cognitive += 1 + self.depth;
                self.cyclomatic += (m.arms.len().max(1) - 1) as u32;

                let saved = self.logical_run.take();
                self.visit_expr(&m.expr);
                self.logical_run = saved;

                for arm in &m.arms {
                    if let Some((_, guard)) = &arm.guard {
                        // A guard is a genuine second condition.
                        self.cognitive += 1;
                        self.cyclomatic += 1;
                        let saved = self.logical_run.take();
                        self.visit_expr(guard);
                        self.logical_run = saved;
                    }
                    self.nested(|b| b.visit_expr(&arm.body));
                }
                return;
            }
            Expr::While(w) => {
                self.cognitive += 1 + self.depth;
                self.cyclomatic += 1;
                let saved = self.logical_run.take();
                self.visit_expr(&w.cond);
                self.logical_run = saved;
                self.nested(|b| b.visit_block(&w.body));
                return;
            }
            Expr::ForLoop(f) => {
                self.cognitive += 1 + self.depth;
                self.cyclomatic += 1;
                self.visit_expr(&f.expr);
                self.nested(|b| b.visit_block(&f.body));
                return;
            }
            Expr::Loop(l) => {
                self.cognitive += 1 + self.depth;
                self.cyclomatic += 1;
                self.nested(|b| b.visit_block(&l.body));
                return;
            }
            Expr::Binary(b) => {
                let this = match b.op {
                    BinOp::And(_) => Some(true),
                    BinOp::Or(_) => Some(false),
                    _ => None,
                };
                if let Some(kind) = this {
                    self.cyclomatic += 1;
                    if self.logical_run != Some(kind) {
                        self.cognitive += 1;
                    }
                    let saved = self.logical_run;
                    self.logical_run = Some(kind);
                    self.visit_expr(&b.left);
                    self.visit_expr(&b.right);
                    self.logical_run = saved;
                    return;
                }
            }
            Expr::Closure(c) => {
                // Costs nothing itself — `.map(|x| x + 1)` should be free — but a closure
                // with a block body is somewhere the reader has to descend into.
                if matches!(c.body.as_ref(), Expr::Block(_) | Expr::Async(_)) {
                    self.nested(|b| b.visit_expr(&c.body));
                } else {
                    self.visit_expr(&c.body);
                }
                return;
            }
            Expr::Unsafe(u) => {
                self.unsafe_blocks += 1;
                visit::visit_block(self, &u.block);
                return;
            }
            Expr::Try(_) => {
                // Cyclomatic only. See the module docs for why `?` costs no cognitive load.
                self.cyclomatic += 1;
            }
            Expr::Break(b) if b.label.is_some() => self.cognitive += 1,
            Expr::Continue(c) if c.label.is_some() => self.cognitive += 1,
            Expr::MethodCall(m) => match m.method.to_string().as_str() {
                "unwrap" | "unwrap_unchecked" => self.unwraps += 1,
                _ => {}
            },
            _ => {}
        }
        visit::visit_expr(self, e);
    }

    /// Every macro position routes here — `Expr::Macro`, `Stmt::Macro` and `Item::Macro`
    /// alike.
    ///
    /// Hooking `Expr::Macro` alone is not enough and fails quietly: a macro used as a
    /// statement (`tracing::warn!("{}", x.unwrap());`, which is most of them) is a
    /// `Stmt::Macro`, so an expression-only hook walks straight past it and the panic debt
    /// inside is never counted.
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        self.handle_macro(m);
    }

    fn visit_block(&mut self, b: &'ast Block) {
        // A logical run does not survive a block boundary.
        let saved = self.logical_run.take();
        visit::visit_block(self, b);
        self.logical_run = saved;
    }
}

// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn measure(src: &str) -> FnMetrics {
        let source = Source::new(src);
        let file = syn::parse_file(src).expect("fixture parses");
        let mut w = FileWalker::new(&source, "t.rs".into(), "t", "");
        w.visit_file(&file);
        assert_eq!(w.fns.len(), 1, "fixture should define exactly one function");
        w.fns.pop().unwrap()
    }

    fn walk(src: &str) -> FileWalker<'static> {
        // Leak the Source so the walker can outlive this frame; test-only.
        let source: &'static Source = Box::leak(Box::new(Source::new(src)));
        let file = syn::parse_file(src).expect("fixture parses");
        let mut w = FileWalker::new(source, "t.rs".into(), "t", "");
        w.visit_file(&file);
        w
    }

    /// Without proc-macro2's `span-locations` feature every `Span::start()` returns line 0.
    /// There is no compile error and no warning: the whole tool just reports every function
    /// as one line long, at the top of its file. This test is the only thing standing
    /// between a dependency tweak and a silently useless baseline.
    #[test]
    fn spans_carry_real_line_numbers() {
        let f: syn::File = syn::parse_str("fn a() {}\n\nfn b() {\n}\n").unwrap();
        let syn::Item::Fn(b) = &f.items[1] else {
            panic!("expected fn b")
        };
        assert_eq!(
            b.sig.fn_token.span.start().line,
            3,
            "proc-macro2's span-locations feature is off — see this test's doc comment"
        );
    }

    #[test]
    fn a_wide_flat_match_is_not_complex() {
        let arms: String = (0..10).map(|i| format!("    V{i} => {i},\n")).collect();
        let m = measure(&format!(
            "fn f(x: E) -> u32 {{\n  match x {{\n{arms}  }}\n}}\n"
        ));
        assert_eq!(m.cognitive, 1, "a 10-arm dispatch is a lookup table");
        assert_eq!(m.cyclomatic, 10, "cyclomatic still counts the paths");
    }

    #[test]
    fn an_else_if_ladder_does_not_nest() {
        let m = measure(
            "fn f(a: bool, b: bool, c: bool) -> u32 {
                 if a { 1 } else if b { 2 } else if c { 3 } else { 4 }
             }",
        );
        // 3 rungs (1 + 1 + 1) + 1 else = 4, and the ladder is flat.
        assert_eq!(m.cognitive, 4);
        assert_eq!(m.nesting, 1, "a ladder is scanned, not descended");
    }

    #[test]
    fn nested_ifs_do_cost_more_than_a_ladder() {
        let m = measure(
            "fn f(a: bool, b: bool, c: bool) -> u32 {
                 if a { if b { if c { 3 } else { 2 } } else { 1 } } else { 0 }
             }",
        );
        assert!(
            m.cognitive > 6,
            "nesting must outweigh a flat ladder, got {}",
            m.cognitive
        );
        assert_eq!(m.nesting, 3);
    }

    #[test]
    fn doc_comments_above_a_function_are_free() {
        let doc: String = (0..20).map(|i| format!("/// line {i}\n")).collect();
        let m = measure(&format!("{doc}fn f() -> u32 {{\n    1\n}}\n"));
        assert_eq!(
            m.lines, 3,
            "fn, body, brace — the 20 doc lines are not taxed"
        );
    }

    #[test]
    fn comments_and_blanks_inside_a_body_are_not_counted() {
        let m = measure(
            "fn f() -> u32 {
    // why this is one
\n
    1
}",
        );
        assert_eq!(m.lines, 3);
    }

    #[test]
    fn the_question_mark_costs_no_cognitive_load() {
        let m = measure(
            "fn f(x: R) -> R {
                 let a = x.one()?;
                 let b = x.two()?;
                 let c = x.three()?;
                 Ok(c)
             }",
        );
        assert_eq!(
            m.cognitive, 0,
            "idiomatic error propagation is not complexity"
        );
        assert_eq!(m.cyclomatic, 4, "but it is still four paths out");
    }

    #[test]
    fn logical_operators_charge_per_sequence_not_per_operator() {
        let same = measure("fn f(a: bool, b: bool, c: bool) -> bool { a && b && c }");
        assert_eq!(same.cognitive, 1, "one run of &&");

        let mixed = measure("fn f(a: bool, b: bool, c: bool) -> bool { a && b || c }");
        assert_eq!(mixed.cognitive, 2, "&& then || is two runs");
    }

    #[test]
    fn cfg_test_modules_are_not_measured_but_cfg_not_test_is() {
        let w = walk(
            "#[cfg(test)]
             mod tests {
                 fn helper() -> u32 { 1 }
             }
             #[cfg(not(test))]
             mod real {
                 fn shipped() -> u32 { 2 }
             }",
        );
        let keys: Vec<&str> = w.fns.iter().map(|f| f.key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["t::real::shipped"],
            "cfg(not(test)) is production code and must be measured"
        );
    }

    #[test]
    fn test_functions_are_skipped_even_outside_a_cfg_test_module() {
        let w = walk("#[test]\nfn t() -> u32 { 1 }\nfn real() -> u32 { 2 }\n");
        assert_eq!(w.fns.len(), 1);
        assert_eq!(w.fns[0].key, "t::real");
    }

    #[test]
    fn a_closure_with_a_block_body_nests_a_bare_expression_does_not() {
        let bare = measure("fn f(v: V) -> V { v.map(|x| x + 1) }");
        assert_eq!(bare.nesting, 0);

        let block = measure("fn f(v: V) -> V { v.map(|x| { if x > 0 { 1 } else { 0 } }) }");
        assert!(block.nesting >= 2, "got {}", block.nesting);
    }

    #[test]
    fn unwrap_hiding_inside_a_macro_is_still_counted() {
        let w = walk(r#"fn f(x: X) { tracing::warn!("{}", x.unwrap()); }"#);
        assert_eq!(w.file.unwraps, 1, "macro token streams are a hiding place");
    }

    #[test]
    fn the_word_unwrap_in_a_string_or_comment_is_not_counted() {
        let w = walk(
            r#"fn f() {
                   // never call unwrap here
                   let s = "unwrap";
                   let _ = s;
               }"#,
        );
        assert_eq!(
            w.file.unwraps, 0,
            "this is what sinks a regex-based counter"
        );
    }

    #[test]
    fn trait_impls_do_not_collide_in_the_key() {
        let w = walk(
            "struct A; struct B;
             impl std::fmt::Display for A { fn fmt(&self) -> u32 { 1 } }
             impl std::fmt::Debug for B { fn fmt(&self) -> u32 { 2 } }",
        );
        let mut keys: Vec<&str> = w.fns.iter().map(|f| f.key.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["t::<A as Display>::fmt", "t::<B as Debug>::fmt"]);
    }

    #[test]
    fn allow_attributes_are_counted_per_lint_named() {
        let w = walk("#[allow(dead_code, unused)]\nfn f() -> u32 { 1 }\n");
        assert_eq!(w.file.allow_lints, 2);
    }

    /// The workspace's only broad suppression is a file-level `#![allow(dead_code)]`.
    /// Counting outer attributes alone reports zero for a repo that has one.
    #[test]
    fn a_file_level_inner_allow_is_counted() {
        let w = walk("#![allow(dead_code)]\nfn f() -> u32 { 1 }\n");
        assert_eq!(w.file.allow_lints, 1);
    }

    #[test]
    fn let_else_is_one_flow_break() {
        let m = measure("fn f(x: O) -> u32 { let Some(v) = x else { return 0 }; v }");
        assert_eq!(m.cognitive, 1);
    }

    #[test]
    fn an_opaque_macro_lands_in_the_unmeasured_ledger_but_format_does_not() {
        let w = walk(r#"fn f() { slint::include_modules!(); let _ = format!("x"); }"#);
        let macs: Vec<&str> = w.unmeasured.iter().map(|u| u.mac.as_str()).collect();
        assert_eq!(macs, vec!["slint::include_modules"]);
    }
}
