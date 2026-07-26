//! Dotted-path access to the config, e.g. `video.blur_strength`.
//!
//! Implemented by round-tripping through `toml::Value` rather than by matching on every
//! key by hand. That is not just less code — it means **every write is validated by the
//! real schema**. `deny_unknown_fields` rejects typos, and `CaptureTarget`'s
//! `TryFrom<String>` rejects a value that would make the microphone capture itself.
//! A hand-written match would have to remember to re-check all of that.

use cleanroom_core::Config;

/// Settings that are `Option<T>` in the schema.
///
/// These need naming explicitly because **TOML has no null**. A `None` field serialises
/// to nothing at all, so it is simply absent from the document — which means the generic
/// "walk the TOML tree" approach cannot discover it, and `set audio.device ...` would
/// fail with "no such setting" on a fresh config. Listing them lets us create a key that
/// is not there yet, and report an absent one as `(unset)` rather than as missing.
const OPTIONAL_KEYS: &[&str] = &[
    "audio.device",
    "video.device",
    "video.background_image",
    "video.matte_tighten",
    "video.matting_width",
    "video.matting_height",
    "gpu.render_node",
];

/// Optional keys whose value is a number rather than a string.
///
/// Type is normally inferred from the value already in the document — but an *unset*
/// optional has no value to infer from, and the fallback was "it must be a string". For
/// `audio.device` that is right. For `video.matte_tighten` it silently produced the TOML
/// `matte_tighten = "0.35"`, which fails to deserialise into `Option<f32>`, so the whole
/// `set` was rejected and the slider sprang back to where it started with no error anywhere
/// the user could see it. An unset optional therefore has to declare its type somewhere,
/// and this is that somewhere.
const OPTIONAL_FLOAT_KEYS: &[&str] = &["video.matte_tighten"];
const OPTIONAL_INT_KEYS: &[&str] = &["video.matting_width", "video.matting_height"];

/// The TOML value an unset optional should be created as.
fn optional_value(key: &str, value: &str) -> toml::Value {
    let v = value.trim();
    if OPTIONAL_FLOAT_KEYS.contains(&key)
        && let Ok(f) = v.parse::<f64>()
    {
        return toml::Value::Float(f);
    }
    if OPTIONAL_INT_KEYS.contains(&key)
        && let Ok(i) = v.parse::<i64>()
    {
        return toml::Value::Integer(i);
    }
    // Unparseable numbers fall through as strings on purpose: the schema rejects them a
    // moment later with a message naming the key and the value, which is a better error
    // than one produced here without that context.
    toml::Value::String(v.to_string())
}

/// What `get` prints, and what `set` accepts, for an optional setting with no value.
pub const UNSET: &str = "(unset)";

/// Words a user might reasonably type to clear an optional setting.
const CLEARING_WORDS: &[&str] = &["", "(unset)", "unset", "none", "null", "auto", "default"];

fn is_optional(key: &str) -> bool {
    OPTIONAL_KEYS.contains(&key)
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("no such setting '{0}' (try `cleanroom-ctl keys`)")]
    UnknownKey(String),

    #[error("'{value}' is not valid for '{key}': {reason}")]
    InvalidValue {
        key: String,
        value: String,
        reason: String,
    },

    #[error("'{0}' is a section, not a setting — name a leaf like 'video.blur_strength'")]
    NotALeaf(String),

    #[error("internal: config could not be represented as TOML: {0}")]
    Serialise(String),
}

/// Read one setting, rendered the way the user would type it.
pub fn get(config: &Config, key: &str) -> Result<String, SettingsError> {
    let root = to_value(config)?;
    match walk(&root, key) {
        Some(toml::Value::Table(_)) => Err(SettingsError::NotALeaf(key.to_string())),
        Some(other) => Ok(render(other)),
        // Absent, but a known optional: it is set to nothing, which is different from
        // not existing.
        None if is_optional(key) => Ok(UNSET.to_string()),
        None => Err(SettingsError::UnknownKey(key.to_string())),
    }
}

/// Write one setting, returning the updated config.
///
/// The caller persists it. Kept pure so a rejected value cannot leave the running
/// pipeline half-updated: either the whole new config validates, or nothing changes.
pub fn set(config: &Config, key: &str, value: &str) -> Result<Config, SettingsError> {
    let mut root = to_value(config)?;

    // Confirm the key is real before touching anything: a typo should be a clean error,
    // not a silently-created field that never does anything.
    let parsed = match walk(&root, key) {
        Some(toml::Value::Table(_)) => return Err(SettingsError::NotALeaf(key.to_string())),
        Some(existing) => coerce(value, existing),
        None if is_optional(key) => {
            // Clearing an optional means removing the key, since TOML cannot hold a null.
            if CLEARING_WORDS.contains(&value.trim().to_ascii_lowercase().as_str()) {
                remove(&mut root, key);
                return finish(root, key, value);
            }
            optional_value(key, value)
        }
        None => return Err(SettingsError::UnknownKey(key.to_string())),
    };

    if is_optional(key) && CLEARING_WORDS.contains(&value.trim().to_ascii_lowercase().as_str()) {
        remove(&mut root, key);
        return finish(root, key, value);
    }

    insert(&mut root, key, parsed).ok_or_else(|| SettingsError::UnknownKey(key.to_string()))?;
    finish(root, key, value)
}

/// The validation gate, shared by every path through `set`.
fn finish(root: toml::Value, key: &str, value: &str) -> Result<Config, SettingsError> {
    // The real validation gate. Deserialising back through the schema is what enforces
    // types, unknown-field rejection and CaptureTarget's self-reference refusal.
    root.try_into::<Config>()
        .map_err(|e| SettingsError::InvalidValue {
            key: key.to_string(),
            value: value.to_string(),
            reason: e.to_string(),
        })
}

/// Every leaf key with its current value, depth-first. Drives `ctl keys` and completion.
pub fn keys(config: &Config) -> Result<Vec<(String, String)>, SettingsError> {
    let root = to_value(config)?;
    let mut out = Vec::new();
    collect(&root, String::new(), &mut out);

    // Optional settings that are currently unset are absent from the document, so they
    // have to be added back by name — otherwise they would be invisible in the UI and in
    // shell completion precisely when the user most needs to discover them.
    for opt in OPTIONAL_KEYS {
        if !out.iter().any(|(k, _)| k == opt) {
            out.push((opt.to_string(), UNSET.to_string()));
        }
    }

    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

// --- internals ---------------------------------------------------------------------

fn to_value(config: &Config) -> Result<toml::Value, SettingsError> {
    toml::Value::try_from(config).map_err(|e| SettingsError::Serialise(e.to_string()))
}

fn walk<'a>(root: &'a toml::Value, key: &str) -> Option<&'a toml::Value> {
    key.split('.').try_fold(root, |cur, seg| cur.get(seg))
}

fn parent_table<'a>(root: &'a mut toml::Value, key: &str) -> Option<(&'a mut toml::Table, String)> {
    let mut segs: Vec<&str> = key.split('.').collect();
    let leaf = segs.pop()?.to_string();
    let mut cur = root;
    for seg in segs {
        cur = cur.get_mut(seg)?;
    }
    Some((cur.as_table_mut()?, leaf))
}

fn insert(root: &mut toml::Value, key: &str, new: toml::Value) -> Option<()> {
    let (table, leaf) = parent_table(root, key)?;
    table.insert(leaf, new);
    Some(())
}

fn remove(root: &mut toml::Value, key: &str) {
    if let Some((table, leaf)) = parent_table(root, key) {
        table.remove(&leaf);
    }
}

/// Turn user text into a TOML value shaped like the one already there.
///
/// Typing `true`, `0.6` or `blur` should all work without quoting, so we take the
/// existing value's type as the hint. Anything we cannot coerce stays a string and gets
/// rejected downstream with a schema error, which is a better message than a parse one.
fn coerce(value: &str, existing: &toml::Value) -> toml::Value {
    let v = value.trim();
    match existing {
        toml::Value::Boolean(_) => match v {
            "true" | "on" | "yes" | "1" => toml::Value::Boolean(true),
            "false" | "off" | "no" | "0" => toml::Value::Boolean(false),
            other => toml::Value::String(other.to_string()),
        },
        toml::Value::Integer(_) => v
            .parse::<i64>()
            .map(toml::Value::Integer)
            .unwrap_or_else(|_| toml::Value::String(v.to_string())),
        toml::Value::Float(_) => v
            .parse::<f64>()
            .map(toml::Value::Float)
            .unwrap_or_else(|_| toml::Value::String(v.to_string())),
        _ => toml::Value::String(v.to_string()),
    }
}

fn render(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        // The schema uses f32 but TOML carries f64, so a straight Display prints
        // `0.6000000238418579` for what the user typed as `0.6`. If the value survives a
        // round-trip through f32 it *came* from an f32 field, so print it as one.
        toml::Value::Float(f) => {
            let narrowed = *f as f32;
            if narrowed as f64 == *f {
                narrowed.to_string()
            } else {
                f.to_string()
            }
        }
        toml::Value::Boolean(b) => b.to_string(),
        other => other.to_string(),
    }
}

fn collect(v: &toml::Value, prefix: String, out: &mut Vec<(String, String)>) {
    match v {
        toml::Value::Table(t) => {
            for (k, child) in t {
                let path = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                collect(child, path, out);
            }
        }
        leaf => out.push((prefix, render(leaf))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cleanroom_core::{BackgroundMode, VIRTUAL_MIC_NODE};

    /// An unset *numeric* optional has no existing value to infer a type from, and the
    /// fallback used to be "assume string". That wrote `matte_tighten = "0.35"`, which the
    /// schema rejects, so the whole `set` failed — and because the GUI writes a value and
    /// then re-reads it, the slider sprang back to zero with no error shown anywhere. The
    /// user-visible symptom was a control that simply refused to move.
    #[test]
    fn an_unset_numeric_optional_takes_a_number() {
        let c = set(&Config::default(), "video.matte_tighten", "0.35").unwrap();
        assert_eq!(c.video.matte_tighten, Some(0.35));
        // And again now that it holds a value, which is the path that always worked.
        let c = set(&c, "video.matte_tighten", "0.5").unwrap();
        assert_eq!(c.video.matte_tighten, Some(0.5));

        let c = set(&Config::default(), "video.matting_width", "384").unwrap();
        assert_eq!(c.video.matting_width, Some(384));
    }

    /// `auto` has to keep meaning "decide it for me" rather than becoming the number 0,
    /// because for `matte_tighten` the two differ: unset is per-mode, 0 is "never tighten".
    #[test]
    fn a_numeric_optional_still_clears() {
        let c = set(&Config::default(), "video.matte_tighten", "0.35").unwrap();
        let c = set(&c, "video.matte_tighten", "auto").unwrap();
        assert_eq!(c.video.matte_tighten, None);
        assert_eq!(get(&c, "video.matte_tighten").unwrap(), UNSET);
        // Unset resolves per mode; explicit zero does not.
        assert_eq!(c.video.tighten_for(BackgroundMode::Replace), 0.12);
        let z = set(&c, "video.matte_tighten", "0").unwrap();
        assert_eq!(z.video.tighten_for(BackgroundMode::Replace), 0.0);
    }

    #[test]
    fn reads_a_nested_value() {
        let c = Config::default();
        assert_eq!(get(&c, "video.fps").unwrap(), "30");
        assert_eq!(get(&c, "audio.denoise.enabled").unwrap(), "true");
    }

    #[test]
    fn writes_a_float() {
        let c = set(&Config::default(), "video.blur_strength", "0.85").unwrap();
        assert_eq!(c.video.blur_strength, 0.85);
    }

    #[test]
    fn writes_a_bool_from_friendly_spellings() {
        for word in ["false", "off", "no", "0"] {
            let c = set(&Config::default(), "video.mirror", word).unwrap();
            assert!(!c.video.mirror, "'{word}' should mean false");
        }
        for word in ["true", "on", "yes", "1"] {
            let c = set(&Config::default(), "video.power_save", word).unwrap();
            assert!(c.video.power_save, "'{word}' should mean true");
        }
    }

    #[test]
    fn writes_an_enum() {
        let c = set(&Config::default(), "video.background", "replace").unwrap();
        assert_eq!(c.video.background, BackgroundMode::Replace);
    }

    #[test]
    fn unknown_key_is_an_error_not_a_new_field() {
        assert!(matches!(
            set(&Config::default(), "video.blur_strenght", "0.5"),
            Err(SettingsError::UnknownKey(_))
        ));
    }

    #[test]
    fn a_section_is_not_settable() {
        assert!(matches!(
            get(&Config::default(), "video"),
            Err(SettingsError::NotALeaf(_))
        ));
    }

    #[test]
    fn bad_type_is_rejected() {
        assert!(matches!(
            set(&Config::default(), "video.fps", "banana"),
            Err(SettingsError::InvalidValue { .. })
        ));
    }

    #[test]
    fn cannot_set_the_mic_to_our_own_node() {
        // The self-capture guard must survive the generic path. This is the reason
        // validation goes back through the schema rather than being reimplemented here.
        let err = set(&Config::default(), "audio.device", VIRTUAL_MIC_NODE)
            .expect_err("setting our own node as the capture source must fail");
        assert!(matches!(err, SettingsError::InvalidValue { .. }));
    }

    // --- optional settings ---------------------------------------------------------
    // TOML has no null, so a None field is simply absent from the document. Without
    // special handling every one of these would report "no such setting" on a fresh
    // config — which is exactly when a user is most likely to want to set them.

    #[test]
    fn an_unset_optional_reads_as_unset_not_as_missing() {
        let c = Config::default();
        assert!(c.audio.device.is_none());
        assert_eq!(get(&c, "audio.device").unwrap(), UNSET);
        assert_eq!(get(&c, "gpu.render_node").unwrap(), UNSET);
    }

    #[test]
    fn an_unset_optional_can_be_set() {
        let c = set(
            &Config::default(),
            "audio.device",
            "alsa_input.usb-Focusrite_Scarlett_Solo-00.HiFi__Mic1__source",
        )
        .expect("setting an absent optional must work");
        assert!(c.audio.device.is_some());
        assert!(get(&c, "audio.device").unwrap().contains("Scarlett"));
    }

    #[test]
    fn an_optional_can_be_cleared_again() {
        let set_once = set(&Config::default(), "gpu.render_node", "/dev/dri/renderD128").unwrap();
        assert!(set_once.gpu.render_node.is_some());

        for word in ["", "unset", "none", "auto", "default"] {
            let cleared = set(&set_once, "gpu.render_node", word)
                .unwrap_or_else(|e| panic!("'{word}' should clear the setting: {e}"));
            assert!(
                cleared.gpu.render_node.is_none(),
                "'{word}' should clear it"
            );
        }
    }

    #[test]
    fn setting_an_absent_optional_still_validates() {
        // The absent-key path must not become a hole in validation: it is a *new* code
        // path to the schema, not a way around it.
        assert!(matches!(
            set(&Config::default(), "audio.device", VIRTUAL_MIC_NODE),
            Err(SettingsError::InvalidValue { .. })
        ));
    }

    #[test]
    fn unset_optionals_still_appear_in_keys() {
        // Otherwise they are undiscoverable from the CLI and from shell completion.
        let ks = keys(&Config::default()).unwrap();
        for opt in OPTIONAL_KEYS {
            let entry = ks.iter().find(|(k, _)| k == opt);
            assert!(entry.is_some(), "optional key '{opt}' missing from keys()");
            assert_eq!(entry.unwrap().1, UNSET);
        }
    }

    #[test]
    fn floats_render_the_way_the_user_typed_them() {
        // The schema is f32 but TOML carries f64, so a naive Display prints
        // 0.6000000238418579 for a value the user set as 0.6.
        let c = set(&Config::default(), "video.blur_strength", "0.85").unwrap();
        assert_eq!(get(&c, "video.blur_strength").unwrap(), "0.85");
        assert_eq!(
            get(&Config::default(), "video.blur_strength").unwrap(),
            "0.6"
        );
    }

    #[test]
    fn a_rejected_write_leaves_the_original_untouched() {
        let original = Config::default();
        let _ = set(&original, "video.fps", "nonsense");
        assert_eq!(original, Config::default());
    }

    #[test]
    fn keys_lists_leaves_only_and_is_sorted() {
        let ks = keys(&Config::default()).unwrap();
        assert!(ks.iter().any(|(k, _)| k == "video.blur_strength"));
        assert!(ks.iter().any(|(k, _)| k == "audio.denoise.attenuation_db"));
        assert!(
            !ks.iter().any(|(k, _)| k == "video"),
            "sections must not appear"
        );
        let mut sorted = ks.clone();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(ks, sorted);
    }

    #[test]
    fn every_listed_key_is_readable() {
        // Guards against `keys` and `get` drifting apart, which would make shell
        // completion offer things that then fail.
        let c = Config::default();
        for (k, _) in keys(&c).unwrap() {
            get(&c, &k).unwrap_or_else(|e| panic!("listed key '{k}' is not readable: {e}"));
        }
    }
}
