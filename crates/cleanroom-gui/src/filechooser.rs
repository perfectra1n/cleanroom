//! Picking a background image through the XDG FileChooser portal.
//!
//! Done by hand over the zbus we already depend on, rather than with `rfd`.
//!
//! `rfd`'s XDG backend pulls in `ashpd`, which pulls in `zbus` with feature choices we do
//! not control — and enabling zbus's `tokio` feature *anywhere* in the graph switches its
//! executor globally, which makes `accesskit_unix` panic ("there is no reactor running")
//! and silently takes screen-reader support with it. That trap is documented in four of
//! this workspace's manifests. Adding a dependency that could re-enable it, to open a file
//! dialog, is a poor trade for roughly a hundred lines.
//!
//! ## The handle-token dance
//!
//! The portal returns an object path and then emits a `Response` signal on it. Subscribing
//! *after* the call is a race — a fast portal can answer first — so the path is predicted
//! from a `handle_token` we choose and subscribed to before calling.

use futures_util::StreamExt;
use std::collections::HashMap;
use zbus::zvariant::{OwnedValue, Value};

/// Ask the portal for an image file. `Ok(None)` means the user cancelled.
///
/// Returns an error when no portal is available, which is a real possibility on a bare
/// wlroots session: FileChooser is implemented by xdg-desktop-portal-gtk or -kde, and a
/// minimal setup may have neither. The caller falls back to a path field rather than
/// pretending the feature exists.
pub async fn pick_image() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let connection = zbus::Connection::session().await?;

    // Unique per call, and restricted to [A-Za-z0-9_] because it becomes part of an object
    // path. A timestamp is enough: two dialogs in the same nanosecond from one process is
    // not a case worth handling.
    let token = format!(
        "cleanroom_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let unique = connection
        .unique_name()
        .ok_or("no unique bus name")?
        .to_string();
    // The sender part of the path is our unique name with the leading ':' dropped and dots
    // turned into underscores. This is specified, not guessed.
    let sender = unique.trim_start_matches(':').replace('.', "_");
    let request_path = format!("/org/freedesktop/portal/desktop/request/{sender}/{token}");

    // Subscribe BEFORE calling, or a fast portal answers into the void.
    let mut responses = zbus::MessageStream::for_match_rule(
        zbus::MatchRule::builder()
            .msg_type(zbus::message::Type::Signal)
            .interface("org.freedesktop.portal.Request")?
            .member("Response")?
            .path(request_path.as_str())?
            .build(),
        &connection,
        Some(4),
    )
    .await?;

    let mut options: HashMap<&str, Value> = HashMap::new();
    options.insert("handle_token", Value::from(token.as_str()));
    options.insert("modal", Value::from(true));
    // A filter the user can see and change, rather than a hard restriction: the daemon
    // accepts PNG and JPEG, and offering everything else would just move the failure later.
    let filters = vec![(
        "Images".to_string(),
        vec![
            (0u32, "*.png".to_string()),
            (0u32, "*.jpg".to_string()),
            (0u32, "*.jpeg".to_string()),
            (0u32, "*.PNG".to_string()),
            (0u32, "*.JPG".to_string()),
        ],
    )];
    options.insert("filters", Value::from(filters));

    let proxy = zbus::Proxy::new(
        &connection,
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
        "org.freedesktop.portal.FileChooser",
    )
    .await?;

    // parent_window is empty: we have no way to get an xdg_foreign handle for a Slint
    // window, so the dialog is unparented. Cosmetic, and better than not opening.
    let _: zbus::zvariant::OwnedObjectPath = proxy
        .call("OpenFile", &("", "Choose a background image", options))
        .await?;

    // Bounded wait. Without one, a portal that dies mid-dialog leaves this task pending
    // forever holding a match rule.
    let message = tokio::time::timeout(std::time::Duration::from_secs(300), responses.next())
        .await
        .map_err(|_| "the file dialog timed out")?
        .ok_or("the portal closed without responding")??;

    let (code, results): (u32, HashMap<String, OwnedValue>) = message.body().deserialize()?;
    if code != 0 {
        // 1 is user cancellation, 2 is "ended some other way". Neither is an error.
        return Ok(None);
    }

    let uris = results.get("uris").ok_or("portal returned no uris")?;
    let uris: Vec<String> = uris.try_clone()?.try_into()?;
    let Some(first) = uris.first() else {
        return Ok(None);
    };

    Ok(Some(uri_to_path(first)))
}

/// `file:///home/me/a%20photo.png` to `/home/me/a photo.png`.
///
/// The daemon takes a filesystem path, not a URI, and percent-encoding is not optional —
/// a space in a filename is common enough that ignoring it would break the feature for a
/// meaningful fraction of people.
fn uri_to_path(uri: &str) -> String {
    let path = uri.strip_prefix("file://").unwrap_or(uri);
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(b) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_uri_becomes_a_path() {
        assert_eq!(uri_to_path("file:///home/me/bg.png"), "/home/me/bg.png");
    }

    /// Spaces in filenames are common, and a portal always percent-encodes them. Leaving
    /// them encoded hands the daemon a path that does not exist, which surfaces as
    /// "cannot read" on a file the user can plainly see.
    #[test]
    fn percent_encoding_is_decoded() {
        assert_eq!(
            uri_to_path("file:///home/me/a%20photo%20(1).png"),
            "/home/me/a photo (1).png"
        );
        assert_eq!(uri_to_path("file:///tmp/%C3%A9t%C3%A9.jpg"), "/tmp/été.jpg");
    }

    /// A stray '%' must not eat the rest of the path or panic.
    #[test]
    fn a_malformed_escape_is_left_alone() {
        assert_eq!(uri_to_path("file:///tmp/100%.png"), "/tmp/100%.png");
        assert_eq!(uri_to_path("file:///tmp/%zz.png"), "/tmp/%zz.png");
        assert_eq!(uri_to_path("file:///tmp/trailing%"), "/tmp/trailing%");
    }

    #[test]
    fn a_bare_path_passes_through() {
        assert_eq!(uri_to_path("/home/me/bg.png"), "/home/me/bg.png");
    }
}
