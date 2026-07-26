# Global shortcuts

Cleanroom deliberately does **not** register global shortcuts itself. Bind a key in your
compositor to `cleanroom-ctl` instead.

## Why not in-process

The Wayland `GlobalShortcuts` portal is not a workable base today:

* it is **unimplemented on sway, river and niri**, so the feature would be missing on a
  large share of this project's likely users;
* on Hyprland it does not show which application registered a binding, so a shortcut that
  stops working has no way to be diagnosed;
* since xdg-desktop-portal 1.21, a host application with no valid app ID is refused
  `GlobalShortcuts` outright.

A compositor binding has none of those problems, works everywhere, is visible in the config
the user already owns, and cannot silently stop working after a portal update.

If a portal-based path is ever added opportunistically for GNOME and KDE, note that
`org.freedesktop.host.portal.Registry.Register()` must be called **once, first, before any
other portal call** — not doing so is what produces the "refused outright" case above.

## The commands worth binding

```sh
cleanroom-ctl set video.background off        # effects off
cleanroom-ctl set video.background blur       # blur
cleanroom-ctl set video.background replace    # replacement image
cleanroom-ctl set audio.denoise.enabled false # noise suppression off
```

Every one of these is applied immediately and saved, and is exactly what the GUI does — the
D-Bus interface is the same for both.

## Hyprland

`~/.config/hypr/hyprland.conf`:

```
bind = SUPER SHIFT, B, exec, cleanroom-ctl set video.background blur
bind = SUPER SHIFT, N, exec, cleanroom-ctl set video.background off
bind = SUPER SHIFT, M, exec, cleanroom-ctl set audio.denoise.enabled false
```

## sway / i3

`~/.config/sway/config`:

```
bindsym $mod+Shift+b exec cleanroom-ctl set video.background blur
bindsym $mod+Shift+n exec cleanroom-ctl set video.background off
```

## niri

`~/.config/niri/config.kdl`:

```kdl
binds {
    Mod+Shift+B { spawn "cleanroom-ctl" "set" "video.background" "blur"; }
    Mod+Shift+N { spawn "cleanroom-ctl" "set" "video.background" "off"; }
}
```

## GNOME

Settings → Keyboard → Custom Shortcuts, with the command
`cleanroom-ctl set video.background blur`.

## KDE Plasma

System Settings → Shortcuts → Add Command, same command.

## Toggling rather than setting

`cleanroom-ctl` has no toggle verb, on purpose: a toggle read from a shell has a race with
the GUI and the tray, and "it went the wrong way sometimes" is a miserable bug to chase.
Read the value and branch if you want one:

```sh
#!/bin/sh
# Toggle blur/off in one keypress.
case "$(cleanroom-ctl get video.background)" in
  off) cleanroom-ctl set video.background blur ;;
  *)   cleanroom-ctl set video.background off ;;
esac
```

The daemon is D-Bus activated, so any of these starts it if it is not already running.
