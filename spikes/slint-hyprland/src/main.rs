//! M0a spikes 1+2 — is Slint viable as the Cleanroom GUI on Hyprland/Wayland?
//!
//! egui was the original choice and was ruled out on two open, unfixed bugs:
//!
//!   * egui#8249 — keyboard input completely dead on Hyprland. Mouse works, zero key
//!     events reach the app. Reported against Hyprland 0.53.3; this box runs 0.56.
//!   * egui#8314 — drag-resize pins a core at 100% and `App::ui` stops being called,
//!     and it stays broken after the drag ends. Root-caused in-thread to winit
//!     withholding `RedrawRequested` while a frame callback is outstanding.
//!
//! Slint uses winit too, so it is not automatically immune — hence this probe. Both
//! failures are visible without instrumentation: keys either arrive or they do not, and
//! a free-running counter either keeps moving or freezes.
//!
//! Run inside the dev shell so Wayland/libxkbcommon resolve:
//!     nix develop -c cargo run -p spike-slint-hyprland

use slint::wgpu_29::wgpu;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

slint::include_modules!();

/// Preview size for the interop test. The real daemon will push 640x360; this is
/// deliberately the same order of magnitude so the per-frame upload cost is honest.
const TEX_W: u32 = 640;
const TEX_H: u32 = 360;

fn main() -> anyhow::Result<()> {
    // Report what we are actually running against, so a failure here is attributable to
    // a compositor/backend combination rather than to "Slint is broken".
    println!("=== Slint on Hyprland spike ===");
    for var in [
        "XDG_SESSION_TYPE",
        "XDG_CURRENT_DESKTOP",
        "WAYLAND_DISPLAY",
        "HYPRLAND_INSTANCE_SIGNATURE",
        "SLINT_BACKEND",
    ] {
        println!("  {var:<28} {}", std::env::var(var).unwrap_or_else(|_| "<unset>".into()));
    }

    // Spike 2: force Slint onto the wgpu renderer so we can share textures with it.
    // `Automatic` lets Slint pick the adapter; `WGPUConfiguration::Manual { instance,
    // adapter, device, queue }` is the other arm, which is how the real GUI would hand
    // Slint a device WE created (e.g. one pinned to a specific DRM render node).
    // Proving the notifier path first is enough to answer the spike, and it is the
    // simpler shape for a GUI whose job is only to *display* frames.
    slint::BackendSelector::new()
        .require_wgpu_29(slint::wgpu_29::WGPUConfiguration::default())
        .select()?;

    let ui = AppWindow::new()?;

    // Spike 1b: a timer entirely independent of input and of the render loop. If the
    // window survives a drag-resize, this keeps counting. If winit stops delivering
    // redraws (the egui#8314 failure), the displayed tick freezes even though the timer
    // itself is still firing — which is precisely the distinction we want to observe.
    let tick_timer = Rc::new(slint::Timer::default());
    {
        let weak = ui.as_weak();
        let started = Instant::now();
        let mut last_report = Instant::now();
        let mut frames_since = 0i32;

        tick_timer.start(
            slint::TimerMode::Repeated,
            Duration::from_millis(16),
            move || {
                let Some(ui) = weak.upgrade() else { return };
                let t = ui.get_tick() + 1;
                ui.set_tick(t);
                frames_since += 1;

                if last_report.elapsed() >= Duration::from_millis(500) {
                    let fps = frames_since as f64 / last_report.elapsed().as_secs_f64();
                    ui.set_fps(format!("{fps:.1}").into());
                    frames_since = 0;
                    last_report = Instant::now();

                    // Also log to stdout: if the GUI freezes but this keeps printing, the
                    // event loop is alive and only rendering is wedged. That is a
                    // different bug from the whole loop stalling, and the distinction
                    // decides whether we can work around it.
                    println!(
                        "[{:>6.1}s] tick={} fps={:.1}",
                        started.elapsed().as_secs_f64(),
                        t,
                        fps
                    );
                }

                let sz = ui.window().size();
                ui.set_win_size(format!("{}x{}", sz.width, sz.height).into());
            },
        );
    }

    ui.on_request_quit(|| {
        let _ = slint::quit_event_loop();
    });

    // --- Spike 2: display a wgpu::Texture we allocated and filled ourselves ---------
    //
    // Slint's `Image::try_from(wgpu::Texture)` has two hard requirements, both checked
    // in i-slint-core/graphics/wgpu_29.rs and both easy to get wrong silently:
    //   * format MUST be Rgba8Unorm or Rgba8UnormSrgb
    //   * usages MUST include TEXTURE_BINDING | RENDER_ATTACHMENT
    // Anything else returns TextureImportError rather than rendering incorrectly.
    let gpu: Rc<RefCell<Option<(wgpu::Device, wgpu::Queue, wgpu::Texture)>>> =
        Rc::new(RefCell::new(None));

    {
        let weak = ui.as_weak();
        let gpu = gpu.clone();
        ui.window().set_rendering_notifier(move |state, api| {
            // We only want the one-shot setup event; ignore the per-frame states.
            let (Some(ui), slint::RenderingState::RenderingSetup, slint::GraphicsAPI::WGPU29 { device, queue, .. }) =
                (weak.upgrade(), state, api)
            else {
                return;
            };

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("cleanroom-preview"),
                size: wgpu::Extent3d { width: TEX_W, height: TEX_H, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            *gpu.borrow_mut() = Some((device.clone(), queue.clone(), texture));
            ui.set_gpu_status(format!("wgpu device acquired; {TEX_W}x{TEX_H} Rgba8Unorm texture allocated").into());
            println!("[spike2] got Slint's wgpu device/queue, allocated our own texture");
        })?;
    }

    // Animate the texture so a static-image false pass is impossible: if what you see
    // moves, the upload path is genuinely live rather than a one-frame fluke.
    let gpu_timer = Rc::new(slint::Timer::default());
    {
        let weak = ui.as_weak();
        let gpu = gpu.clone();
        let mut phase = 0u32;
        let mut announced = false;
        gpu_timer.start(slint::TimerMode::Repeated, Duration::from_millis(33), move || {
            let Some(ui) = weak.upgrade() else { return };
            let borrowed = gpu.borrow();
            let Some((_device, queue, texture)) = borrowed.as_ref() else { return };

            phase = phase.wrapping_add(3);
            let pixels = render_pattern(phase);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(TEX_W * 4),
                    rows_per_image: Some(TEX_H),
                },
                wgpu::Extent3d { width: TEX_W, height: TEX_H, depth_or_array_layers: 1 },
            );

            match slint::Image::try_from(texture.clone()) {
                Ok(img) => {
                    ui.set_gpu_frame(img);
                    if !announced {
                        ui.set_gpu_status("wgpu interop WORKING — this is our own texture".into());
                        println!("[spike2] Image::try_from(wgpu::Texture) succeeded — interop works");
                        announced = true;
                    }
                }
                Err(e) => {
                    ui.set_gpu_status(format!("interop FAILED: {e}").into());
                    println!("[spike2] Image::try_from failed: {e}");
                }
            }
        });
    }

    println!("\nWindow open. Checks:");
    println!("  1. Type — 'keys received' must climb. If it stays 0, Slint has egui#8249.");
    println!("  2. Drag the window edge — tick/fps must keep moving during AND after.");
    println!("  3. Watch stdout: if it prints while the window is frozen, only rendering stalled.");

    ui.run()?;

    println!("\nevent loop exited cleanly");
    Ok(())
}

/// Animated RGBA8 pattern, generated on the CPU. Deliberately not a compute shader:
/// this spike is about the *interop*, and adding a compute pipeline would confound a
/// failure between "Slint won't take our texture" and "our shader is wrong". The real
/// pipeline writes the same texture from WGSL with STORAGE_BINDING added.
fn render_pattern(phase: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (TEX_W * TEX_H * 4) as usize];
    let p = phase as f32 * 0.02;
    for y in 0..TEX_H {
        for x in 0..TEX_W {
            let fx = x as f32 / TEX_W as f32;
            let fy = y as f32 / TEX_H as f32;
            let wave = ((fx * 8.0 + p).sin() * (fy * 6.0 - p * 0.7).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
            let i = ((y * TEX_W + x) * 4) as usize;
            buf[i] = (40.0 + wave * 60.0) as u8;
            buf[i + 1] = (90.0 + wave * 140.0) as u8;
            buf[i + 2] = (120.0 + wave * 100.0) as u8;
            buf[i + 3] = 255;
        }
    }
    buf
}
