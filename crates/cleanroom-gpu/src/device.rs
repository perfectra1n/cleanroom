//! GPU device selection.
//!
//! The one thing this module exists to prevent: **taking whatever adapter happens to be
//! first.** On the reference machine that is a coin flip between an RTX 5090 and a 2-CU
//! Radeon iGPU, and the measured difference on the matting workload is 4.53 ms against
//! 38.77 ms per frame — same binary, same model, an 8.5x gap that decides whether 1080p30
//! fits in budget at all.
//!
//! So the adapter is chosen deliberately: by DRM render node if the user pinned one, and
//! otherwise by preferring discrete hardware. Whatever is chosen is *reported*, because
//! "which GPU am I actually using" should never be something you infer from frame times.

use std::path::{Path, PathBuf};

/// A Vulkan-only instance.
///
/// `InstanceDescriptor` has no `Default` in wgpu 29, so every field is named — which is
/// no bad thing here, since the backend choice is a deliberate constraint rather than a
/// detail: Vulkan is the one backend both vendors expose without a vendor SDK.
fn vulkan_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        // Headless: the daemon never presents to a surface, it only computes.
        display: None,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error(
        "no Vulkan adapter found. Cleanroom requires a GPU — there is deliberately no CPU \
         fallback, because a CPU fallback nobody notices is worse than an error. Check \
         that a Vulkan driver is installed (`cleanroom-ctl doctor`)."
    )]
    NoAdapter,

    #[error("no adapter matches the pinned render node {0}. Available: {1}")]
    PinnedNotFound(String, String),

    #[error("could not create a device on {adapter}: {source}")]
    DeviceCreation {
        adapter: String,
        #[source]
        source: wgpu::RequestDeviceError,
    },
}

/// A chosen adapter, with everything needed to explain the choice.
#[derive(Debug, Clone)]
pub struct AdapterChoice {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    /// The DRM render node this adapter corresponds to, when it could be determined.
    pub render_node: Option<PathBuf>,
}

impl std::fmt::Display for AdapterChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [{}, {}]", self.name, self.device_type, self.backend)?;
        if let Some(n) = &self.render_node {
            write!(f, " {}", n.display())?;
        }
        Ok(())
    }
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    pub choice: AdapterChoice,
}

impl Gpu {
    /// Create a device, optionally pinned to a DRM render node.
    pub fn new(render_node: Option<&Path>) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(render_node))
    }

    pub async fn new_async(render_node: Option<&Path>) -> Result<Self, GpuError> {
        // Vulkan only. It is the one backend available on both vendors without a vendor
        // SDK, which is the premise of the project. Allowing GL would silently work and
        // be far slower — exactly the quiet degradation we refuse.
        let instance = vulkan_instance();
        // enumerate_adapters is async in wgpu 29.
        let adapters: Vec<wgpu::Adapter> =
            instance.enumerate_adapters(wgpu::Backends::VULKAN).await;
        if adapters.is_empty() {
            return Err(GpuError::NoAdapter);
        }

        let described: Vec<(wgpu::Adapter, AdapterChoice)> = adapters
            .into_iter()
            .map(|a| {
                let info = a.get_info();
                let choice = AdapterChoice {
                    name: info.name.clone(),
                    backend: format!("{:?}", info.backend),
                    device_type: format!("{:?}", info.device_type),
                    render_node: guess_render_node(&info),
                };
                (a, choice)
            })
            .collect();

        for (_, c) in &described {
            tracing::debug!(adapter = %c, "available");
        }

        let available = described
            .iter()
            .map(|(_, c)| c.to_string())
            .collect::<Vec<_>>()
            .join("; ");

        let (adapter, choice) = match render_node {
            Some(want) => described
                .into_iter()
                .find(|(_, c)| c.render_node.as_deref() == Some(want))
                .ok_or_else(|| GpuError::PinnedNotFound(want.display().to_string(), available))?,
            None => pick_best(described),
        };

        tracing::info!(adapter = %choice, "GPU selected");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("cleanroom"),
                required_features: wgpu::Features::empty(),
                // Ask only for what the pipeline needs. Requesting the adapter's full
                // limits would fail on modest hardware for no benefit.
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: 8192,
                    ..wgpu::Limits::downlevel_defaults()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(|source| GpuError::DeviceCreation {
                adapter: choice.name.clone(),
                source,
            })?;

        Ok(Self {
            device,
            queue,
            adapter,
            choice,
        })
    }

    /// Every adapter Vulkan can see, for `doctor` and for a GUI picker.
    pub fn list_adapters() -> Vec<AdapterChoice> {
        let instance = vulkan_instance();
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
            .into_iter()
            .map(|a| {
                let info = a.get_info();
                AdapterChoice {
                    name: info.name.clone(),
                    backend: format!("{:?}", info.backend),
                    device_type: format!("{:?}", info.device_type),
                    render_node: guess_render_node(&info),
                }
            })
            .collect()
    }
}

/// Prefer a discrete GPU, then integrated, then anything.
///
/// `Cpu` (lavapipe) is ranked last rather than excluded: software Vulkan is genuinely
/// useful for CI, where it is the only way to exercise the GPU path headlessly. It just
/// must never be picked while real hardware exists.
fn pick_best(mut described: Vec<(wgpu::Adapter, AdapterChoice)>) -> (wgpu::Adapter, AdapterChoice) {
    described.sort_by_key(|(a, _)| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    });
    described
        .into_iter()
        .next()
        .expect("non-empty, checked by the caller")
}

/// Map an adapter back to its DRM render node.
///
/// wgpu does not expose the DRM node, so this matches on the PCI device id recorded in
/// each node's sysfs entry. Best-effort by design: `None` only means we cannot *name* the
/// node, not that the adapter is unusable.
fn guess_render_node(info: &wgpu::AdapterInfo) -> Option<PathBuf> {
    let entries = std::fs::read_dir("/dev/dri").ok()?;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with("renderD") {
            continue;
        }
        let device_id = std::fs::read_to_string(format!("/sys/class/drm/{name}/device/device"))
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        if device_id == Some(info.device) {
            return Some(e.path());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_can_be_created_and_names_itself() {
        let gpu = match Gpu::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU available ({e}); skipping");
                return;
            }
        };
        assert!(!gpu.choice.name.is_empty());
        eprintln!("selected: {}", gpu.choice);
    }

    #[test]
    fn discrete_hardware_is_preferred_over_integrated_and_software() {
        // The 8.5x decision. If this regresses the pipeline still "works", just several
        // times slower — exactly the kind of silent loss worth a test.
        let adapters = Gpu::list_adapters();
        if adapters.len() < 2 {
            eprintln!("only {} adapter(s); nothing to rank", adapters.len());
            return;
        }
        for a in &adapters {
            eprintln!("  {a}");
        }
        let Ok(gpu) = Gpu::new(None) else { return };
        if adapters.iter().any(|a| a.device_type == "DiscreteGpu") {
            assert_eq!(
                gpu.choice.device_type, "DiscreteGpu",
                "a discrete GPU was available but {} was chosen",
                gpu.choice
            );
        }
    }

    #[test]
    fn pinning_to_a_render_node_selects_that_adapter() {
        let adapters = Gpu::list_adapters();
        let Some(target) = adapters.iter().find_map(|a| a.render_node.clone()) else {
            eprintln!("no adapter mapped to a render node; skipping");
            return;
        };
        let Ok(gpu) = Gpu::new(Some(&target)) else {
            eprintln!(
                "could not create a device on {}; skipping",
                target.display()
            );
            return;
        };
        assert_eq!(gpu.choice.render_node.as_deref(), Some(target.as_path()));
    }

    #[test]
    fn pinning_to_a_nonexistent_node_errors_rather_than_falling_back() {
        // Silently using a different GPU than the one pinned would be the worst outcome:
        // the user pinned it for a reason and would never learn it was ignored.
        match Gpu::new(Some(Path::new("/dev/dri/renderD999"))) {
            Err(GpuError::PinnedNotFound(..)) => {}
            Err(GpuError::NoAdapter) => eprintln!("no GPU at all; skipping"),
            Err(e) => panic!("wrong error for a bad pin: {e}"),
            Ok(g) => panic!("silently fell back to {}", g.choice),
        }
    }
}
