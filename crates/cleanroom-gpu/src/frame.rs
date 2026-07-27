//! The GPU frame pipeline: YUY2 in, effects, YUY2 out.
//!
//! Every stage between the upload and the readback stays on the GPU. The two copies at
//! the ends are unavoidable — a UVC camera DMAs into system RAM over USB, and
//! v4l2loopback has no `V4L2_MEMORY_DMABUF` support — but they are also cheap: 1080p YUY2
//! is about 124 MB/s, roughly 0.2 ms per transfer on PCIe 5.0, against a 33 ms frame
//! budget. Chasing zero-copy here would optimise the wrong axis.

use crate::device::Gpu;
use cleanroom_core::BackgroundMode;
use wgpu::util::DeviceExt;

/// Uniform block for the composite pass. `repr(C)` and 16-byte aligned because WGSL
/// uniform layout rules are unforgiving about padding.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    mode: u32,
    mirror: u32,
    desaturate: f32,
    dim: f32,
    guided: u32,
    tighten: f32,
    feather: f32,
    _pad0: u32,
}

/// Guided-filter window radius, in matte pixels.
///
/// 3 gives a 7x7 window: 49 taps over 512x288, which is 7.2M texture reads a frame. That is
/// nothing on a discrete GPU and still affordable on the 2-CU iGPU that is this project's
/// slow-hardware conformance target. Larger windows smooth more but start pulling in
/// structure that has nothing to do with the subject's edge.
///
/// The default; `video.guided_radius` overrides it.
const GUIDED_RADIUS: i32 = 3;

/// Guided-filter regularisation.
///
/// Sets what counts as a "flat" region: below this variance the filter smooths, above it
/// the alpha is allowed to follow the image. 1e-4 on luma in 0..1 corresponds to roughly a
/// 1% contrast step, which keeps sensor noise on a plain wall from being read as an edge
/// while still tracking a real shoulder against a similarly-lit background.
///
/// The default; `video.guided_eps` overrides it.
const GUIDED_EPS: f32 = 1e-4;

/// Everything about how a frame is *composited*, as one value.
///
/// A struct rather than eight positional arguments to `process`: `mirror`, `desaturate` and
/// `dim` are all small scalars, and a call site that reads `(.., 0.0, 0.0, true, false)` is
/// one transposition away from a bug nothing would catch.
#[derive(Debug, Clone, Copy)]
pub struct Look {
    pub mode: BackgroundMode,
    /// 0.0..=1.0, mapped continuously onto pyramid depth plus tap spacing.
    pub blur_strength: f32,
    pub mirror: bool,
    /// Pull the background toward luma, 0.0..=1.0. Background plane only.
    pub desaturate: f32,
    /// Darken the background, 0.0..=1.0. Background plane only.
    pub dim: f32,
    /// Pull the alpha edge inward, 0.0..=0.9.
    ///
    /// Against a blurred copy of the same room a slightly generous silhouette is invisible,
    /// because what bleeds through is the same colours. Against a swapped background it is a
    /// bright fringe tracing shoulders and ears — the single most recognisable "bad virtual
    /// background" artifact, and the reason replace wants tighter morphology than blur.
    ///
    /// Note this *sharpens* as it erodes: the remap has gain `1/(1 - tighten)`, so 0.34 is
    /// a 51% steeper ramp. Reach for [`feather`] to soften, not for a smaller `tighten`.
    pub tighten: f32,
    /// Widen the alpha ramp, 0.0..=1.0, without moving where it crosses 0.5.
    ///
    /// The knob for "make the cut-out less like a sticker". `tighten` decides *where* the
    /// silhouette ends; this decides how abruptly. 0.0 is the historical behaviour exactly.
    pub feather: f32,
}

impl Default for Look {
    fn default() -> Self {
        Self {
            mode: BackgroundMode::Blur,
            blur_strength: 0.6,
            mirror: false,
            desaturate: 0.0,
            dim: 0.0,
            tighten: 0.0,
            feather: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GuidedParams {
    radius: i32,
    eps: f32,
}

/// Padded to 16 bytes because a uniform buffer binding has to be.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    offset: f32,
    _pad: [f32; 3],
}

/// How deep the blur pyramid can go.
///
/// Each level halves both dimensions, so the effective radius roughly doubles per level:
/// at 1080p, five levels reach 60x33 and the room behind you is unrecognisable, which is
/// what the top of the slider ought to mean.
///
/// This replaced a fixed *pass count* at a fixed resolution. That arrangement grew the
/// radius like sqrt(N) rather than 2^N, so the whole slider spanned about one level of the
/// blur this now reaches at 20%. It was also strictly more work: eight passes at half
/// resolution against a pyramid that costs half res plus a rapidly shrinking tail.
const MAX_BLUR_LEVELS: u32 = 5;

/// Smallest pyramid level worth building, per side.
///
/// Below roughly this the level is smaller than the compute workgroup and the up-sample
/// starts to alias visibly rather than smooth.
const MIN_BLUR_SIDE: u32 = 16;

pub struct FramePipeline {
    gpu: Gpu,
    width: u32,
    height: u32,

    // Full-resolution working textures.
    packed_in: wgpu::Texture,
    rgba: wgpu::Texture,
    composited: wgpu::Texture,
    packed_out: wgpu::Texture,

    /// The blur pyramid: `[0]` is half the frame, each subsequent level half again. The
    /// composite reads `[0]`, which is where the up chain lands.
    blur_levels: Vec<wgpu::Texture>,
    /// Tap-spacing scale, which is what makes the strength control continuous between
    /// levels. See [`MAX_BLUR_LEVELS`].
    blur_params: wgpu::Buffer,

    /// A 1x1 fully-opaque matte, used when no matting model is loaded so the composite
    /// pass is identical in both cases rather than branching on the host.
    matte: wgpu::Texture,

    /// The replacement background plate, already cover-fitted to the frame by the caller.
    /// 1x1 black until one is loaded, for the same reason as `matte`: the binding is
    /// referenced by live shader code, so it has to exist whether or not it is meaningful.
    bg_image: wgpu::Texture,
    /// Whether `bg_image` holds a real plate. `Replace` without one would key the subject
    /// onto flat black, which looks like a broken effect rather than an unset setting, so
    /// the host falls back to blur and says why.
    has_bg_image: bool,

    sampler: wgpu::Sampler,
    params: wgpu::Buffer,

    unpack: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    blur_down: wgpu::ComputePipeline,
    blur_up: wgpu::ComputePipeline,
    composite: wgpu::ComputePipeline,

    /// Small RGBA texture holding the frame at matting-input size, plus its readback.
    /// Sized by the caller so this crate does not need to know the network's geometry.
    matte_input: Option<(wgpu::Texture, wgpu::Buffer, u32, u32, u32)>,
    downscale: wgpu::ComputePipeline,

    /// Guided-filter coefficient field (a, b) at matte resolution, and its pass.
    /// `None` until a matte has been set, since there is nothing to filter before that.
    guided_ab: Option<wgpu::Texture>,
    guided: wgpu::ComputePipeline,
    guided_params: wgpu::Buffer,
    /// Whether `guided_ab` holds coefficients for the current matte.
    guided_ready: bool,
    /// Set by `set_matte`, cleared by `finish_frame` once it has refitted the coefficients.
    /// Without it a frame that reuses the previous matte would refit it for nothing.
    matte_dirty: bool,
    /// Live guided-filter settings, from `video.guided_*`. See [`FramePipeline::set_guided`].
    guided_enabled: bool,
    guided_radius: i32,
    guided_eps: f32,

    /// Staging buffer for the readback. Persistent so the hot path allocates nothing.
    readback: wgpu::Buffer,
    /// Bytes per row after alignment to wgpu's 256-byte requirement.
    padded_row: u32,
}

impl FramePipeline {
    pub fn new(gpu: Gpu, width: u32, height: u32) -> Self {
        // YUY2 carries two pixels per texel, so the packed textures are half width. An
        // odd width would lose the final column, so round up.
        let packed_w = width.div_ceil(2);
        let dev = &gpu.device;

        let make_tex = |label: &str, w: u32, h: u32, format: wgpu::TextureFormat, storage: bool| {
            let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
            if storage {
                usage |= wgpu::TextureUsages::STORAGE_BINDING;
            }
            usage |= wgpu::TextureUsages::COPY_SRC;
            dev.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };

        let packed_in = make_tex(
            "packed-in",
            packed_w,
            height,
            wgpu::TextureFormat::Rgba8Uint,
            false,
        );
        let rgba = make_tex("rgba", width, height, wgpu::TextureFormat::Rgba8Unorm, true);
        let composited = make_tex(
            "composited",
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            true,
        );
        let packed_out = make_tex(
            "packed-out",
            packed_w,
            height,
            wgpu::TextureFormat::Rgba8Uint,
            true,
        );
        // The pyramid, built once. Stops early on a small frame rather than creating
        // degenerate levels: at 320x180 there is no useful 10x5 level to make.
        let mut blur_levels = Vec::new();
        let (mut lw, mut lh) = (width / 2, height / 2);
        for _ in 0..MAX_BLUR_LEVELS {
            if lw < MIN_BLUR_SIDE || lh < MIN_BLUR_SIDE {
                break;
            }
            blur_levels.push(make_tex(
                "blur-level",
                lw,
                lh,
                wgpu::TextureFormat::Rgba8Unorm,
                true,
            ));
            lw /= 2;
            lh /= 2;
        }
        // A frame too small for even one level still has to composite, so guarantee one.
        if blur_levels.is_empty() {
            blur_levels.push(make_tex(
                "blur-level",
                (width / 2).max(1),
                (height / 2).max(1),
                wgpu::TextureFormat::Rgba8Unorm,
                true,
            ));
        }

        let blur_params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur-params"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Opaque 1x1 matte: alpha 1 everywhere means "all foreground", so a composite
        // with no model is a passthrough rather than a special case.
        let matte = dev.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("matte"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255u8],
        );

        // 1x1 black plate, for the same reason as the matte above: `comp_bg_image` is
        // referenced by live shader code and must always be bound, whether or not the user
        // has chosen an image. It is never actually sampled without one, because the host
        // refuses to select Replace until `has_bg_image`.
        let bg_image = dev.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("bg-image"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[0u8, 0, 0, 255],
        );

        let sampler = dev.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite-params"),
            size: std::mem::size_of::<CompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // wgpu requires readback rows to be a multiple of 256 bytes, so the staging
        // buffer is wider than the frame and the copy back out has to un-pad.
        let unpadded_row = packed_w * 4;
        let padded_row = unpadded_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let colour = include_str!("../shaders/colour.wgsl");
        let module = |label: &str, body: &str, with_colour: bool| {
            let src = if with_colour {
                format!("{colour}\n{body}")
            } else {
                body.to_string()
            };
            dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };

        let m_unpack = module(
            "yuy2->rgba",
            include_str!("../shaders/yuy2_to_rgba.wgsl"),
            true,
        );
        let m_pack = module(
            "rgba->yuy2",
            include_str!("../shaders/rgba_to_yuy2.wgsl"),
            true,
        );
        let m_blur = module("blur", include_str!("../shaders/blur.wgsl"), false);
        let m_comp = module(
            "composite",
            include_str!("../shaders/composite.wgsl"),
            false,
        );
        let m_down = module(
            "downscale",
            include_str!("../shaders/downscale.wgsl"),
            false,
        );
        let m_guided = module(
            "guided-ab",
            include_str!("../shaders/guided_ab.wgsl"),
            false,
        );

        let pipeline = |label: &str, m: &wgpu::ShaderModule, entry: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            width,
            height,
            unpack: pipeline("unpack", &m_unpack, "main"),
            pack: pipeline("pack", &m_pack, "main"),
            blur_down: pipeline("blur-down", &m_blur, "down"),
            blur_up: pipeline("blur-up", &m_blur, "up"),
            composite: pipeline("composite", &m_comp, "main"),
            downscale: pipeline("downscale", &m_down, "main"),
            guided: pipeline("guided-ab", &m_guided, "main"),
            guided_ab: None,
            guided_ready: false,
            matte_dirty: false,
            guided_enabled: true,
            guided_radius: GUIDED_RADIUS,
            guided_eps: GUIDED_EPS,
            guided_params: {
                let b = dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("guided-params"),
                    size: std::mem::size_of::<GuidedParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue.write_buffer(
                    &b,
                    0,
                    bytemuck::bytes_of(&GuidedParams {
                        radius: GUIDED_RADIUS,
                        eps: GUIDED_EPS,
                    }),
                );
                b
            },
            matte_input: None,
            packed_in,
            rgba,
            composited,
            packed_out,
            blur_levels,
            blur_params,
            matte,
            bg_image,
            has_bg_image: false,
            sampler,
            params,
            readback,
            padded_row,
            gpu,
        }
    }

    pub fn adapter_name(&self) -> &str {
        &self.gpu.choice.name
    }

    /// Upload the replacement background plate.
    ///
    /// `data` is tightly packed RGBA8 at `width` x `height`. The caller cover-fits it to
    /// the frame before calling, so this does no scaling of its own — the resampling is a
    /// once-per-image cost and belongs where the image is decoded, not on the GPU where it
    /// would either run per frame or need its own pass.
    ///
    /// Uploading once and keeping it is the point: a 1080p plate is 8 MB, and re-uploading
    /// that every frame would cost more PCIe traffic than the entire rest of the pipeline.
    pub fn set_background_image(&mut self, data: &[u8], width: u32, height: u32) {
        debug_assert_eq!(data.len(), (width as usize) * (height as usize) * 4);

        if self.bg_image.width() != width || self.bg_image.height() != height {
            self.bg_image = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("bg-image"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
        }
        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.bg_image,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.has_bg_image = true;
    }

    /// Whether a replacement plate has been uploaded.
    ///
    /// `Replace` without one would key the subject onto flat black, which reads as a broken
    /// effect rather than as an unset setting — so the caller checks this and degrades to
    /// blur, loudly, instead.
    pub fn has_background_image(&self) -> bool {
        self.has_bg_image
    }

    /// Forget the current plate, so `Replace` stops being honoured.
    pub fn clear_background_image(&mut self) {
        self.has_bg_image = false;
    }

    /// Replace the alpha matte.
    ///
    /// Call between [`begin_frame`] and [`finish_frame`], with a matte derived from the
    /// buffer `begin_frame` handed back. That is the whole contract: a matte set here
    /// applies to the frame currently in flight, not to the next one.
    ///
    /// `data` is single-channel R8, `alpha = 1` meaning foreground. It is normally much
    /// smaller than the frame — the matting network runs at 512x288 — and the composite
    /// pass samples it bilinearly, which is what keeps the edge smooth rather than blocky.
    ///
    /// Until a matting model is wired in, the default is a 1x1 opaque texel. That is not
    /// a placeholder to be special-cased: alpha=1 everywhere means the composite is
    /// exactly a passthrough, so the same code path serves both cases.
    pub fn set_matte(&mut self, data: &[u8], width: u32, height: u32) {
        if self.matte.width() != width || self.matte.height() != height {
            self.matte = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("matte"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
        }
        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.matte,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.matte_dirty = true;
    }

    /// Whether the guided filter can run against the matte currently uploaded.
    ///
    /// The guidance and the matte must be the same size for the window arithmetic to line
    /// up. They always are in practice (both INFER_W x INFER_H), so a mismatch is a caller
    /// bug rather than a case to paper over.
    fn guided_possible(&self) -> bool {
        if !self.guided_enabled {
            // Turned off in config. The composite falls back to sampling the matte directly.
            return false;
        }
        // No matting input configured means no guidance image, so there is nothing to fit.
        self.matte_input.as_ref().is_some_and(|(_, _, gw, gh, _)| {
            (*gw, *gh) == (self.matte.width(), self.matte.height())
        })
    }

    /// Allocate the coefficient field for the current matte size, if it is not already.
    fn ensure_guided_ab(&mut self) {
        let (width, height) = (self.matte.width(), self.matte.height());
        let needs_alloc = self
            .guided_ab
            .as_ref()
            .is_none_or(|t| t.width() != width || t.height() != height);
        if needs_alloc {
            // rgba16float: `a` is unbounded in principle and `b` is signed, so an 8-bit unorm
            // pair would clip both. Half floats are exact enough for coefficients that get
            // multiplied by a 0..1 luma.
            self.guided_ab = Some(self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("guided-ab"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }));
        }
    }

    /// Encode the guided-coefficient pass into the caller's encoder.
    ///
    /// Encoding rather than submitting is the point. The guidance image has to be the frame
    /// the matte was computed from, *and* the coefficients have to be consumed by the
    /// composite of that same frame. Putting both in one command buffer is what makes the
    /// second half true: `finish_frame` fits and uses them without a frame boundary in
    /// between, so there is no window in which `matte_input` could move on.
    ///
    /// Pairing a matte with the wrong guidance is subtle and ugly. `a` is large at an edge
    /// by design, so stale coefficients do not decay into soft ghosting — they extrapolate,
    /// and a moving edge picks up a luminance-keyed fringe.
    ///
    /// Caller guarantees [`guided_possible`] and that [`ensure_guided_ab`] has run.
    fn encode_guided(&self, enc: &mut wgpu::CommandEncoder) {
        let (width, height) = (self.matte.width(), self.matte.height());
        let guide = &self
            .matte_input
            .as_ref()
            .expect("guided_possible checked this")
            .0;

        let guide_view = guide.create_view(&Default::default());
        let matte_view = self.matte.create_view(&Default::default());
        let ab_view = self
            .guided_ab
            .as_ref()
            .expect("ensure_guided_ab ran")
            .create_view(&Default::default());

        // Written every time rather than only on change: this is 8 bytes against a pass that
        // reads millions of texels, and a uniform that silently lags the config by a frame is
        // the kind of bug that only shows up while somebody is dragging a slider.
        self.gpu.queue.write_buffer(
            &self.guided_params,
            0,
            bytemuck::bytes_of(&GuidedParams {
                radius: self.guided_radius,
                eps: self.guided_eps,
            }),
        );

        self.dispatch(
            enc,
            &self.guided,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&guide_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&matte_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ab_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.guided_params.as_entire_binding(),
                },
            ],
            width,
            height,
        );
    }

    /// Prepare a downscaled RGBA readback at `w`x`h`, for feeding a matting network.
    ///
    /// Separate from `new` so this crate never needs to know the network's input geometry,
    /// and so a pipeline with no matting model pays for neither the texture nor the buffer.
    pub fn enable_matte_input(&mut self, w: u32, h: u32) {
        let unpadded = w * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let tex = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("matte-input"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matte-input-readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.matte_input = Some((tex, buf, w, h, padded));
    }

    /// Upload a YUY2 frame, unpack it, and read back the matting network's input.
    ///
    /// The first half of a frame. Pair it with [`finish_frame`], optionally calling
    /// [`set_matte`] in between with a matte derived from the very buffer this returns.
    ///
    /// Returns false when no matte was asked for, or when [`enable_matte_input`] was never
    /// called — in both cases the frame is still uploaded and unpacked, so `finish_frame`
    /// composites it normally against whatever matte is already set.
    ///
    /// ## Why the split exists
    ///
    /// This used to be one `process` call that composited *and then* downscaled, so the
    /// caller could only ever infer a matte for a frame it had already sent to the virtual
    /// camera. The matte therefore landed on the *next* frame — 33 ms of misalignment
    /// between an image and its own alpha at 30 fps, which is plainly visible as a fringe
    /// dragging behind anything that moves.
    ///
    /// The old arrangement was justified as avoiding a pipeline stall. It did not: `process`
    /// already blocked on a readback and the downscale blocked on a second one, for three
    /// queue submissions a frame. This split has the same two stalls and two submissions,
    /// and the matte belongs to the frame it is composited onto.
    ///
    /// The readback is small on purpose: 512x288 RGBA is 590 KB, where reading back a full
    /// 1080p frame to downscale on the CPU would be 8 MB and put the scaling on the wrong
    /// processor.
    pub fn begin_frame(&mut self, input: &[u8], matte_in: Option<&mut [u8]>) -> bool {
        let packed_w = self.width.div_ceil(2);

        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.packed_in,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            input,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(packed_w * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: packed_w,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let v = |t: &wgpu::Texture| t.create_view(&Default::default());
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());

        // 1. unpack YUY2 -> RGBA, which is what every later pass reads.
        self.dispatch(
            &mut enc,
            &self.unpack,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&v(&self.packed_in)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&v(&self.rgba)),
                },
            ],
            packed_w,
            self.height,
        );

        // 2. downscale for the matting network, when one is wired up and wanted.
        let Some(out) = matte_in else {
            // Nothing to read back. Submit the unpack so `finish_frame` sees it, and skip
            // the stall entirely — with no matting this is now one submission and no wait,
            // where the old `process` always paid for a full round trip.
            self.gpu.queue.submit([enc.finish()]);
            return false;
        };
        // Taken and put back so `dispatch` can borrow `&self` while these are in hand.
        let Some((tex, buf, w, h, padded)) = self.matte_input.take() else {
            self.gpu.queue.submit([enc.finish()]);
            return false;
        };

        self.dispatch(
            &mut enc,
            &self.downscale,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&v(&self.rgba)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&v(&tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            w,
            h,
        );
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.gpu.queue.submit([enc.finish()]);

        {
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.gpu.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            if rx.recv().is_ok() {
                let data = slice.get_mapped_range();
                let row = (w * 4) as usize;
                for y in 0..h as usize {
                    let src = y * padded as usize;
                    let dst = y * row;
                    if dst + row <= out.len() && src + row <= data.len() {
                        out[dst..dst + row].copy_from_slice(&data[src..src + row]);
                    }
                }
            }
        }
        buf.unmap();

        self.matte_input = Some((tex, buf, w, h, padded));
        true
    }

    /// Configure the guided-filter upsample. Takes effect on the next `set_matte`.
    ///
    /// `enabled = false` falls back to sampling the matte directly, which is faster and
    /// visibly worse: bilinear does not know where the subject ends.
    pub fn set_guided(&mut self, enabled: bool, radius: u32, eps: f32) {
        // Clamped rather than trusted: radius feeds a `(2r+1)^2` inner loop in a compute
        // shader, so a typo'd 500 in a config file is a hung GPU, not a slow one.
        self.guided_enabled = enabled;
        self.guided_radius = radius.clamp(1, 16) as i32;
        self.guided_eps = eps.max(1e-8);
    }

    /// Upload, composite and read back in one call, for callers with no matting of their own.
    ///
    /// Exactly `begin_frame(input, None)` then `finish_frame`. The matte is whatever was
    /// last handed to [`set_matte`], which is fine here and emphatically not fine in the
    /// daemon — hence the split. Note that the old bug cannot be written through this
    /// entry point any more: there is no longer a way to ask for the frame back after
    /// compositing it, so "composite now, infer later" has no way to spell itself.
    pub fn process(&mut self, input: &[u8], output: &mut [u8], look: Look) {
        self.begin_frame(input, None);
        self.finish_frame(output, look);
    }

    /// Composite the frame [`begin_frame`] uploaded, and read the result back as YUY2.
    ///
    /// The second half of a frame. Deliberately takes no image: the frame is whatever
    /// `begin_frame` put in `rgba`, so there is no way for a caller to composite one frame
    /// with another frame's matte. That used to be exactly the bug — the ordering was the
    /// caller's to get right, and it got it wrong by one frame.
    pub fn finish_frame(&mut self, output: &mut [u8], look: Look) {
        let (mode, blur_strength, mirror) = (look.mode, look.blur_strength, look.mirror);
        let packed_w = self.width.div_ceil(2);

        // Refit the guided coefficients whenever a new matte has arrived. Both the fit and
        // its use land in this frame's command buffer, below.
        if self.matte_dirty {
            self.guided_ready = self.guided_possible();
            if self.guided_ready {
                self.ensure_guided_ab();
            }
            self.matte_dirty = false;
        }

        let dev = &self.gpu.device;
        let params = CompositeParams {
            mode: match mode {
                BackgroundMode::Off => 0,
                BackgroundMode::Blur => 1,
                BackgroundMode::Replace => 2,
                BackgroundMode::Remove => 3,
            },
            mirror: mirror as u32,
            desaturate: look.desaturate.clamp(0.0, 1.0),
            dim: look.dim.clamp(0.0, 1.0),
            guided: self.guided_ready as u32,
            tighten: look.tighten.clamp(0.0, 0.9),
            feather: look.feather.clamp(0.0, 1.0),
            _pad0: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let v = |t: &wgpu::Texture| t.create_view(&Default::default());
        let mut enc = dev.create_command_encoder(&Default::default());

        // 1. guided-filter coefficients, fitted on the frame this matte came from and
        //    consumed by the composite two passes below — same frame, same command buffer.
        if self.guided_ready {
            self.encode_guided(&mut enc);
        }

        // 2. blur pyramid, only when the mode needs a background plane
        //
        // The ping-pong bookkeeping is tracked explicitly rather than derived from the
        // pass count. An earlier version computed which texture held the result from
        // `passes % 2` and got it wrong by one, so the composite read the *input* to the
        // final pass — the blur ran correctly and was then thrown away, showing up as a
        // barely-there 7% variance drop instead of a real blur.
        if matches!(mode, BackgroundMode::Blur) {
            // Continuous, not stepped. The old mapping truncated 0..1 onto four integer
            // pass counts, so two-thirds of the slider's travel changed nothing at all and
            // the top of the range was reachable only at exactly 1.0. Here the integer part
            // picks a pyramid depth and the fraction slides the tap spacing from 1.0 to 2.0
            // within that depth — and 2.0 is what the next level down does at 1.0, so the
            // two controls meet without a step.
            let depth = self.blur_levels.len() as f32;
            let pos = (blur_strength.clamp(0.0, 1.0) * depth).min(depth - 1e-4);
            let levels = pos.floor() as usize + 1;
            let offset = 1.0 + pos.fract();

            self.gpu.queue.write_buffer(
                &self.blur_params,
                0,
                bytemuck::bytes_of(&BlurParams {
                    offset,
                    _pad: [0.0; 3],
                }),
            );

            // Down: full frame into level 0, then each level into the next one down. The
            // halving is what makes the radius grow geometrically.
            let l0 = &self.blur_levels[0];
            self.blur_pass(
                &mut enc,
                &self.blur_down,
                &self.rgba,
                l0,
                l0.width(),
                l0.height(),
            );
            for i in 1..levels {
                let (src, dst) = (&self.blur_levels[i - 1], &self.blur_levels[i]);
                self.blur_pass(
                    &mut enc,
                    &self.blur_down,
                    src,
                    dst,
                    dst.width(),
                    dst.height(),
                );
            }
            // Up: back along the same chain, each level reading the smaller one below it.
            // Overwriting the down result at each level is correct — it has already been
            // consumed by the level below, and the up tap pattern is what does the
            // reconstruction.
            for i in (0..levels.saturating_sub(1)).rev() {
                let (src, dst) = (&self.blur_levels[i + 1], &self.blur_levels[i]);
                self.blur_pass(&mut enc, &self.blur_up, src, dst, dst.width(), dst.height());
            }
        }

        // The up chain always lands in level 0.
        let bg = if matches!(mode, BackgroundMode::Blur) {
            &self.blur_levels[0]
        } else {
            &self.rgba
        };

        // 3. composite
        self.dispatch(
            &mut enc,
            &self.composite,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&v(&self.rgba)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&v(bg)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&v(&self.matte)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&v(&self.composited)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&v(&self.bg_image)),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    // Falls back to the matte's own view when no coefficients exist yet.
                    // The shader will not read it (guided == 0), but the binding still has
                    // to be filled with something of the right dimension.
                    resource: wgpu::BindingResource::TextureView(&v(self
                        .guided_ab
                        .as_ref()
                        .unwrap_or(&self.matte))),
                },
            ],
            self.width,
            self.height,
        );

        // 4. pack
        self.dispatch(
            &mut enc,
            &self.pack,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&v(&self.composited)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&v(&self.packed_out)),
                },
            ],
            packed_w,
            self.height,
        );

        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.packed_out,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: packed_w,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.gpu.queue.submit([enc.finish()]);

        // Read back, un-padding each row.
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.gpu.device.poll(wgpu::PollType::Wait {
            // Wait for the most recent submission — the one we just made.
            submission_index: None,
            timeout: None,
        });
        if rx.recv().is_ok() {
            let data = slice.get_mapped_range();
            let row_bytes = (packed_w * 4) as usize;
            for y in 0..self.height as usize {
                let src = y * self.padded_row as usize;
                let dst = y * row_bytes;
                if dst + row_bytes <= output.len() && src + row_bytes <= data.len() {
                    output[dst..dst + row_bytes].copy_from_slice(&data[src..src + row_bytes]);
                }
            }
        }
        self.readback.unmap();
    }

    fn dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        entries: &[wgpu::BindGroupEntry],
        w: u32,
        h: u32,
    ) {
        let bg = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        // Workgroups are 8x8, matching the @workgroup_size in every kernel.
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }

    fn blur_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        w: u32,
        h: u32,
    ) {
        let v = |t: &wgpu::Texture| t.create_view(&Default::default());
        self.dispatch(
            enc,
            pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&v(src)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&v(dst)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.blur_params.as_entire_binding(),
                },
            ],
            w,
            h,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 64;
    const H: u32 = 32;

    /// Luma of white and of the `Remove` key colour, as this pipeline writes them.
    ///
    /// `Remove` composites pure green behind the subject, so reading luma back is enough to
    /// say where alpha was 1 and where it was 0 — 235 against 145 is not a distinction any
    /// tolerance can blur.
    const FG_WHITE: i32 = 235;
    const BG_GREEN: i32 = 145;

    /// Build a YUY2 frame from a per-pixel RGB function, via the same BT.601 limited-range
    /// matrix `colour.wgsl` uses. Generating YUV directly would ask for colours outside the
    /// RGB gamut, which the shader clamps — correctly — and the test then reads as a bug.
    fn yuy2_from(rgb: impl Fn(u32) -> (f32, f32, f32)) -> Vec<u8> {
        let pw = (W / 2) as usize;
        let mut out = vec![0u8; pw * H as usize * 4];
        let to_yuv = |(r, g, b): (f32, f32, f32)| {
            (
                0.2568 * r + 0.5041 * g + 0.0979 * b + 0.0627451,
                -0.1482 * r - 0.2910 * g + 0.4392 * b + 0.5019608,
                0.4392 * r - 0.3678 * g - 0.0714 * b + 0.5019608,
            )
        };
        for y in 0..H as usize {
            for xq in 0..pw {
                let x = xq as u32 * 2;
                let (y0, u0, v0) = to_yuv(rgb(x));
                let (y1, u1, v1) = to_yuv(rgb(x + 1));
                let i = (y * pw + xq) * 4;
                out[i] = (y0 * 255.0 + 0.5) as u8;
                out[i + 1] = ((u0 + u1) * 0.5 * 255.0 + 0.5) as u8;
                out[i + 2] = (y1 * 255.0 + 0.5) as u8;
                out[i + 3] = ((v0 + v1) * 0.5 * 255.0 + 0.5) as u8;
            }
        }
        out
    }

    /// Luma of pixel `x` on row `y` of a packed YUY2 buffer.
    fn luma(buf: &[u8], x: u32, y: u32) -> i32 {
        let pw = (W / 2) as usize;
        buf[(y as usize * pw + (x / 2) as usize) * 4 + (x % 2) as usize * 2] as i32
    }

    /// A frame that is white on one half and black on the other.
    fn split_frame(white_left: bool) -> Vec<u8> {
        yuy2_from(|x| {
            if (x < W / 2) == white_left {
                (1.0, 1.0, 1.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        })
    }

    /// A matte at `mw`x`mh` that is opaque over the same half as `split_frame`.
    fn split_matte(white_left: bool, mw: u32, mh: u32) -> Vec<u8> {
        (0..mw * mh)
            .map(|i| {
                if ((i % mw) < mw / 2) == white_left {
                    255
                } else {
                    0
                }
            })
            .collect()
    }

    /// Every pixel outside a margin around the centre seam must be foreground on the white
    /// half and keyed out on the black half. The margin exists because the matte is sampled
    /// bilinearly, so the two texels straddling the boundary are legitimately in between.
    fn assert_keyed(out: &[u8], white_left: bool, margin: u32, ctx: &str) {
        for y in (0..H).step_by(7) {
            for x in 0..W {
                if x.abs_diff(W / 2) <= margin || x < margin || x >= W - margin {
                    continue;
                }
                let want = if (x < W / 2) == white_left {
                    FG_WHITE
                } else {
                    BG_GREEN
                };
                let got = luma(out, x, y);
                assert!(
                    (got - want).abs() <= 6,
                    "{ctx}: pixel ({x},{y}) reads {got}, wanted {want} \
                     ({}). The matte belongs to a different frame than the image.",
                    if want == FG_WHITE {
                        "subject, should be white"
                    } else {
                        "background, should be keyed green"
                    }
                );
            }
        }
    }

    fn new_pipeline() -> Option<FramePipeline> {
        match Gpu::new(None) {
            Ok(g) => Some(FramePipeline::new(g, W, H)),
            Err(e) => {
                eprintln!("no GPU available ({e}); skipping");
                None
            }
        }
    }

    fn remove() -> Look {
        Look {
            mode: BackgroundMode::Remove,
            ..Default::default()
        }
    }

    /// The invariant this whole reordering exists to establish.
    ///
    /// A matte handed over between `begin_frame` and `finish_frame` must apply to *that*
    /// frame. The second pass is the one that matters: it flips both the image and the
    /// matte, so a pipeline that held either one from the previous call composites a
    /// subject where there is now background and fails loudly.
    ///
    /// The bug this guards against was not in this file — `FramePipeline` faithfully
    /// applied whatever matte was set — it was in the daemon's *ordering*, which
    /// composited before inferring. `finish_frame` takes no frame argument precisely so
    /// that ordering is no longer something a caller can get wrong.
    #[test]
    fn the_matte_applies_to_the_frame_it_was_derived_from() {
        let Some(mut pipe) = new_pipeline() else {
            return;
        };
        // Sample the matte directly, so this test is about *which frame* the alpha came
        // from and not about the guided filter's reconstruction.
        pipe.set_guided(false, 3, 1e-4);

        let mut out = vec![0u8; (W * H * 2) as usize];
        for white_left in [true, false] {
            pipe.begin_frame(&split_frame(white_left), None);
            pipe.set_matte(&split_matte(white_left, W, H), W, H);
            pipe.finish_frame(&mut out, remove());
            assert_keyed(
                &out,
                white_left,
                2,
                if white_left { "first frame" } else { "flipped" },
            );
        }
    }

    /// The same invariant for the guided path, which is where a stale matte does the real
    /// damage.
    ///
    /// The composite evaluates `alpha = a*I + b` against the frame's own luma. `a` is large
    /// at an edge by design, so coefficients fitted on the previous frame do not decay into
    /// soft ghosting — they extrapolate, and a moving edge picks up a luminance-keyed
    /// fringe. Fitting and consuming them within one `begin_frame`/`finish_frame` pair is
    /// what keeps that from being possible.
    #[test]
    fn the_guided_coefficients_belong_to_the_frame_they_are_used_on() {
        let Some(mut pipe) = new_pipeline() else {
            return;
        };
        let (mw, mh) = (W / 2, H / 2);
        pipe.set_guided(true, 3, 1e-4);
        pipe.enable_matte_input(mw, mh);

        let mut matte_in = vec![0u8; (mw * mh * 4) as usize];
        let mut out = vec![0u8; (W * H * 2) as usize];

        for white_left in [true, false] {
            assert!(
                pipe.begin_frame(&split_frame(white_left), Some(&mut matte_in)),
                "matte input was enabled, so the readback must happen"
            );
            pipe.set_matte(&split_matte(white_left, mw, mh), mw, mh);
            pipe.finish_frame(&mut out, remove());
            // A wider margin than the direct-sample test: the guided window is 7 matte
            // texels, which is 14 frame pixels here, so the seam is legitimately soft.
            assert_keyed(
                &out,
                white_left,
                8,
                if white_left {
                    "guided, first frame"
                } else {
                    "guided, flipped"
                },
            );
        }
    }

    /// `begin_frame` must hand back the frame it was just given, not the one before it.
    ///
    /// This is what the daemon feeds to the matting network, so an off-by-one here would
    /// reintroduce exactly the misalignment the reordering removes — silently, because a
    /// network fed the previous frame still returns a perfectly plausible matte.
    #[test]
    fn the_matte_input_is_the_frame_just_uploaded() {
        let Some(mut pipe) = new_pipeline() else {
            return;
        };
        let (mw, mh) = (W / 2, H / 2);
        pipe.enable_matte_input(mw, mh);
        let mut matte_in = vec![0u8; (mw * mh * 4) as usize];

        for white_left in [true, false] {
            pipe.begin_frame(&split_frame(white_left), Some(&mut matte_in));
            // Sample well inside each half, away from the bilinear seam.
            let at = |x: u32| matte_in[((mh / 2) * mw + x) as usize * 4] as i32;
            let (left, right) = (at(mw / 8), at(mw - mw / 8));
            let (bright, dark) = if white_left {
                (left, right)
            } else {
                (right, left)
            };
            assert!(
                bright > 200 && dark < 55,
                "downscaled frame reads {bright}/{dark} for the white/black halves \
                 (white_left = {white_left}); the readback is a frame behind"
            );
        }
    }
}
