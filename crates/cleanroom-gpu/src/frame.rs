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
    feather_px: f32,
    _pad0: u32,
}

/// Feather radius at 1080p, in full-resolution pixels, for `feather = 1.0`.
///
/// Scaled by frame height at use, so 720p and 1080p soften the same *fraction* of the
/// picture rather than the same pixel count — a 12 px ramp is soft at 1080p and mush at
/// 480p. 12 px is about three matte pixels at the default 512x288 inference size, which is
/// roughly where an edge stops reading as cut out with scissors and starts reading as mist.
const FEATHER_MAX_PX_AT_1080P: f32 = 12.0;

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
    /// The knob for "make the cut-out less like a sticker". [`tighten`](Self::tighten)
    /// decides *where* the silhouette ends; this decides how abruptly. 0.0 leaves the alpha
    /// untouched, so a config that never sets it composites exactly as before.
    ///
    /// Implemented as a spatial average of the alpha over a disc whose radius scales with
    /// this value — see [`FEATHER_MAX_PX_AT_1080P`]. It has to be spatial: an earlier
    /// version remapped alpha *values* through a widening `smoothstep`, which cannot widen
    /// an edge, because the transition still lands on the same pixels. It also saturated
    /// immediately, so the whole slider had two states.
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

/// Storage format for the blur pyramid.
///
/// Half floats rather than 8-bit unorm, and the reason is the weighting. The pyramid carries
/// the background *premultiplied* by how much of each tap was background, and the composite
/// divides that weight back out — so quantisation error is amplified by `1/w`. An isolated
/// background pixel in a narrow gap, say between an arm and the torso, can sit at `w ≈ 0.05`,
/// where 8-bit would recover with about 0.04 of error: roughly ten levels of 255, and plainly
/// visible as banding across a smoothly blurred region.
///
/// The cost is about 2.7 MB more across the pyramid, some 330 MB/s at 30 fps, which is
/// nothing against even an iGPU's bandwidth. `guided_ab` already writes and linearly samples
/// this format, including under lavapipe in CI, so it is known to work on this stack.
const BLUR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

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
    /// Level 0 only: the same kernel, weighted so the subject is kept out of its own blur.
    blur_down_weighted: wgpu::ComputePipeline,
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
            blur_levels.push(make_tex("blur-level", lw, lh, BLUR_FORMAT, true));
            lw /= 2;
            lh /= 2;
        }
        // A frame too small for even one level still has to composite, so guarantee one.
        if blur_levels.is_empty() {
            blur_levels.push(make_tex(
                "blur-level",
                (width / 2).max(1),
                (height / 2).max(1),
                BLUR_FORMAT,
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
            blur_down_weighted: pipeline("blur-down-weighted", &m_blur, "down_weighted"),
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
            feather_px: look.feather.clamp(0.0, 1.0)
                * FEATHER_MAX_PX_AT_1080P
                * (self.height as f32 / 1080.0),
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
            // Level 0 is the only pass that reads the full-resolution frame, and therefore
            // the only place the matte can be applied. Everything below it inherits the
            // weighting for free, because the remaining passes sum `vec4` linearly.
            let l0 = &self.blur_levels[0];
            self.blur_pass_weighted(&mut enc, &self.rgba, l0, l0.width(), l0.height());
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

    /// Level 0 of the down chain, with the matte bound so the subject can be weighted out.
    ///
    /// Separate from [`blur_pass`](Self::blur_pass) rather than a flag on it because the bind
    /// group layouts genuinely differ — this entry point reads a fifth binding — and wgpu
    /// derives the layout per entry point, so the two cannot share one helper.
    fn blur_pass_weighted(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        w: u32,
        h: u32,
    ) {
        let v = |t: &wgpu::Texture| t.create_view(&Default::default());
        self.dispatch(
            enc,
            &self.blur_down_weighted,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&v(&self.matte)),
                },
            ],
            w,
            h,
        );
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
    fn yuy2_from(w: u32, h: u32, rgb: impl Fn(u32) -> (f32, f32, f32)) -> Vec<u8> {
        let pw = (w / 2) as usize;
        let mut out = vec![0u8; pw * h as usize * 4];
        let to_yuv = |(r, g, b): (f32, f32, f32)| {
            (
                0.2568 * r + 0.5041 * g + 0.0979 * b + 0.0627451,
                -0.1482 * r - 0.2910 * g + 0.4392 * b + 0.5019608,
                0.4392 * r - 0.3678 * g - 0.0714 * b + 0.5019608,
            )
        };
        for y in 0..h as usize {
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

    /// Luma of pixel `x` on row `y` of a packed YUY2 buffer `w` pixels wide.
    fn luma_at(buf: &[u8], w: u32, x: u32, y: u32) -> i32 {
        let pw = (w / 2) as usize;
        buf[(y as usize * pw + (x / 2) as usize) * 4 + (x % 2) as usize * 2] as i32
    }

    /// Luma at this module's default frame size.
    fn luma(buf: &[u8], x: u32, y: u32) -> i32 {
        luma_at(buf, W, x, y)
    }

    /// A frame that is white on one half and black on the other, at an arbitrary size.
    fn split_frame_at(w: u32, h: u32, white_left: bool) -> Vec<u8> {
        yuy2_from(w, h, |x| {
            if (x < w / 2) == white_left {
                (1.0, 1.0, 1.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        })
    }

    /// A split frame at this module's default size.
    fn split_frame(white_left: bool) -> Vec<u8> {
        split_frame_at(W, H, white_left)
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

    // --- edge feather -----------------------------------------------------------------
    //
    // These run on a larger frame than the tests above, and not for realism: the feather
    // radius scales with frame height (12 px at 1080p), so on this module's 32-row frame the
    // widest possible radius is a third of a pixel and there would be nothing to measure.

    const FW: u32 = 640;
    const FH: u32 = 360;

    /// The band of pixels around the centre seam where the two planes are being mixed.
    ///
    /// Measured from composited luma rather than from alpha, because that is what anyone
    /// actually sees. `Remove` keys the background to flat green, so any luma strictly
    /// between the two plateaux is a pixel whose alpha is neither 0 nor 1.
    fn transition_band(out: &[u8], w: u32, y: u32) -> Option<(u32, u32)> {
        let (lo, hi) = (BG_GREEN + 3, FG_WHITE - 3);
        let (mut first, mut last) = (None, None);
        for x in 0..w {
            let v = luma_at(out, w, x, y);
            if v > lo && v < hi {
                first.get_or_insert(x);
                last = Some(x);
            }
        }
        first.zip(last)
    }

    fn transition_width(out: &[u8], w: u32, y: u32) -> u32 {
        transition_band(out, w, y).map_or(0, |(a, b)| b - a + 1)
    }

    /// Composite a hard-edged split matte at `feather`/`tighten` and hand back the frame.
    ///
    /// The *frame* is uniform white and only the matte splits, which is the whole point.
    /// A white/black split frame — what the tests above use — puts a colour edge in the same
    /// place as the alpha edge, and `Remove` then mixes flat green against black on one side,
    /// so the composited luma dips *below* the background plateau instead of ramping between
    /// the two. Measured, that reads as 97 in the middle of a 145..235 ramp, and any
    /// threshold-based band detector loses half the transition. Uniform white makes luma a
    /// direct, monotone readout of alpha: 235 at alpha 1, 145 at alpha 0.
    fn feathered(pipe: &mut FramePipeline, feather: f32, tighten: f32) -> Vec<u8> {
        let mut out = vec![0u8; (FW * FH * 2) as usize];
        pipe.begin_frame(&yuy2_from(FW, FH, |_| (1.0, 1.0, 1.0)), None);
        pipe.set_matte(&split_matte(true, FW, FH), FW, FH);
        pipe.finish_frame(
            &mut out,
            Look {
                mode: BackgroundMode::Remove,
                feather,
                tighten,
                ..Default::default()
            },
        );
        out
    }

    /// A pipeline big enough for the edge and blur tests, with the guided filter off.
    ///
    /// Larger than this module's default frame on purpose: the feather radius scales with
    /// frame height, so at 32 rows the widest possible radius is a third of a pixel.
    fn wide_pipeline() -> Option<FramePipeline> {
        let gpu = match Gpu::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU available ({e}); skipping");
                return None;
            }
        };
        let mut pipe = FramePipeline::new(gpu, FW, FH);
        // Sample the matte directly. These tests are about the spatial average, not about
        // the guided filter's reconstruction.
        pipe.set_guided(false, 3, 1e-4);
        Some(pipe)
    }

    /// The regression this rewrite exists for.
    ///
    /// The old implementation remapped alpha *values* through `smoothstep(c - w, c + w, a)`
    /// with `w = clamp((1 - tighten) * 0.5 * (1 + 3 * feather), 0.01, 0.5)`. At `tighten = 0`
    /// — the default for blur — `(1 - 0) * 0.5` already sat on the clamp ceiling, so every
    /// non-zero feather produced the identical curve. Measured across the slider, `w` was
    /// 0.5 at feather 0.05, 0.1, 0.5 and 1.0 alike: the control had two states.
    ///
    /// Nothing caught it, because the only tests on feather checked that it round-trips
    /// through the config and does not restart the pipeline. This measures the thing the
    /// knob claims to do.
    #[test]
    fn feather_widens_the_edge_monotonically() {
        let Some(mut pipe) = wide_pipeline() else {
            return;
        };

        let mut widths = Vec::new();
        for feather in [0.0f32, 0.25, 0.5, 1.0] {
            let out = feathered(&mut pipe, feather, 0.0);
            let width = transition_width(&out, FW, FH / 2);
            widths.push(width);
            // The profile itself, not just its width: a ramp that is wide but takes only two
            // or three distinct values reads as banding rather than as a soft edge, and the
            // width alone cannot tell the two apart. This is what caught the first attempt,
            // a hexagonal ring, producing four coarse steps instead of a ramp.
            let profile: Vec<i32> = (FW / 2 - 10..FW / 2 + 10)
                .map(|x| luma_at(&out, FW, x, FH / 2))
                .collect();
            eprintln!("  feather {feather:<4}  width {width:>2}  {profile:?}");
        }

        for pair in widths.windows(2) {
            assert!(
                pair[1] > pair[0],
                "feather must widen the edge at every step, got {widths:?}; \
                 a flat run means it has saturated, which is the original bug"
            );
        }

        // Pins the scaling constant as well as the direction. At 360 rows the widest radius
        // is 12 * 360/1080 = 4 px, so the ring spans about 8 px either side of the seam.
        // A 1-2 px result means FEATHER_MAX_PX_AT_1080P was dropped; 80 means the height
        // scaling is off by an order of magnitude.
        let widest = *widths.last().expect("four measurements");
        assert!(
            (5..=14).contains(&widest),
            "feather = 1.0 gave a {widest} px transition; expected roughly 8 px at {FH} rows"
        );
    }

    /// Zero must mean *nothing*, or every existing config silently changes look.
    #[test]
    fn feather_zero_leaves_the_edge_hard() {
        let Some(mut pipe) = wide_pipeline() else {
            return;
        };
        let out = feathered(&mut pipe, 0.0, 0.0);
        let width = transition_width(&out, FW, FH / 2);
        assert!(
            width <= 1,
            "feather = 0 produced a {width} px ramp; the spatial average must collapse to \
             the centre tap so an unfeathered config composites exactly as before"
        );
    }

    /// The two edge controls must finally be independent, which is what the docs claim.
    ///
    /// `tighten` moves where alpha crosses 0.5 and — being a shift-and-rescale with gain
    /// `1/(1 - t)` — steepens the ramp as it does so. Under the old value-remap that
    /// steepening fought feather directly, and `tighten` also fed feather's own width term,
    /// so the two were tangled. Now feather works in screen space: raising tighten must move
    /// the edge without collapsing the softening back to nothing.
    #[test]
    fn tighten_moves_the_edge_without_undoing_feather() {
        let Some(mut pipe) = wide_pipeline() else {
            return;
        };

        let hard = transition_width(&feathered(&mut pipe, 0.0, 0.0), FW, FH / 2);

        let soft = feathered(&mut pipe, 1.0, 0.0);
        let tight = feathered(&mut pipe, 1.0, 0.3);

        let (a0, b0) = transition_band(&soft, FW, FH / 2).expect("a feathered edge has a band");
        let (a1, b1) = transition_band(&tight, FW, FH / 2).expect("still a band once tightened");
        let (mid_soft, mid_tight) = ((a0 + b0) as f32 / 2.0, (a1 + b1) as f32 / 2.0);

        // The subject is on the left, so eroding it moves the crossing left.
        assert!(
            mid_tight < mid_soft,
            "tighten must pull the silhouette inward: crossing sat at {mid_soft} and moved \
             to {mid_tight}"
        );
        assert!(
            (b1 - a1 + 1) > hard + 2,
            "tighten collapsed the feather back to a hard edge ({} px against {hard} px \
             unfeathered); the two controls are supposed to be independent",
            b1 - a1 + 1
        );
    }

    // --- the background blur must not contain the subject -------------------------------

    /// Luma of a flat RGB level once it has been through this pipeline's colour conversion.
    fn expected_luma(level: f32) -> i32 {
        (((0.2568 + 0.5041 + 0.0979) * level + 0.0627451) * 255.0).round() as i32
    }

    /// The reported symptom, as a test.
    ///
    /// The blur pyramid used to be built from the whole frame, subject included, so a
    /// low-frequency copy of the subject lived inside their own background. Standing still it
    /// is invisible; under quick movement the smear slides around behind them, and their
    /// colours halo out around the silhouette.
    ///
    /// The frame here is white on the subject's half and mid-grey on the background's. If any
    /// of the subject reaches the background plane, the grey is pulled measurably toward
    /// white — at this blur strength the whole-frame average would land near luma 180 against
    /// a true background of 126, which is not a difference any tolerance can hide.
    #[test]
    fn the_blur_source_excludes_the_subject() {
        let Some(mut pipe) = wide_pipeline() else {
            return;
        };

        const SUBJECT: f32 = 1.0;
        const BACKDROP: f32 = 0.5;

        // Subject on the left, backdrop on the right, with the matte agreeing.
        let frame = yuy2_from(FW, FH, |x| {
            let v = if x < FW / 2 { SUBJECT } else { BACKDROP };
            (v, v, v)
        });
        let matte = split_matte(true, FW, FH);

        let mut out = vec![0u8; (FW * FH * 2) as usize];
        pipe.begin_frame(&frame, None);
        pipe.set_matte(&matte, FW, FH);
        pipe.finish_frame(
            &mut out,
            Look {
                mode: BackgroundMode::Blur,
                blur_strength: 1.0,
                ..Default::default()
            },
        );

        let want = expected_luma(BACKDROP);
        let mut worst = (0u32, want);
        for x in (FW / 2 + 4..FW - 4).step_by(8) {
            let got = luma_at(&out, FW, x, FH / 2);
            if (got - want).abs() > (worst.1 - want).abs() {
                worst = (x, got);
            }
        }

        eprintln!(
            "background luma: wanted {want}, worst {} at x={} (subject is luma {})",
            worst.1,
            worst.0,
            expected_luma(SUBJECT)
        );
        assert!(
            (worst.1 - want).abs() <= 4,
            "background reads {} at x={}, against a true backdrop of {want}: the subject is \
             bleeding into its own blurred background",
            worst.1,
            worst.0
        );
    }

    /// The degenerate case the weight division has to survive.
    ///
    /// With no matting model loaded the matte is a single opaque texel, so every tap in the
    /// pyramid has zero background weight and the composite's divide falls onto its `max()`
    /// floor. That is fine only because alpha is 1 everywhere, so the foreground is taken
    /// whole and the meaningless background is never sampled. This asserts that rather than
    /// leaving it as an argument in a comment — a regression here would show as the whole
    /// frame going black the moment the model failed to load.
    #[test]
    fn an_all_foreground_matte_still_composites_the_subject() {
        let Some(mut pipe) = wide_pipeline() else {
            return;
        };

        let frame = yuy2_from(FW, FH, |x| {
            let v = if x < FW / 2 { 1.0 } else { 0.5 };
            (v, v, v)
        });
        pipe.set_matte(&[255u8], 1, 1);

        let mut passthrough = vec![0u8; (FW * FH * 2) as usize];
        let mut blurred = vec![0u8; (FW * FH * 2) as usize];
        pipe.process(
            &frame,
            &mut passthrough,
            Look {
                mode: BackgroundMode::Off,
                ..Default::default()
            },
        );
        pipe.process(
            &frame,
            &mut blurred,
            Look {
                mode: BackgroundMode::Blur,
                blur_strength: 1.0,
                ..Default::default()
            },
        );

        for x in (0..FW).step_by(16) {
            let (a, b) = (
                luma_at(&passthrough, FW, x, FH / 2),
                luma_at(&blurred, FW, x, FH / 2),
            );
            assert!(
                (a - b).abs() <= 2,
                "an all-foreground matte must composite as a passthrough, but x={x} reads \
                 {b} against {a} — the zero-weight background is leaking through"
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
