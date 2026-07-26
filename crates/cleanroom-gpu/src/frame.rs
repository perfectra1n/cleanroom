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
}

/// How many down/up pairs the blur runs.
///
/// Each pair roughly doubles the effective radius at constant cost per pass, so this is a
/// geometric knob: 4 iterations is a heavily blurred room, 1 is a gentle softening.
const MAX_BLUR_PASSES: u32 = 4;

pub struct FramePipeline {
    gpu: Gpu,
    width: u32,
    height: u32,

    // Full-resolution working textures.
    packed_in: wgpu::Texture,
    rgba: wgpu::Texture,
    composited: wgpu::Texture,
    packed_out: wgpu::Texture,

    /// Half-resolution ping-pong pair for the blur pyramid.
    blur_a: wgpu::Texture,
    blur_b: wgpu::Texture,

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
        let blur_a = make_tex(
            "blur-a",
            width / 2,
            height / 2,
            wgpu::TextureFormat::Rgba8Unorm,
            true,
        );
        let blur_b = make_tex(
            "blur-b",
            width / 2,
            height / 2,
            wgpu::TextureFormat::Rgba8Unorm,
            true,
        );

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
            matte_input: None,
            packed_in,
            rgba,
            composited,
            packed_out,
            blur_a,
            blur_b,
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

    /// Downscale the current frame and read it back as tightly-packed RGBA8.
    ///
    /// Call after [`process`], which is what leaves the unpacked RGBA in place. Returns
    /// false if [`enable_matte_input`] was never called.
    ///
    /// The readback is small on purpose: 512x288 RGBA is 590 KB, where reading back a full
    /// 1080p frame to downscale on the CPU would be 8 MB and put the scaling on the wrong
    /// processor.
    pub fn read_matte_input(&mut self, out: &mut [u8]) -> bool {
        let Some((tex, buf, w, h, padded)) = self.matte_input.take() else {
            return false;
        };

        let v = |t: &wgpu::Texture| t.create_view(&Default::default());
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
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

    /// Run one frame: upload YUY2, apply effects, read YUY2 back.
    pub fn process(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        mode: BackgroundMode,
        blur_strength: f32,
        mirror: bool,
    ) {
        let packed_w = self.width.div_ceil(2);
        let dev = &self.gpu.device;

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

        let params = CompositeParams {
            mode: match mode {
                BackgroundMode::Off => 0,
                BackgroundMode::Blur => 1,
                BackgroundMode::Replace => 2,
                BackgroundMode::Remove => 3,
            },
            mirror: mirror as u32,
            desaturate: 0.0,
            dim: 0.0,
        };
        self.gpu
            .queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let v = |t: &wgpu::Texture| t.create_view(&Default::default());
        let mut enc = dev.create_command_encoder(&Default::default());

        // 1. unpack
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

        // 2. blur pyramid, only when the mode needs a background plane
        //
        // The ping-pong bookkeeping is tracked explicitly rather than derived from the
        // pass count. An earlier version computed which texture held the result from
        // `passes % 2` and got it wrong by one, so the composite read the *input* to the
        // final pass — the blur ran correctly and was then thrown away, showing up as a
        // barely-there 7% variance drop instead of a real blur.
        let mut blur_in_a = true;
        if matches!(mode, BackgroundMode::Blur) {
            // Map 0..1 onto 1..MAX passes. Each down/up pair roughly doubles the radius,
            // so this is a geometric control rather than a linear one.
            let passes = 1 + (blur_strength.clamp(0.0, 1.0) * (MAX_BLUR_PASSES - 1) as f32) as u32;
            let (hw, hh) = (self.width / 2, self.height / 2);

            // First pass reads the full-res frame; the rest ping-pong at half res.
            self.blur_pass(&mut enc, &self.blur_down, &self.rgba, &self.blur_a, hw, hh);
            for _ in 1..passes {
                let (src, dst) = if blur_in_a {
                    (&self.blur_a, &self.blur_b)
                } else {
                    (&self.blur_b, &self.blur_a)
                };
                self.blur_pass(&mut enc, &self.blur_down, src, dst, hw, hh);
                blur_in_a = !blur_in_a;
            }
            for _ in 0..passes {
                let (src, dst) = if blur_in_a {
                    (&self.blur_a, &self.blur_b)
                } else {
                    (&self.blur_b, &self.blur_a)
                };
                self.blur_pass(&mut enc, &self.blur_up, src, dst, hw, hh);
                blur_in_a = !blur_in_a;
            }
        }

        // Whichever half of the ping-pong the last pass actually wrote into.
        let bg = if !matches!(mode, BackgroundMode::Blur) {
            &self.rgba
        } else if blur_in_a {
            &self.blur_a
        } else {
            &self.blur_b
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
            ],
            w,
            h,
        );
    }
}
