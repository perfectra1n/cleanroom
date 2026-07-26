// Downscale the full-res RGBA frame to the matting network's input size.
//
// Done on the GPU rather than the CPU because the frame is already there, and because the
// readback is then only 512x288x4 = 590 KB rather than a full 1080p frame.
//
// Bilinear via the sampler rather than a box filter: the matting network is not sensitive
// to the difference, and one sample per output pixel is as cheap as this gets.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var samp: sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    textureStore(dst, vec2<i32>(gid.xy), textureSampleLevel(src, samp, uv, 0.0));
}
