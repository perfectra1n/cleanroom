// Pack RGBA back into YUY2 for the virtual camera.

@group(0) @binding(0) var rgba_in: texture_2d<f32>;
@group(0) @binding(1) var packed_out: texture_storage_2d<rgba8uint, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(packed_out);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let x = i32(gid.x) * 2;
    let y = i32(gid.y);
    let src_dims = textureDimensions(rgba_in);

    let c0 = textureLoad(rgba_in, vec2<i32>(x, y), 0).rgb;
    // Clamp rather than wrap at the right edge, so an odd width cannot sample row n+1.
    let x1 = min(x + 1, i32(src_dims.x) - 1);
    let c1 = textureLoad(rgba_in, vec2<i32>(x1, y), 0).rgb;

    let yuv0 = rgb_to_yuv(c0);
    let yuv1 = rgb_to_yuv(c1);

    // Chroma box-averaged across the pair rather than point-sampled from the first pixel;
    // point sampling shows up as chroma shimmer on vertical edges.
    let u = (yuv0.y + yuv1.y) * 0.5;
    let v = (yuv0.z + yuv1.z) * 0.5;

    textureStore(packed_out, vec2<i32>(gid.xy), vec4<u32>(
        u32(yuv0.x * 255.0 + 0.5),
        u32(u * 255.0 + 0.5),
        u32(yuv1.x * 255.0 + 0.5),
        u32(v * 255.0 + 0.5),
    ));
}
