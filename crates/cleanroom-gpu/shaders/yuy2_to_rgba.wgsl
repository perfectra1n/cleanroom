// Unpack YUY2 into linear-indexed RGBA.
//
// YUY2 packs two pixels into four bytes as Y0 U Y1 V, sharing one chroma sample between
// them. We carry it as an Rgba8Uint texture of half the width, so each texel is exactly
// one such quad — the packing lives in the type rather than in the indexing.

@group(0) @binding(0) var packed_in: texture_2d<u32>;
@group(0) @binding(1) var rgba_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(packed_in);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let quad = textureLoad(packed_in, vec2<i32>(gid.xy), 0);
    let y0 = f32(quad.r) / 255.0;
    let u  = f32(quad.g) / 255.0;
    let y1 = f32(quad.b) / 255.0;
    let v  = f32(quad.a) / 255.0;

    let x = i32(gid.x) * 2;
    let y = i32(gid.y);
    textureStore(rgba_out, vec2<i32>(x, y), vec4<f32>(yuv_to_rgb(y0, u, v), 1.0));
    textureStore(rgba_out, vec2<i32>(x + 1, y), vec4<f32>(yuv_to_rgb(y1, u, v), 1.0));
}
