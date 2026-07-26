// --- dual-Kawase blur ----------------------------------------------------------------
//
// Two passes, halving then doubling resolution, rather than a wide Gaussian. A Gaussian
// large enough to hide a room needs a very wide kernel and cost grows with radius; the
// dual-Kawase gets an equivalent look from a handful of bilinear taps per pixel, and the
// resampling softens residual detail the way real lens bokeh does. Repeating the pair
// grows the effective radius geometrically at linear cost.

@group(0) @binding(0) var blur_src: texture_2d<f32>;
@group(0) @binding(1) var blur_dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var blur_samp: sampler;

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(blur_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let half_px = 0.5 / vec2<f32>(dims);

    var sum = textureSampleLevel(blur_src, blur_samp, uv, 0.0) * 4.0;
    sum += textureSampleLevel(blur_src, blur_samp, uv - half_px, 0.0);
    sum += textureSampleLevel(blur_src, blur_samp, uv + half_px, 0.0);
    sum += textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(half_px.x, -half_px.y), 0.0);
    sum += textureSampleLevel(blur_src, blur_samp, uv - vec2<f32>(half_px.x, -half_px.y), 0.0);

    textureStore(blur_dst, vec2<i32>(gid.xy), sum / 8.0);
}

@compute @workgroup_size(8, 8)
fn up(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(blur_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let px = 1.0 / vec2<f32>(dims);

    var sum = textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(-px.x * 2.0, 0.0), 0.0);
    sum += textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(-px.x, px.y), 0.0) * 2.0;
    sum += textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(0.0, px.y * 2.0), 0.0);
    sum += textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(px.x, px.y), 0.0) * 2.0;
    sum += textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(px.x * 2.0, 0.0), 0.0);
    sum += textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(px.x, -px.y), 0.0) * 2.0;
    sum += textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(0.0, -px.y * 2.0), 0.0);
    sum += textureSampleLevel(blur_src, blur_samp, uv + vec2<f32>(-px.x, -px.y), 0.0) * 2.0;

    textureStore(blur_dst, vec2<i32>(gid.xy), sum / 12.0);
}

