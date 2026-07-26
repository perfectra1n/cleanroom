// --- dual-Kawase blur ----------------------------------------------------------------
//
// Passes that halve then double resolution, rather than a wide Gaussian. A Gaussian large
// enough to hide a room needs a very wide kernel and cost grows with radius; the
// dual-Kawase gets an equivalent look from a handful of bilinear taps per pixel, and the
// resampling softens residual detail the way real lens bokeh does.
//
// The geometric growth comes entirely from the *resolution halving*, not from repetition:
// a fixed 1.5-texel tap pattern covers twice as much of the picture at each level down, so
// N levels give a radius on the order of 2^N. Running the same tap pattern N times at one
// fixed resolution — which is what this shader used to be driven with — grows the radius
// like sqrt(N) instead, which is why the strength slider used to do so little. The host
// must therefore dispatch these against a real pyramid of textures.
//
// `offset` scales the tap spacing within a level, which is what makes the strength control
// continuous: it slides from 1.0 to 2.0 across a level, and at 2.0 it has reached what the
// next level down produces at 1.0, so there is no visible step as levels change.

struct BlurParams {
    offset: f32,
}

@group(0) @binding(0) var blur_src: texture_2d<f32>;
@group(0) @binding(1) var blur_dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var blur_samp: sampler;
@group(0) @binding(3) var<uniform> blur: BlurParams;

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(blur_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let half_px = (0.5 * blur.offset) / vec2<f32>(dims);

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
    let px = blur.offset / vec2<f32>(dims);

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

