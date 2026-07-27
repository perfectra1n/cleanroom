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
@group(0) @binding(1) var blur_dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var blur_samp: sampler;
@group(0) @binding(3) var<uniform> blur: BlurParams;
// Only read by `down_weighted`, which is the one pass that reads the full-resolution frame.
@group(0) @binding(4) var blur_matte: texture_2d<f32>;

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

// --- the weighted variant, for level 0 only -------------------------------------------
//
// The background plane used to be blurred from the whole frame, subject included. That put
// a low-frequency copy of the subject *inside* their own background: standing still it is
// invisible, but under quick movement the smear slides around behind them, and their
// colours halo out around the silhouette. It is the single most recognisable artifact of a
// naive background blur.
//
// The fix is normalized convolution. Each tap contributes its colour premultiplied by how
// much of that point is background, and carries that weight along in `.a`; the composite
// divides the weight back out at the end. Where the neighbourhood is all subject the weight
// goes to zero — and so does the visible contribution, because alpha is 1 there and the
// composite takes the foreground whole.
//
// This costs one extra texture binding and nothing else. Every later level and the whole up
// chain sum `vec4` linearly, so the weight rides through the rest of the pyramid for free.
//
// The weight comes from the matte directly rather than from the guided filter's refined
// alpha. That is deliberate: this mask is about to be blurred by the entire pyramid, so its
// edge precision is irrelevant, and a bilinearly-stretched matte gives a softer exclusion
// than a hard silhouette would. The composite still uses the guided model for the edge
// anyone actually sees.
fn weighted_tap(uv: vec2<f32>) -> vec4<f32> {
    let bg_weight = 1.0 - clamp(textureSampleLevel(blur_matte, blur_samp, uv, 0.0).r, 0.0, 1.0);
    let rgb = textureSampleLevel(blur_src, blur_samp, uv, 0.0).rgb;
    return vec4<f32>(rgb * bg_weight, bg_weight);
}

@compute @workgroup_size(8, 8)
fn down_weighted(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(blur_dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let half_px = (0.5 * blur.offset) / vec2<f32>(dims);

    // The same tap pattern as `down` above — it has to be, or level 0 would be built with a
    // different kernel from every level beneath it.
    var sum = weighted_tap(uv) * 4.0;
    sum += weighted_tap(uv - half_px);
    sum += weighted_tap(uv + half_px);
    sum += weighted_tap(uv + vec2<f32>(half_px.x, -half_px.y));
    sum += weighted_tap(uv - vec2<f32>(half_px.x, -half_px.y));

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

