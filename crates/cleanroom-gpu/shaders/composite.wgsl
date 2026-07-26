// --- composite -----------------------------------------------------------------------

struct CompositeParams {
    // 0 = off, 1 = blur, 2 = replace, 3 = remove (solid key colour).
    mode: u32,
    mirror: u32,
    // Pulls the background toward luma. Applied only to the background plane.
    desaturate: f32,
    dim: f32,
}

@group(0) @binding(0) var comp_fg: texture_2d<f32>;
@group(0) @binding(1) var comp_bg: texture_2d<f32>;
@group(0) @binding(2) var comp_matte: texture_2d<f32>;
@group(0) @binding(3) var comp_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var comp_samp: sampler;
@group(0) @binding(5) var<uniform> comp: CompositeParams;
// The replacement plate. Always bound — it is referenced from live code below, so it
// cannot be optimised out — and defaults to a 1x1 texel when no image is loaded, the same
// trick the matte uses to stay a valid binding before the first inference.
@group(0) @binding(6) var comp_bg_image: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(comp_out);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    // Mirroring is done here rather than as a separate pass: it is a free index flip on a
    // pass we are already running, and a dedicated pass would cost a full-frame copy.
    var src = vec2<i32>(gid.xy);
    if (comp.mirror != 0u) {
        src.x = i32(dims.x) - 1 - src.x;
    }

    let fg = textureLoad(comp_fg, src, 0).rgb;

    if (comp.mode == 0u) {
        textureStore(comp_out, vec2<i32>(gid.xy), vec4<f32>(fg, 1.0));
        return;
    }

    // The matte is computed at a lower resolution than the frame, so it is *sampled*
    // rather than loaded — bilinear interpolation is what keeps the edge smooth instead
    // of blocky.
    let uv = (vec2<f32>(src) + 0.5) / vec2<f32>(dims);
    let alpha = clamp(textureSampleLevel(comp_matte, comp_samp, uv, 0.0).r, 0.0, 1.0);

    var bg: vec3<f32>;
    if (comp.mode == 3u) {
        bg = vec3<f32>(0.0, 1.0, 0.0);
    } else if (comp.mode == 2u) {
        // Deliberately NOT `uv`. `src` was mirrored above, and sampling the plate through a
        // mirrored coordinate flips it with the subject — which is right for blur, where the
        // background *is* the same frame, and wrong for a photograph, where it means any
        // text in the plate reads backwards. The subject flips; the room behind them does
        // not.
        let plate_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
        bg = textureSampleLevel(comp_bg_image, comp_samp, plate_uv, 0.0).rgb;
    } else {
        bg = textureSampleLevel(comp_bg, comp_samp, uv, 0.0).rgb;
    }

    if (comp.desaturate > 0.0) {
        // Rec.601 luma weights, matching the colour space everything else uses.
        let luma = dot(bg, vec3<f32>(0.299, 0.587, 0.114));
        bg = mix(bg, vec3<f32>(luma), comp.desaturate);
    }
    bg *= (1.0 - comp.dim);

    textureStore(comp_out, vec2<i32>(gid.xy), vec4<f32>(mix(bg, fg, alpha), 1.0));
}
