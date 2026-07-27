// --- composite -----------------------------------------------------------------------

struct CompositeParams {
    // 0 = off, 1 = blur, 2 = replace, 3 = remove (solid key colour).
    mode: u32,
    mirror: u32,
    // Pulls the background toward luma. Applied only to the background plane.
    desaturate: f32,
    dim: f32,
    // Non-zero when comp_ab holds a valid guided-filter coefficient field. Zero falls back
    // to sampling the matte directly, which is what happens before the first inference.
    guided: u32,
    // Pulls the alpha edge inward. Replace needs more of this than blur: against a blurred
    // version of the same room a slightly generous silhouette is invisible, but against a
    // swapped background it is a bright halo tracing the shoulders and ears.
    tighten: f32,
    // Widens the alpha ramp about its crossing, without moving the crossing. Independent of
    // `tighten`, which moves the crossing and — being a shift-and-rescale — actually makes
    // the ramp *steeper* as it does so.
    feather: f32,
    _pad0: u32,
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
// Guided-filter coefficients (a in .r, b in .g) at matte resolution. Two smooth fields,
// which is exactly what bilinear upsampling is good at — unlike the matte itself.
@group(0) @binding(7) var comp_ab: texture_2d<f32>;

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

    var alpha: f32;
    if (comp.guided != 0u) {
        // Evaluate the local linear model at full resolution. `ab` is smooth, so sampling
        // it bilinearly is legitimate; the sharpness comes from `i`, which is this pixel's
        // own luma at full resolution rather than anything interpolated.
        let ab = textureSampleLevel(comp_ab, comp_samp, uv, 0.0).rg;
        let i = dot(fg, vec3<f32>(0.299, 0.587, 0.114));
        alpha = clamp(ab.r * i + ab.g, 0.0, 1.0);
    } else {
        alpha = clamp(textureSampleLevel(comp_matte, comp_samp, uv, 0.0).r, 0.0, 1.0);
    }

    // Shape the edge by remapping the ramp rather than by a morphological pass: a real
    // erode would need another full-resolution pass and a second texture, and this is a
    // matte whose edge is already a soft gradient, so remapping achieves the same thing
    // for free.
    //
    // Two independent controls, and it is worth being clear that they are not the same
    // knob. `tighten` moves where alpha crosses 0.5, pulling the silhouette inward — and
    // because it is a shift-and-rescale with gain 1/(1-tighten), it also makes the ramp
    // steeper, which is the opposite of softening. `feather` widens the ramp around
    // whatever crossing `tighten` chose, leaving the crossing itself alone.
    if (comp.feather > 0.0) {
        // Same crossing as the linear remap below, a wider and C1-smooth ramp around it.
        // The 3x is what makes the top of the slider a genuinely soft edge rather than a
        // barely-perceptible change.
        let c = (1.0 + comp.tighten) * 0.5;
        let w = clamp((1.0 - comp.tighten) * 0.5 * (1.0 + 3.0 * comp.feather), 0.01, 0.5);
        alpha = smoothstep(c - w, c + w, alpha);
    } else if (comp.tighten > 0.0) {
        // feather = 0 keeps the original behaviour exactly, so an existing config that
        // never sets feather composites bit for bit as it did before.
        alpha = clamp((alpha - comp.tighten) / max(1.0 - comp.tighten, 0.001), 0.0, 1.0);
    }

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
