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
    // Radius, in full-resolution pixels, over which the alpha is spatially averaged. Zero
    // disables the average entirely, which is bit-for-bit the un-feathered path.
    //
    // A *spatial* radius rather than an alpha-space width, because the latter cannot work.
    // See `resolve_alpha` below.
    feather_px: f32,
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

// Alpha at one point, before any edge shaping.
//
// Two sources, and the branch is on whether the guided coefficient field has been fitted
// for the matte currently uploaded. Before the first inference it has not.
fn resolve_alpha(uv: vec2<f32>) -> f32 {
    if (comp.guided != 0u) {
        // Evaluate the local linear model at full resolution. `ab` is smooth, so sampling it
        // bilinearly is legitimate; the sharpness comes from the luma, which is this point's
        // own at full resolution rather than anything upsampled from the matte.
        let ab = textureSampleLevel(comp_ab, comp_samp, uv, 0.0).rg;
        let i = dot(
            textureSampleLevel(comp_fg, comp_samp, uv, 0.0).rgb,
            vec3<f32>(0.299, 0.587, 0.114),
        );
        return clamp(ab.r * i + ab.g, 0.0, 1.0);
    }
    // The matte is smaller than the frame, so it is sampled rather than loaded — bilinear
    // interpolation is what keeps the edge smooth instead of blocky.
    return clamp(textureSampleLevel(comp_matte, comp_samp, uv, 0.0).r, 0.0, 1.0);
}

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

    let uv = (vec2<f32>(src) + 0.5) / vec2<f32>(dims);

    var alpha = resolve_alpha(uv);

    // Feather: a genuine spatial average, because there is no other way to do it.
    //
    // This control used to remap alpha *values* — `smoothstep(c - w, c + w, alpha)` for a
    // width `w` grown from the slider. That cannot feather, and the reason is worth keeping.
    // Remapping values can move where the edge sits and reshape its profile, but the
    // transition still occupies exactly the same *pixels*. Worse, `w` was clamped to 0.5,
    // and at `tighten = 0` — the default for blur — `(1 - 0) * 0.5` already sat on that
    // ceiling, so every non-zero feather produced the identical curve. The slider had two
    // states, and the one it reached had gradient 1.5 at the midpoint: it made the edge
    // *harder* through the transition and only softened the tails.
    //
    // Averaging over a disc is what actually spreads the ramp across more pixels. It has to
    // happen here, at full resolution, and not on the matte: anything softened at 512x288 is
    // re-sharpened by `a*I + b` on the way up.
    if (comp.feather_px > 0.0) {
        // A Vogel disc — radius sqrt((i + 0.5)/N), golden-angle rotation — rather than a
        // ring or a box.
        //
        // What matters for an edge is how the taps project onto the edge *normal*, and a
        // single ring projects terribly: six points at radius r land on just three distinct
        // offsets (0, +/-r/2, +/-r) with a hole between, so a straight edge comes out as four
        // coarse steps instead of a ramp. Measured on a hard test edge that was alpha
        // 0.14 / 0.43 / 0.57 / 0.86 and nothing in between. The Vogel spiral spreads twelve
        // taps evenly over the disc, so every normal direction sees twelve distinct offsets
        // and the ramp is smooth whichever way the edge runs.
        //
        // Only paid when the control is non-zero, and the branch is on a uniform, so it is
        // coherent across the dispatch rather than divergent.
        let r = comp.feather_px / vec2<f32>(dims);
        var sum = alpha;
        sum = sum + resolve_alpha(uv + vec2<f32>( 0.204124,  0.000000) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>(-0.260699,  0.238822) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>( 0.039904, -0.454688) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>( 0.328595,  0.428593) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>(-0.603011, -0.106664) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>( 0.571225, -0.363367) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>(-0.191064,  0.710747) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>(-0.364379, -0.701590) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>( 0.790557,  0.288710) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>(-0.822442,  0.339492) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>( 0.396472, -0.847237) * r);
        sum = sum + resolve_alpha(uv + vec2<f32>( 0.292982,  0.934074) * r);
        alpha = sum / 13.0;
    }

    // Tighten: unchanged, and applied *after* the average so the two controls are finally
    // independent. Feather decides how wide the ramp is; this decides where it sits, by
    // moving the 0.5 crossing inward. Being a shift-and-rescale its gain is 1/(1-tighten),
    // so it still steepens as it erodes — which is why the two are separate knobs and why
    // softening is feather's job, not a smaller tighten's.
    if (comp.tighten > 0.0) {
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
