// --- guided filter, coefficient pass ------------------------------------------------
//
// Fast guided filter (He & Sun, arXiv:1505.00996). The matte comes out of RVM at 512x288
// and is stretched over a 1920x1080 frame, and plain bilinear does not know where the
// subject ends — it spreads alpha evenly across the boundary, which is what puts a soft
// halo around a shoulder and lets background colour bleed into hair.
//
// The guided filter fixes the edge to *image* structure instead. For each pixel it solves
// a local linear model
//
//     q = a * I + b
//
// relating the guidance image I (the frame's luma, which has the real edges in it) to the
// filter input p (the matte). Over a window centred on k:
//
//     a_k = cov(I, p) / (var(I) + eps)
//     b_k = mean(p) - a_k * mean(I)
//
// Where the frame has an edge, var(I) is large, a -> 1 and the matte is allowed to follow
// it. Where the frame is flat, a -> 0 and the matte is smoothed instead. eps sets what
// counts as "flat"; it is the only real tuning knob.
//
// This pass computes a and b at *matte* resolution and writes them into one RG texture.
// The composite then samples them bilinearly at full resolution and evaluates q there,
// which is the "fast" part: upsampling two smooth coefficient fields is cheap and correct,
// where upsampling the matte itself is what we are trying to stop doing.

struct GuidedParams {
    // Window radius in matte pixels.
    radius: i32,
    // Regularisation. Larger means more smoothing and less edge-following.
    eps: f32,
}

@group(0) @binding(0) var g_guide: texture_2d<f32>;   // RGBA frame at matte resolution
@group(0) @binding(1) var g_matte: texture_2d<f32>;   // R8 matte from the network
@group(0) @binding(2) var g_ab: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> gp: GuidedParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(g_ab);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let c = vec2<i32>(gid.xy);
    let last = vec2<i32>(dims) - vec2<i32>(1, 1);

    var sum_i = 0.0;
    var sum_p = 0.0;
    var sum_ii = 0.0;
    var sum_ip = 0.0;
    var n = 0.0;

    // Clamped rather than skipped at the border. Skipping would shrink the window at the
    // frame edge and make the statistics there noisier than everywhere else, which shows
    // up as alpha crawling along the sides of the picture.
    for (var dy = -gp.radius; dy <= gp.radius; dy = dy + 1) {
        for (var dx = -gp.radius; dx <= gp.radius; dx = dx + 1) {
            let s = clamp(c + vec2<i32>(dx, dy), vec2<i32>(0, 0), last);
            let rgb = textureLoad(g_guide, s, 0).rgb;
            // Rec.601 luma, matching the colour space the rest of the pipeline uses.
            let i = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
            let p = textureLoad(g_matte, s, 0).r;

            sum_i = sum_i + i;
            sum_p = sum_p + p;
            sum_ii = sum_ii + i * i;
            sum_ip = sum_ip + i * p;
            n = n + 1.0;
        }
    }

    let mean_i = sum_i / n;
    let mean_p = sum_p / n;
    // var and cov in the biased form, which is what the paper uses and what keeps
    // a in [0, 1] for a matte that is already in [0, 1].
    let var_i = max(sum_ii / n - mean_i * mean_i, 0.0);
    let cov_ip = sum_ip / n - mean_i * mean_p;

    let a = cov_ip / (var_i + gp.eps);
    let b = mean_p - a * mean_i;

    textureStore(g_ab, vec2<i32>(gid.xy), vec4<f32>(a, b, 0.0, 0.0));
}
