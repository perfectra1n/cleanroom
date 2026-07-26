// BT.601 limited-range (studio swing) colour conversion, shared by every kernel.
//
// This is what UVC cameras produce and what the virtual camera is expected to carry.
// Black is Y=16/255 and white is Y=235/255, not 0 and 1. Treating it as full range gives
// a washed-out or crushed picture that is only obvious when compared side by side with
// the raw camera — which nobody does by accident.
//
// Included textually by the host rather than by a WGSL import, because naga has no
// include and every entry point needs its own module (bindings cannot be shared across
// entry points in one file).

const Y_OFFSET: f32 = 0.0627451;   // 16/255
const Y_SCALE: f32  = 1.1643836;   // 255/219, the studio-swing luma expansion
const C_OFFSET: f32 = 0.5019608;   // 128/255

fn yuv_to_rgb(y_raw: f32, u_raw: f32, v_raw: f32) -> vec3<f32> {
    let y = (y_raw - Y_OFFSET) * Y_SCALE;
    let u = u_raw - C_OFFSET;
    let v = v_raw - C_OFFSET;
    return clamp(
        vec3<f32>(
            y + 1.5960 * v,
            y - 0.3917 * u - 0.8129 * v,
            y + 2.0172 * u,
        ),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn rgb_to_yuv(c: vec3<f32>) -> vec3<f32> {
    let y = (0.2568 * c.r + 0.5041 * c.g + 0.0979 * c.b) + Y_OFFSET;
    let u = (-0.1482 * c.r - 0.2910 * c.g + 0.4392 * c.b) + C_OFFSET;
    let v = (0.4392 * c.r - 0.3678 * c.g - 0.0714 * c.b) + C_OFFSET;
    return clamp(vec3<f32>(y, u, v), vec3<f32>(0.0), vec3<f32>(1.0));
}
