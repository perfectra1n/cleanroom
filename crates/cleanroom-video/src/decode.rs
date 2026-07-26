//! MJPEG decode, and the raw-format conversions that feed the GPU.
//!
//! This is the only CPU step in the video path, and it is not a fallback: there is no
//! portable GPU JPEG decode. Vulkan Video covers H.264/H.265/AV1 and not JPEG; nvjpeg is
//! CUDA-only; AMD's VCN JPEG block is reachable only through VA-API. Since MJPEG is also
//! the *only* way to get 1080p30 out of a USB 2 camera, decoding it on the CPU is simply
//! part of the design.
//!
//! It is cheap enough not to matter. libjpeg-turbo decodes 1080p in a few milliseconds
//! with SIMD, against a 33 ms frame budget.
//!
//! We decode straight to **YUV planes** rather than to RGB. The camera gives us YUV, the
//! virtual camera wants YUV, and the GPU can sample YUV directly — converting to RGB on
//! the way in would mean converting back on the way out, for nothing.

use crate::format::PixelFormat;
use turbojpeg::{Decompressor, Image, PixelFormat as TjFormat};

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("JPEG decode failed: {0}")]
    Jpeg(String),

    #[error("frame is {got} bytes but {expected} were expected for {width}x{height} {format}")]
    ShortFrame {
        got: usize,
        expected: usize,
        width: u32,
        height: u32,
        format: &'static str,
    },

    #[error("cannot decode {0}")]
    Unsupported(&'static str),
}

/// A decoded frame as packed YUY2 — the format the sink publishes and the GPU uploads.
///
/// Packed rather than planar because YUY2 is what v4l2loopback carries, so a passthrough
/// path can hand this straight to the sink with no further work.
pub struct Yuy2Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl Yuy2Frame {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0x80; (width * height * 2) as usize],
            width,
            height,
        }
    }
}

/// Decodes camera frames into YUY2, reusing its scratch buffers across calls.
///
/// Stateful on purpose: allocating a 1080p intermediate per frame is 4 MB of churn at
/// 30 Hz, which is exactly the kind of avoidable allocation that shows up as jitter.
pub struct FrameDecoder {
    jpeg: Decompressor,
    /// RGB scratch for the JPEG path.
    rgb: Vec<u8>,
    width: u32,
    height: u32,
}

impl FrameDecoder {
    pub fn new(width: u32, height: u32) -> Result<Self, DecodeError> {
        Ok(Self {
            jpeg: Decompressor::new().map_err(|e| DecodeError::Jpeg(e.to_string()))?,
            rgb: vec![0u8; (width * height * 3) as usize],
            width,
            height,
        })
    }

    /// Convert a camera frame to YUY2, decoding if necessary.
    pub fn to_yuy2(
        &mut self,
        data: &[u8],
        format: PixelFormat,
        width: u32,
        height: u32,
        out: &mut Yuy2Frame,
    ) -> Result<(), DecodeError> {
        if (width, height) != (self.width, self.height) {
            self.width = width;
            self.height = height;
            self.rgb.resize((width * height * 3) as usize, 0);
        }
        if out.width != width || out.height != height {
            *out = Yuy2Frame::new(width, height);
        }

        match format {
            // Already the output format. This is the passthrough fast path and involves
            // no conversion at all.
            PixelFormat::Yuyv => {
                let expected = (width * height * 2) as usize;
                if data.len() < expected {
                    return Err(DecodeError::ShortFrame {
                        got: data.len(),
                        expected,
                        width,
                        height,
                        format: format.as_str(),
                    });
                }
                out.data[..expected].copy_from_slice(&data[..expected]);
                Ok(())
            }
            PixelFormat::Mjpeg => self.jpeg_to_yuy2(data, width, height, out),
            PixelFormat::Nv12 => nv12_to_yuy2(data, width, height, out),
            PixelFormat::Yu12 => yu12_to_yuy2(data, width, height, out),
        }
    }

    fn jpeg_to_yuy2(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        out: &mut Yuy2Frame,
    ) -> Result<(), DecodeError> {
        let mut image = Image {
            pixels: self.rgb.as_mut_slice(),
            width: width as usize,
            pitch: (width * 3) as usize,
            height: height as usize,
            format: TjFormat::RGB,
        };
        self.jpeg
            .decompress(data, image.as_deref_mut())
            .map_err(|e| DecodeError::Jpeg(e.to_string()))?;
        rgb_to_yuy2(&self.rgb, width, height, &mut out.data);
        Ok(())
    }
}

/// BT.601 limited range, matching what UVC cameras produce and what the virtual camera
/// is expected to carry. Getting the range wrong is the classic washed-out or crushed
/// picture, and it is invisible until you compare against the raw camera.
fn rgb_to_yuy2(rgb: &[u8], width: u32, height: u32, out: &mut [u8]) {
    let w = width as usize;
    for y in 0..height as usize {
        let row_rgb = y * w * 3;
        let row_out = y * w * 2;
        // YUY2 packs two pixels per four bytes and shares one chroma sample between them,
        // so the loop steps in pairs.
        for x in (0..w).step_by(2) {
            let i0 = row_rgb + x * 3;
            let i1 = i0 + 3;

            let (r0, g0, b0) = (rgb[i0] as i32, rgb[i0 + 1] as i32, rgb[i0 + 2] as i32);
            let (r1, g1, b1) = if x + 1 < w {
                (rgb[i1] as i32, rgb[i1 + 1] as i32, rgb[i1 + 2] as i32)
            } else {
                (r0, g0, b0)
            };

            let y0 = ((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16;
            let y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;

            // Chroma from the averaged pair rather than point-sampled from the first
            // pixel: box-averaging is what every reference implementation does and it
            // avoids chroma shimmer on vertical edges.
            let ra = (r0 + r1 + 1) >> 1;
            let ga = (g0 + g1 + 1) >> 1;
            let ba = (b0 + b1 + 1) >> 1;
            let u = ((-38 * ra - 74 * ga + 112 * ba + 128) >> 8) + 128;
            let v = ((112 * ra - 94 * ga - 18 * ba + 128) >> 8) + 128;

            let o = row_out + x * 2;
            out[o] = y0.clamp(0, 255) as u8;
            out[o + 1] = u.clamp(0, 255) as u8;
            if x + 1 < w {
                out[o + 2] = y1.clamp(0, 255) as u8;
                out[o + 3] = v.clamp(0, 255) as u8;
            }
        }
    }
}

fn nv12_to_yuy2(
    data: &[u8],
    width: u32,
    height: u32,
    out: &mut Yuy2Frame,
) -> Result<(), DecodeError> {
    let (w, h) = (width as usize, height as usize);
    let expected = w * h * 3 / 2;
    if data.len() < expected {
        return Err(DecodeError::ShortFrame {
            got: data.len(),
            expected,
            width,
            height,
            format: "NV12",
        });
    }
    let (y_plane, uv_plane) = data.split_at(w * h);
    for row in 0..h {
        // 4:2:0 shares one chroma row between two luma rows.
        let uv_row = (row / 2) * w;
        for x in (0..w).step_by(2) {
            let o = (row * w + x) * 2;
            out.data[o] = y_plane[row * w + x];
            out.data[o + 1] = uv_plane[uv_row + x];
            if x + 1 < w {
                out.data[o + 2] = y_plane[row * w + x + 1];
                out.data[o + 3] = uv_plane[uv_row + x + 1];
            }
        }
    }
    Ok(())
}

fn yu12_to_yuy2(
    data: &[u8],
    width: u32,
    height: u32,
    out: &mut Yuy2Frame,
) -> Result<(), DecodeError> {
    let (w, h) = (width as usize, height as usize);
    let expected = w * h * 3 / 2;
    if data.len() < expected {
        return Err(DecodeError::ShortFrame {
            got: data.len(),
            expected,
            width,
            height,
            format: "YU12",
        });
    }
    let y_plane = &data[..w * h];
    let u_plane = &data[w * h..w * h + w * h / 4];
    let v_plane = &data[w * h + w * h / 4..];
    let cw = w / 2;
    for row in 0..h {
        let crow = (row / 2) * cw;
        for x in (0..w).step_by(2) {
            let o = (row * w + x) * 2;
            let c = x / 2;
            out.data[o] = y_plane[row * w + x];
            out.data[o + 1] = u_plane[crow + c];
            if x + 1 < w {
                out.data[o + 2] = y_plane[row * w + x + 1];
                out.data[o + 3] = v_plane[crow + c];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_passthrough_is_byte_exact() {
        // The passthrough path must not touch the data at all.
        let (w, h) = (64u32, 32u32);
        let src: Vec<u8> = (0..(w * h * 2)).map(|i| (i % 251) as u8).collect();
        let mut dec = FrameDecoder::new(w, h).unwrap();
        let mut out = Yuy2Frame::new(w, h);
        dec.to_yuy2(&src, PixelFormat::Yuyv, w, h, &mut out)
            .unwrap();
        assert_eq!(out.data, src);
    }

    #[test]
    fn a_truncated_frame_is_an_error_not_a_torn_picture() {
        let (w, h) = (64u32, 32u32);
        let mut dec = FrameDecoder::new(w, h).unwrap();
        let mut out = Yuy2Frame::new(w, h);
        let short = vec![0u8; 10];
        assert!(matches!(
            dec.to_yuy2(&short, PixelFormat::Yuyv, w, h, &mut out),
            Err(DecodeError::ShortFrame { .. })
        ));
    }

    #[test]
    fn grey_rgb_maps_to_neutral_chroma() {
        // A neutral input must produce U=V=128. Any drift here is a colour cast that is
        // hard to spot by eye and impossible to unsee once noticed.
        let (w, h) = (4u32, 2u32);
        let rgb = vec![128u8; (w * h * 3) as usize];
        let mut out = vec![0u8; (w * h * 2) as usize];
        rgb_to_yuy2(&rgb, w, h, &mut out);
        for px in out.chunks(4) {
            assert!((px[1] as i32 - 128).abs() <= 1, "U drifted: {}", px[1]);
            assert!((px[3] as i32 - 128).abs() <= 1, "V drifted: {}", px[3]);
        }
    }

    #[test]
    fn black_and_white_land_on_studio_swing_limits() {
        // BT.601 limited range: black is Y=16, white is Y=235. Full-range would give
        // 0 and 255, and the difference is a visibly washed-out or crushed picture.
        let (w, h) = (2u32, 1u32);
        let mut out = vec![0u8; 4];

        rgb_to_yuy2(&vec![0u8; 6], w, h, &mut out);
        assert_eq!(out[0], 16, "black must be Y=16, not 0");

        rgb_to_yuy2(&vec![255u8; 6], w, h, &mut out);
        assert!(
            (out[0] as i32 - 235).abs() <= 1,
            "white must be Y=235, got {}",
            out[0]
        );
    }

    #[test]
    fn nv12_converts_with_the_right_chroma_row_sharing() {
        // 4:2:0 shares a chroma row between two luma rows; getting that wrong shifts
        // colour down the image by one row, which looks like fringing.
        let (w, h) = (4u32, 4u32);
        let mut src = vec![0u8; (w * h * 3 / 2) as usize];
        for (i, v) in src[..(w * h) as usize].iter_mut().enumerate() {
            *v = i as u8;
        }
        // Distinct chroma per row-pair.
        let uv_start = (w * h) as usize;
        src[uv_start..].fill(200);

        let mut out = Yuy2Frame::new(w, h);
        nv12_to_yuy2(&src, w, h, &mut out).unwrap();
        assert_eq!(out.data[0], 0, "first luma sample");
        assert_eq!(out.data[1], 200, "first chroma sample");
    }

    #[test]
    fn odd_widths_do_not_panic() {
        // YUY2 pairs pixels, so an odd width is the natural off-by-one trap.
        let (w, h) = (5u32, 3u32);
        let rgb = vec![100u8; (w * h * 3) as usize];
        let mut out = vec![0u8; (w * h * 2) as usize];
        rgb_to_yuy2(&rgb, w, h, &mut out);
    }
}
