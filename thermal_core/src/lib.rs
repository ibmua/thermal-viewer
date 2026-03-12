//! thermal_core — hot-path helpers for thermal_viewer.py
//!
//! Build:
//!   cd thermal_core
//!   pip install maturin
//!   maturin develop --release          # installs into current venv
//!   # or: maturin build --release      # produces a .whl to pip install
//!
//! Usage in Python:
//!   import thermal_core
//!   bgr = thermal_core.process_frame(gray_u8, correction_f32, bad_mask_u8,
//!                                    nuc_offset, prev_f32,
//!                                    mu, thresh, iron_lut)

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

// ── Iron colour-map LUT (256 entries, BGR) ────────────────────────────────────
fn build_iron_lut() -> [[u8; 3]; 256] {
    let stops: &[(f32, [f32; 3])] = &[
        (0.0,   [0.0,   0.0,   0.0]),
        (64.0,  [102.0, 0.0,   102.0]),
        (128.0, [204.0, 0.0,   0.0]),
        (192.0, [255.0, 153.0, 0.0]),
        (255.0, [255.0, 255.0, 255.0]),
    ];
    let mut lut = [[0u8; 3]; 256];
    for i in 0usize..256 {
        let v = i as f32;
        for w in 0..stops.len() - 1 {
            let (v0, c0) = stops[w];
            let (v1, c1) = stops[w + 1];
            if v >= v0 && v <= v1 {
                let f = (v - v0) / (v1 - v0);
                // LUT stored as BGR (OpenCV convention)
                lut[i][0] = (c0[0] + f * (c1[0] - c0[0])) as u8; // B = R channel
                lut[i][1] = (c0[1] + f * (c1[1] - c0[1])) as u8; // G
                lut[i][2] = (c0[2] + f * (c1[2] - c0[2])) as u8; // R = B channel
                break;
            }
        }
    }
    lut
}

// ── Row-wise box-blur (horizontal pass only — caller applies twice) ────────────
fn hbox_blur_f32(src: &[f32], dst: &mut [f32], w: usize, r: usize) {
    let rr = r as i64;
    for row in 0..src.len() / w {
        let s = &src[row * w..(row + 1) * w];
        let d = &mut dst[row * w..(row + 1) * w];
        let mut sum: f32 = s[0] * (rr + 1) as f32;
        for x in 0..=r.min(w - 1) { sum += s[x]; }
        for x in 0..w {
            d[x] = sum / (2 * r + 1) as f32;
            let add = s[(x + r + 1).min(w - 1)];
            let sub = if x as i64 - rr - 1 >= 0 { s[x - r - 1] } else { s[0] };
            sum += add - sub;
        }
    }
}

fn vbox_blur_f32(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let rr = r as i64;
    for col in 0..w {
        let mut sum: f32 = src[col] * (rr + 1) as f32;
        for y in 0..=r.min(h - 1) { sum += src[y * w + col]; }
        for y in 0..h {
            dst[y * w + col] = sum / (2 * r + 1) as f32;
            let ay = (y + r + 1).min(h - 1);
            let sy = if y as i64 - rr - 1 >= 0 { y - r - 1 } else { 0 };
            sum += src[ay * w + col] - src[sy * w + col];
        }
    }
}

/// Separable box blur on f32 image (faster than GaussianBlur for large kernels).
/// radius = half-width; total kernel = 2*radius+1
fn box_blur_f32(src: &[f32], tmp: &mut [f32], dst: &mut [f32], w: usize, h: usize, radius: usize) {
    hbox_blur_f32(src, tmp, w, radius);
    vbox_blur_f32(tmp, dst, w, h, radius);
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// process_frame(gray_u8, correction_f32, bad_mask_u8,
///               nuc_offset_f32, prev_f32,
///               mu, thresh, out_nuc_offset, out_prev) -> bytes (BGR u8, CAM_W x CAM_H)
///
/// All arrays are flat row-major, length = width * height.
/// correction_f32 / bad_mask_u8 may be None (no flat-field calibration).
/// nuc_offset_f32 / prev_f32 are read and written in-place (must be writeable).
#[pyfunction]
fn process_frame<'py>(
    py:           Python<'py>,
    gray:         &[u8],          // input: 8-bit gray (CAM_W x CAM_H)
    width:        usize,
    height:       usize,
    correction:   Option<&[f32]>, // flat-field additive correction
    bad_mask:     Option<&[u8]>,  // 1 = bad pixel, 0 = ok
    nuc_offset:   &mut [f32],     // NUC offset (mutated in-place)
    prev_frame:   &mut [f32],     // previous f32 frame (mutated)
    mu:           f32,            // NUC learning rate
    thresh:       f32,            // NUC motion threshold
    iron_lut:     &[[u8; 3]],    // 256-entry Iron LUT (BGR)
) -> PyResult<Bound<'py, PyBytes>> {
    let n = width * height;
    assert_eq!(gray.len(), n);
    assert_eq!(nuc_offset.len(), n);
    assert_eq!(prev_frame.len(), n);

    // 1. Cast u8 → f32
    let mut f: Vec<f32> = gray.iter().map(|&v| v as f32).collect();

    // 2. Flat-field correction
    if let Some(corr) = correction {
        assert_eq!(corr.len(), n);
        f.par_iter_mut().zip(corr.par_iter()).for_each(|(v, c)| *v += c);
    }
    if let (Some(bm), Some(corr)) = (bad_mask, correction) {
        assert_eq!(bm.len(), n);
        // neighbour mean via 3×3 box blur of the corrected image
        let mut tmp = vec![0f32; n];
        let mut blurred = vec![0f32; n];
        box_blur_f32(&f, &mut tmp, &mut blurred, width, height, 1);
        f.iter_mut()
         .zip(bm.iter())
         .zip(blurred.iter())
         .filter(|((_, &bad), _)| bad != 0)
         .for_each(|((v, _), &b)| *v = b);
        let _ = corr; // suppress unused warning
    }

    // 3. NUC update (motion-gated LMS, same logic as MotionGatedNUC.update)
    let corrected: Vec<f32> = f.iter().zip(nuc_offset.iter()).map(|(v, o)| v - o).collect();

    // Sub-sample 1/16 for MAE motion check (every 4th row/col)
    let sub_n = (width / 4) * (height / 4);
    let mae: f32 = if sub_n > 0 {
        let sum: f32 = (0..height/4).flat_map(|r| {
            let row = r * 4;
            (0..width/4).map(move |c| {
                let idx = row * width + c * 4;
                (f[idx] - prev_frame[idx]).abs()
            })
        }).sum();
        sum / sub_n as f32
    } else { 0.0 };

    if mae > thresh {
        // Box blur of corrected (radius 6 ~ 13×13 kernel, matches sigma=2 Gaussian support)
        let mut tmp  = vec![0f32; n];
        let mut desired = vec![0f32; n];
        box_blur_f32(&corrected, &mut tmp, &mut desired, width, height, 6);
        nuc_offset.iter_mut()
                  .zip(corrected.iter())
                  .zip(desired.iter())
                  .for_each(|((o, c), d)| *o += mu * (c - d));
    }
    prev_frame.copy_from_slice(&f);

    // 4. Normalize corrected to [0, 255]
    let (mn, mx) = corrected.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let scale = if mx > mn { 255.0 / (mx - mn) } else { 1.0 };

    // 5. Apply Iron LUT → BGR bytes
    let bgr: Vec<u8> = corrected.par_iter().flat_map(|&v| {
        let idx = ((v - mn) * scale).round().clamp(0.0, 255.0) as usize;
        let [b, g, r] = iron_lut[idx];
        [b, g, r]
    }).collect();

    Ok(PyBytes::new(py, &bgr))
}

#[pymodule]
fn thermal_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_frame, m)?)?;
    Ok(())
}
