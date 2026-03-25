//! thermal_core — hot-path helpers for thermal_viewer.py
//!
//! Build (from thermal_core/ directory):
//!   python3 -m pip install maturin
//!   python3 -m maturin build --release
//!   python3 -m pip install target/wheels/thermal_core-*.whl --force-reinstall
//!
//! Python usage:
//!   import thermal_core, numpy as np
//!
//!   # Persistent arrays (allocate once, reuse every frame):
//!   nuc_offset = np.zeros(CAM_H * CAM_W, np.float32)
//!   prev_frame = np.zeros(CAM_H * CAM_W, np.float32)
//!   iron_flat  = iron_lut.ravel().astype(np.uint8)   # shape (768,)
//!
//!   bgr_bytes = thermal_core.process_frame(
//!       gray.ravel(),          # 1-D uint8 numpy array
//!       CAM_W, CAM_H,
//!       nuc_offset,            # 1-D float32, modified in-place
//!       prev_frame,            # 1-D float32, modified in-place
//!       mu, thresh,
//!       iron_flat,             # 1-D uint8 (256*3)
//!   )
//!   bgr = np.frombuffer(bgr_bytes, np.uint8).reshape(CAM_H, CAM_W, 3)

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

// ── Separable box blur on f32 ─────────────────────────────────────────────────
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

fn box_blur_f32(src: &[f32], tmp: &mut [f32], dst: &mut [f32], w: usize, h: usize, radius: usize) {
    hbox_blur_f32(src, tmp, w, radius);
    vbox_blur_f32(tmp, dst, w, h, radius);
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// process_frame(gray, width, height, nuc_offset, prev_frame, mu, thresh, iron_lut,
///               out_width=0, out_height=0, correction=None, bad_mask=None)
///               -> bytes  (BGR uint8, out_width * out_height * 3)
///
/// gray        — 1-D contiguous uint8 numpy array, shape (width*height,)
/// nuc_offset  — 1-D contiguous float32 numpy array, MODIFIED IN-PLACE
/// prev_frame  — 1-D contiguous float32 numpy array, MODIFIED IN-PLACE
/// iron_lut    — 1-D contiguous uint8 numpy array, shape (256*3,), BGR triples
/// out_width   — output width  (0 = same as input width)
/// out_height  — output height (0 = same as input height)
/// correction  — optional 1-D float32 array (flat-field additive offset)
/// bad_mask    — optional 1-D uint8 array (1 = dead pixel, 0 = good)
///
/// When out_width/out_height differ from width/height, nearest-neighbour
/// upscaling is performed in parallel via rayon — no extra Python resize needed.
/// process_frame(..., out_buf) → None
///
/// out_buf — pre-allocated 1-D uint8 numpy array of length out_width*out_height*3.
///           Rust writes the BGR result directly into it — no intermediate Vec,
///           no PyBytes copy.  Python side reuses the same buffer every frame.
#[pyfunction]
#[pyo3(signature = (gray, width, height, nuc_offset, prev_frame, mu, thresh, iron_lut, out_buf, out_width=0, out_height=0, correction=None, bad_mask=None))]
fn process_frame<'py>(
    _py:          Python<'py>,
    gray:         PyReadonlyArray1<'py, u8>,
    width:        usize,
    height:       usize,
    nuc_offset:   PyReadwriteArray1<'py, f32>,
    prev_frame:   PyReadwriteArray1<'py, f32>,
    mu:           f32,
    thresh:       f32,
    iron_lut:     PyReadonlyArray1<'py, u8>,
    out_buf:      PyReadwriteArray1<'py, u8>,   // ← pre-allocated output buffer
    out_width:    usize,
    out_height:   usize,
    correction:   Option<PyReadonlyArray1<'py, f32>>,
    bad_mask:     Option<PyReadonlyArray1<'py, u8>>,
) -> PyResult<()> {
    let n = width * height;

    let gray_s  = gray.as_slice()?;
    let mut nuc_offset = nuc_offset;                  // need owned mut for as_slice_mut
    let mut prev_frame = prev_frame;
    let nuc_s   = nuc_offset.as_slice_mut()?;
    let prev_s  = prev_frame.as_slice_mut()?;
    let lut_raw = iron_lut.as_slice()?;               // 256*3 bytes

    assert_eq!(gray_s.len(), n);
    assert_eq!(nuc_s.len(),  n);
    assert_eq!(prev_s.len(), n);
    assert_eq!(lut_raw.len(), 256 * 3, "iron_lut must be shape (768,)");

    // Reinterpret flat u8 LUT as &[[u8;3]] for indexed lookup
    let lut: &[[u8; 3]] = bytemuck::cast_slice(lut_raw);

    // 1. Cast u8 → f32
    let mut f: Vec<f32> = gray_s.iter().map(|&v| v as f32).collect();

    // 2. Flat-field correction (additive per-pixel offset, optional)
    if let Some(corr) = &correction {
        let cs = corr.as_slice()?;
        assert_eq!(cs.len(), n);
        f.par_iter_mut().zip(cs.par_iter()).for_each(|(v, c)| *v += c);
    }

    // 2b. Dead-pixel replacement — in-place direct neighbour mean.
    //
    // We visit only the bad pixels (typically < 1% of the sensor) and for
    // each one sum up its GOOD neighbours in a 5×5 window.  Because we only
    // ever READ good pixels (bm[j] == 0) as neighbours and never write them,
    // in-place modification of `f` is safe — a freshly-patched bad pixel is
    // never used as a source for another bad pixel.
    //
    // This replaces the previous full-frame box-blur approach that required
    // ~8 MB of Vec allocations (f_good, good_wt, 4 blur buffers) per frame,
    // causing cache thrashing on M-series Macs (12 MB L3).  The new approach
    // uses zero extra allocation and processes only the ~few-hundred bad pixels.
    if let Some(bm_arr) = &bad_mask {
        let bm = bm_arr.as_slice()?;
        assert_eq!(bm.len(), n);
        for row in 0..height {
            for col in 0..width {
                let i = row * width + col;
                if bm[i] == 0 { continue; }           // good pixel — skip
                let r0 = row.saturating_sub(2);
                let r1 = (row + 3).min(height);
                let c0 = col.saturating_sub(2);
                let c1 = (col + 3).min(width);
                let mut sum = 0.0f32;
                let mut cnt = 0u32;
                for r in r0..r1 {
                    for c in c0..c1 {
                        let j = r * width + c;
                        if bm[j] == 0 {                // count good neighbours only
                            sum += f[j];
                            cnt += 1;
                        }
                    }
                }
                if cnt > 0 { f[i] = sum / cnt as f32; }
            }
        }
    }

    // 3. NUC update — motion-gated LMS
    let corrected: Vec<f32> = f.iter().zip(nuc_s.iter()).map(|(v, o)| v - o).collect();

    // Sub-sample 1/16 for fast motion estimate (explicit loops avoid closure capture)
    let sub_n = (width / 4) * (height / 4);
    let mae: f32 = if sub_n > 0 {
        let mut motion_sum: f32 = 0.0;
        for r in 0..height / 4 {
            for c in 0..width / 4 {
                let idx = r * 4 * width + c * 4;
                motion_sum += (f[idx] - prev_s[idx]).abs();
            }
        }
        motion_sum / sub_n as f32
    } else { 0.0 };

    if mae > thresh {
        let mut tmp     = vec![0f32; n];
        let mut desired = vec![0f32; n];
        box_blur_f32(&corrected, &mut tmp, &mut desired, width, height, 6);
        nuc_s.iter_mut()
             .zip(corrected.iter())
             .zip(desired.iter())
             .for_each(|((o, c), d)| *o += mu * (c - d));
    }
    prev_s.copy_from_slice(&f);

    // 4. Normalize corrected → [0, 255]
    let (mn, mx) = corrected.iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let scale = if mx > mn { 255.0 / (mx - mn) } else { 1.0 };

    // 5. Apply Iron LUT + optional nearest-neighbour resize — write directly
    //    into the caller-supplied out_buf (zero extra Vec allocation / copy).
    let ow = if out_width  == 0 { width  } else { out_width  };
    let oh = if out_height == 0 { height } else { out_height };

    let mut out_buf = out_buf;
    let out_s = out_buf.as_slice_mut()?;
    assert_eq!(out_s.len(), ow * oh * 3, "out_buf length must be out_width*out_height*3");

    if ow == width && oh == height {
        // No resize — straight colourize, parallel by pixel
        out_s.par_chunks_mut(3).zip(corrected.par_iter()).for_each(|(pix, &v)| {
            let idx = ((v - mn) * scale).round().clamp(0.0, 255.0) as usize;
            let [b, g, r] = lut[idx];
            pix[0] = b; pix[1] = g; pix[2] = r;
        });
    } else {
        // INTER_NEAREST upscale fused with LUT — parallel by output row
        out_s.par_chunks_mut(ow * 3).enumerate().for_each(|(y_out, row)| {
            let y_in = y_out * height / oh;
            for x_out in 0..ow {
                let x_in  = x_out * width / ow;
                let v     = corrected[y_in * width + x_in];
                let idx   = ((v - mn) * scale).round().clamp(0.0, 255.0) as usize;
                let [b, g, r] = lut[idx];
                let off = x_out * 3;
                row[off] = b; row[off + 1] = g; row[off + 2] = r;
            }
        });
    }
    Ok(())
}

#[pymodule]
fn thermal_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_frame, m)?)?;
    Ok(())
}
