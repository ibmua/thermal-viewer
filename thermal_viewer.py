#!/usr/bin/env python3
"""
thermal_viewer.py — Open-source USB thermal camera viewer
https://github.com/sharpy/thermal-viewer

Supported cameras (auto-detected by resolution):
  FLIR Boson 640 (640×512), FLIR Boson 320 (320×256),
  FLIR Lepton 3.x via PureThermal (160×120),
  InfiRay / Xinfrared (256×192), Seek Compact (320×240),
  any generic UVC thermal camera

Controls:
  Left-click image   Place marker        Right-click   Remove nearest marker
  Sidebar buttons    Clickable           Mouse hover   Highlights
  SPACE              Freeze / unfreeze   R             Record video on/off
  C                  Cycle colormap      N             Cycle view (Live/NUC Map/Raw)
  D                  Motion NUC toggle   F / Shift+F   Flat-field calibrate
  T                  Temporal smooth     H / V         Flip H / V
  S                  Save snapshot       A             Cycle audio input
  Q / Esc            Quit
"""

import cv2, numpy as np, time, os, sys, threading, subprocess, shutil, queue, re, wave
from collections import deque

try:
    import thermal_core as _tc
    _HAS_TC = True
except ImportError:
    _HAS_TC = False

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

try:
    import AVFoundation
    import Foundation
    _HAS_AVAUDIORECORDER = True
except ImportError:
    _HAS_AVAUDIORECORDER = False
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List

# ── Sensor profiles ───────────────────────────────────────────────────────────
@dataclass
class SensorProfile:
    name: str
    w: int
    h: int
    fps: int
    note: str = ""

SENSOR_PROFILES: List[SensorProfile] = [
    SensorProfile("FLIR Boson 640",        640, 512, 60, "USB Boson core module"),
    SensorProfile("FLIR Boson 320",        320, 256, 60, "USB Boson core module"),
    SensorProfile("FLIR Lepton 3.x",       160, 120,  9, "PureThermal USB board"),
    SensorProfile("InfiRay / Xinfrared",   256, 192, 25, "USB-C thermal camera"),
    SensorProfile("Seek Compact",          320, 240, 15, "USB Seek Thermal"),
    SensorProfile("Generic UVC 640x480",   640, 480, 30, ""),
    SensorProfile("Generic UVC 320x240",   320, 240, 30, ""),
]

def match_sensor(w: int, h: int) -> SensorProfile:
    for s in SENSOR_PROFILES:
        if s.w == w and s.h == h:
            return s
    return SensorProfile(f"Unknown {w}x{h}", w, h, 30, "")

# ── Layout (computed after sensor detection) ──────────────────────────────────
TARGET_IW = 960     # target display width for camera image
SB_W      = 268     # sidebar width
BAR_H     = 32      # status bar height

# These are initialised in setup_layout():
sensor    : Optional[SensorProfile] = None
CAM_W, CAM_H = 640, 512    # overwritten
SCALE = 1.5
IW    = 960
IH    = 768
WIN_W = IW + SB_W
WIN_H = IH + BAR_H
SCALE_INT = 1   # integer multiplier used by Rust norm8 sub-sample

def setup_layout(s: SensorProfile):
    global sensor, CAM_W, CAM_H, SCALE, IW, IH, WIN_W, WIN_H, SCALE_INT
    sensor = s
    CAM_W, CAM_H = s.w, s.h
    SCALE  = TARGET_IW / s.w
    IW     = int(s.w * SCALE)
    IH     = int(s.h * SCALE)
    SCALE_INT = max(1, round(SCALE))
    WIN_W  = IW + SB_W
    WIN_H  = IH + BAR_H

WIN_NAME = "Thermal Viewer"

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
BG        = ( 13,  13,  13)
BG_CARD   = ( 21,  21,  21)
BG_DARK   = (  8,   8,   8)
BORDER    = ( 36,  36,  36)
BORDER_HI = ( 70,  70,  70)
C_BRIGHT  = (230, 230, 230)
C_MED     = (158, 158, 158)
C_DIM     = ( 78,  78,  78)
C_GREEN   = ( 55, 200,  80)
C_ACCENT  = (  0, 218, 168)
C_ORANGE  = (  0, 165, 255)
C_BLUE    = (200, 138,  42)
C_RED     = ( 62,  72, 212)
C_FROST   = (188, 138,  48)
C_REC     = ( 50,  50, 220)   # recording red

MARKER_COLORS = [
    (  0, 230, 255), (255, 215,   0), ( 60,  60, 255),
    ( 50, 220,  50), (255,  50, 255), (  0, 145, 255),
    (200, 180,  70), (255, 255, 255),
]

CMAPS = [
    ("Iron",    None),
    ("Inferno", cv2.COLORMAP_INFERNO),
    ("Turbo",   cv2.COLORMAP_TURBO),
    ("Jet",     cv2.COLORMAP_JET),
    ("Hot",     cv2.COLORMAP_HOT),
    ("Gray",    None),
]
VIEW_MODES = ["Live", "NUC Map", "Raw"]

# ── Iron LUT ──────────────────────────────────────────────────────────────────
def _build_iron_lut() -> np.ndarray:
    stops = [(0,(0,0,0)),(64,(102,0,102)),(128,(0,0,204)),
             (192,(0,153,255)),(255,(255,255,255))]
    lut = np.zeros((256,1,3), np.uint8)
    for i in range(256):
        for j in range(len(stops)-1):
            v0,c0 = stops[j]; v1,c1 = stops[j+1]
            if v0 <= i <= v1:
                f = (i-v0)/(v1-v0)
                lut[i,0] = [int(c0[k]+f*(c1[k]-c0[k])) for k in range(3)]
                break
    return lut
IRON_LUT      = _build_iron_lut()
IRON_LUT_FLAT = IRON_LUT[:, 0, :].ravel().copy()   # (768,) uint8 for thermal_core

def colorize(gray8: np.ndarray, idx: int) -> np.ndarray:
    name, cv_cm = CMAPS[idx]
    if name == "Iron":
        # IRON_LUT shape (256,1,3); fancy-index dim-0 → (..., 1, 3), squeeze dim -2
        return IRON_LUT[gray8, 0]          # vectorised, no Python loop
    if name == "Gray": return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray8, cv_cm)

def cmap_color_at(v255: int) -> Tuple[int,int,int]:
    """BGR color for 0-255 value in current colormap."""
    g = np.array([[int(v255)]], dtype=np.uint8)   # 2D shape (1,1)
    return tuple(int(x) for x in colorize(g, cm_idx)[0, 0])

# ── FPN Noise Correction ─────────────────────────────────────────────────────

class MotionGatedNUC:
    """Scene-based LMS NUC — learns fixed noise pattern from scene motion."""
    def __init__(self, mu=0.004, thresh=4.0, sigma=2):
        self.mu, self.thresh, self.sigma = mu, thresh, sigma
        # Pre-compute kernel size: 2*ceil(3σ)+1, must be odd
        ks = int(2 * np.ceil(3 * sigma) + 1) | 1
        self._ksize = (ks, ks)   # sigma=2 → 13×13 (vs sigma=11 → 67×67, 25× faster)
        self.O    : Optional[np.ndarray] = None
        self.prev : Optional[np.ndarray] = None
        self.updates = 0

    def _ensure(self):
        if self.O is None:
            self.O = np.zeros((CAM_H, CAM_W), np.float32)

    def update(self, f32: np.ndarray) -> np.ndarray:
        self._ensure()
        corrected = f32 - self.O
        if self.prev is not None:
            # Sub-sample 1-in-4 in each axis (1/16 of pixels) for motion estimate —
            # statistically identical result, ~16× less memory/arithmetic work.
            mae = float(np.mean(np.abs(f32[::4, ::4] - self.prev[::4, ::4])))
            if mae > self.thresh:
                # Box blur (integral image, O(n) regardless of kernel size) is
                # ~26x faster than separable Gaussian for a 13x13 kernel and
                # produces visually identical NUC results — the offset estimate
                # only needs low-frequency spatial info.
                desired = cv2.blur(corrected, self._ksize)
                self.O  += self.mu * (corrected - desired)
                self.updates += 1
        self.prev = f32.copy()
        return corrected

    def reset(self):
        self.O = None; self.prev = None; self.updates = 0


class FlatFieldNUC:
    """Flat-field calibration — slowly slide camera over a uniform surface.

    Captures N frames while the camera moves, then detects bad pixels via
    two independent methods:

    Method A — Stuck / frozen pixels
        Temporal std AND temporal range are far below the sensor average.
        A truly stuck pixel never responds to scene changes so both metrics
        are near-zero.  Threshold: std < 25 % of median std, or range < 20 %
        of median range.  Moving the camera during calibration maximises the
        scene variation seen by good pixels, making stuck ones stand out more.

    Method B — Persistent spatial outliers (hot / cold pixels)
        Each pixel's mean is compared to its 9×9 Gaussian-blurred local
        average.  We use a MAD-based robust sigma so that the threshold
        adapts to sensor quality rather than always flagging a fixed fraction.
        Pixels more than 5 robust-σ above the median local deviation are
        flagged.  Using a fixed percentile threshold (original code) always
        flagged exactly 0.8 % regardless of sensor quality → false positives.

    Both masks are OR'd.  Bad pixels are replaced with the weighted mean of
    their GOOD neighbours (5×5 kernel, bad pixels excluded from the kernel so
    clusters don't contaminate each other).

    Runtime discovery (see _rtbp_update) adds further bad pixels over time
    as the camera operates normally.  Those are tracked separately from the
    calibration masks so they can be revalidated and removed later if they
    stop behaving like isolated bad pixels.

    Two calibration modes (F vs Shift+F):

    'offset' (F) — colour calibration only.
        32 frames, uniform surface or lens cap.  Recomputes the per-pixel
        additive offset so colour / brightness looks correct after the sensor
        warms up.  Existing calibration bad pixels are revalidated, but this
        quick mode does not aggressively discover new ones from scratch, since
        random live scenes can otherwise create large false-positive clusters.
        Runtime-confirmed pixels are preserved separately and revalidated
        during live use.

    'full' (Shift+F) — full calibration.
        128 frames, slide slowly over a uniform surface.  Recomputes both the
        offset correction AND the complete dead-pixel map from scratch.  Use
        this when first setting up or after replacing the camera.
    """
    N_OFFSET = 32    # frames for colour-only calibration
    N_FULL   = 128   # frames for full calibration

    def __init__(self):
        self.correction  : Optional[np.ndarray] = None
        self.bad_mask    : Optional[np.ndarray] = None   # all bad pixels combined
        self.stuck_mask  : Optional[np.ndarray] = None   # method A only
        self.outlier_mask: Optional[np.ndarray] = None   # method B only
        self.runtime_mask: Optional[np.ndarray] = None   # runtime-found bad pixels
        self.n_bad = self.n_stuck = self.n_outlier = self.n_runtime = 0
        self.enabled     = False
        self.calibrating = False
        self._mode       = 'offset'   # current calibration mode
        self.N           = self.N_OFFSET
        self._buf: List[np.ndarray] = []
        self._done_t: float = 0
        # Pre-built flat contiguous arrays for Rust — recomputed only when the
        # masks change (calibration done / runtime pixel added), NOT every frame.
        self.corr_flat: Optional[np.ndarray] = None   # float32 (CAM_H*CAM_W,)
        self.bad_flat : Optional[np.ndarray] = None   # uint8   (CAM_H*CAM_W,)
        self._load()

    @property
    def savepath(self) -> str:
        d = os.path.expanduser("~/.thermal_viewer")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"flatfield_{CAM_W}x{CAM_H}.npz")

    def start(self, mode: str = 'offset'):
        """Begin a calibration capture.
        mode='offset'  — F key:       32 frames, revalidates existing calibration bad pixels.
        mode='full'    — Shift+F key: 128 frames, full dead-pixel re-detection.
        """
        self._mode = mode
        self.N     = self.N_FULL if mode == 'full' else self.N_OFFSET
        self._buf  = []; self.calibrating = True; self._done_t = 0

    def _empty_mask(self) -> np.ndarray:
        if self.correction is not None:
            return np.zeros_like(self.correction, dtype=bool)
        return np.zeros((CAM_H, CAM_W), dtype=bool)

    def calibration_mask(self) -> np.ndarray:
        stuck = self.stuck_mask if self.stuck_mask is not None else self._empty_mask()
        out   = self.outlier_mask if self.outlier_mask is not None else self._empty_mask()
        return stuck | out

    def _sync_masks(self, persist: bool = False) -> None:
        if self.stuck_mask is None and self.outlier_mask is None and self.runtime_mask is None:
            self.bad_mask = None
            self.n_bad = self.n_stuck = self.n_outlier = self.n_runtime = 0
            self._rebuild_flat_cache()
            return

        cal_bad = self.calibration_mask()
        runtime = self.runtime_mask if self.runtime_mask is not None else np.zeros_like(cal_bad)
        runtime = runtime & ~cal_bad
        self.runtime_mask = runtime
        self.bad_mask = cal_bad | runtime
        self.n_stuck = int(self.stuck_mask.sum()) if self.stuck_mask is not None else 0
        self.n_outlier = int((self.outlier_mask & ~self.stuck_mask).sum()) \
            if self.outlier_mask is not None and self.stuck_mask is not None else 0
        self.n_runtime = int(runtime.sum())
        self.n_bad = int(self.bad_mask.sum())
        self._rebuild_flat_cache()
        if persist and self.correction is not None:
            np.savez(
                self.savepath,
                c=self.correction,
                b=self.bad_mask.astype(np.uint8),
                bs=self.stuck_mask.astype(np.uint8) if self.stuck_mask is not None else np.zeros_like(self.bad_mask, np.uint8),
                bo=self.outlier_mask.astype(np.uint8) if self.outlier_mask is not None else np.zeros_like(self.bad_mask, np.uint8),
                br=self.runtime_mask.astype(np.uint8) if self.runtime_mask is not None else np.zeros_like(self.bad_mask, np.uint8),
            )

    def feed(self, f32: np.ndarray) -> bool:
        if not self.calibrating: return False
        self._buf.append(f32.copy())
        if len(self._buf) >= self.N: self._finish(); return True
        return False

    def _finish(self):
        stack = np.stack(self._buf, 0)
        mean  = stack.mean(0).astype(np.float32)
        std   = stack.std(0).astype(np.float32)

        # Per-pixel offset correction.  Works correctly for moving or static
        # calibration over a uniform surface — the per-pixel mean converges to
        # the global mean for good pixels; stuck pixels always deviate.
        self.correction = (mean.mean() - mean)

        # ── Dead-pixel detection (run in both modes) ──────────────────────────
        # Method A: stuck / frozen pixels
        med_std   = float(np.median(std))
        pix_range = (stack.max(0) - stack.min(0)).astype(np.float32)
        med_range = float(np.median(pix_range))
        # Compare against both the global distribution and the local temporal
        # activity field.  If a whole region barely moved during calibration we
        # do NOT want to mark that whole region as "stuck".
        local_std   = cv2.blur(std, (9, 9))
        local_range = cv2.blur(pix_range, (9, 9))
        new_stuck = (
            (std < (med_std * 0.25)) &
            (pix_range < (med_range * 0.20)) &
            (std < (local_std * 0.55)) &
            (pix_range < (local_range * 0.55))
        )

        # Method B: persistent spatial outliers (hot / cold pixels)
        local_avg  = cv2.GaussianBlur(mean, (9, 9), 2.0)
        local_dev  = np.abs(mean - local_avg)
        med_dev    = float(np.median(local_dev))
        mad        = float(np.median(np.abs(local_dev - med_dev)))
        rob_sigma  = 1.4826 * mad
        threshold  = max(med_dev + 5.0 * rob_sigma, 1.0)
        new_outlier = local_dev > threshold

        prior_cal = self.calibration_mask() if (self.stuck_mask is not None or self.outlier_mask is not None) \
            else np.zeros_like(new_stuck)
        prior_rt = self.runtime_mask.copy() if self.runtime_mask is not None else np.zeros_like(new_stuck)
        prior_all = prior_cal | prior_rt

        prior_stuck = self.stuck_mask.copy() if self.stuck_mask is not None else np.zeros_like(new_stuck)
        prior_outlier = self.outlier_mask.copy() if self.outlier_mask is not None else np.zeros_like(new_stuck)

        # Full calibration rebuilds the dead-pixel DB from scratch.
        # Colour-only calibration only revalidates the existing calibration
        # masks.  That makes F safe to use during normal operation instead of
        # letting arbitrary scene content generate a giant new bad-pixel block.
        if self._mode == 'full':
            self.stuck_mask = new_stuck
            self.outlier_mask = new_outlier
            self.runtime_mask = np.zeros_like(prior_rt)
        else:
            self.stuck_mask = prior_stuck & new_stuck
            self.outlier_mask = prior_outlier & new_outlier
            self.runtime_mask = prior_rt & ~(self.stuck_mask | self.outlier_mask)
        removed_total = int((prior_all & ~(self.stuck_mask | self.outlier_mask | self.runtime_mask)).sum())

        self.calibrating = False; self.enabled = True; self._done_t = time.time()
        self._buf = []
        self._sync_masks(persist=True)
        mode_tag = "full" if self._mode == 'full' else "colour"
        refresh_note = f", removed {removed_total} stale" if removed_total else ""
        print(f"  Flat-field ({mode_tag}) done — {self.n_bad} bad px "
              f"({self.n_stuck} stuck, {self.n_outlier} hot/cold{refresh_note})")

    def apply(self, f32: np.ndarray) -> np.ndarray:
        if not self.enabled or self.correction is None: return f32
        out = f32 + self.correction
        if self.n_bad > 0:
            # Replace bad pixels with the weighted mean of their GOOD neighbours.
            # Zero bad pixels before blurring so they don't contaminate the
            # kernel average (original bug: blur included the bad pixel itself,
            # so a stuck pixel offset of 1000 ADU leaked ~111 ADU into its own
            # replacement → visible dot remained).
            #
            # cv2.blur(x, k) = Σx / k²  ← identical divisor for all positions
            # → blur(good_f) / blur(good_wt) = Σ(good values) / count(good)
            #   = true mean of good neighbours only.
            good_f  = out.copy()
            good_f[self.bad_mask] = 0.0
            good_wt = (~self.bad_mask).astype(np.float32)
            nbr_sum = cv2.blur(good_f, (5, 5))   # Σ(good_values) / 25
            nbr_cnt = cv2.blur(good_wt, (5, 5))  # count(good) / 25
            # In-place divide where we have at least one good neighbour
            valid = nbr_cnt > 0
            nbr_sum[valid] /= nbr_cnt[valid]
            out[self.bad_mask] = nbr_sum[self.bad_mask]
        return out

    def nuc_map_image(self, ci: int) -> np.ndarray:
        """NUC correction heatmap with dead-pixel locations visually marked.

        Background: correction strength (bright = pixel was too bright, dark = too dark).
        Overlay:
          Orange halo (3×3 dilated) — any bad pixel
          Red centre dot             — stuck / frozen pixel  (method A)
          Cyan centre dot            — hot / cold outlier    (method B)
        """
        if self.correction is None:
            return np.zeros((CAM_H, CAM_W, 3), np.uint8)

        # Stable symmetric scaling around zero makes successive NUC maps far
        # easier to compare.  Plain min/max normalization can make two very
        # similar correction fields look wildly different just because of one
        # extreme outlier pixel.
        lim = float(np.percentile(np.abs(self.correction), 99.5))
        lim = max(lim, 1.0)
        n   = np.clip((self.correction / lim) * 127.5 + 127.5, 0, 255).astype(np.uint8)
        img = colorize(n, ci).copy()

        if self.bad_mask is not None and self.n_bad > 0:
            kern    = np.ones((3, 3), np.uint8)
            halo    = cv2.dilate(self.bad_mask.astype(np.uint8), kern).astype(bool)
            img[halo]                = (30,  150, 255)   # orange halo
            if self.stuck_mask   is not None:
                img[self.stuck_mask]   = (40,  40,  220)   # red   — stuck/dead
            if self.outlier_mask is not None:
                img[self.outlier_mask & ~self.stuck_mask] = (220, 220, 0)  # cyan — hot/cold

        return img

    def _load(self):
        # Savepath requires CAM_W/CAM_H which aren't set at import time;
        # actual loading is deferred to load_if_needed() called after setup_layout().
        pass

    def load_if_needed(self):
        if self.correction is not None: return
        p = self.savepath
        if not os.path.exists(p): return
        try:
            d = np.load(p)
            self.correction   = d["c"]
            loaded_bad        = d["b"].astype(bool)
            self.stuck_mask   = d["bs"].astype(bool) if "bs" in d else loaded_bad
            self.outlier_mask = d["bo"].astype(bool) if "bo" in d else np.zeros_like(loaded_bad)
            if "br" in d:
                self.runtime_mask = d["br"].astype(bool)
            else:
                self.runtime_mask = loaded_bad & ~(self.stuck_mask | self.outlier_mask)
            self.enabled      = True
            self._sync_masks()
            print(f"  Loaded flat-field ({CAM_W}×{CAM_H}) — "
                  f"{self.n_bad} bad px ({self.n_stuck} stuck, {self.n_outlier} hot/cold"
                  f"{f', {self.n_runtime} runtime' if self.n_runtime else ''})")
        except Exception as e:
            print(f"  Could not load flat-field: {e}")

    def _rebuild_flat_cache(self) -> None:
        """Rebuild corr_flat / bad_flat contiguous arrays used by Rust.
        Called after any mask change so the hot path never recomputes them."""
        if self.correction is not None:
            self.corr_flat = np.ascontiguousarray(
                self.correction.ravel().astype(np.float32))
        else:
            self.corr_flat = None
        if self.bad_mask is not None:
            self.bad_flat = np.ascontiguousarray(
                self.bad_mask.ravel().astype(np.uint8))
        else:
            self.bad_flat = None

    def add_runtime_bad(self, new_mask: np.ndarray) -> int:
        """Merge runtime-discovered bad pixels into bad_mask.  Returns count added."""
        if self.runtime_mask is None:
            self.runtime_mask = np.zeros_like(new_mask, dtype=bool)
        newly = new_mask & ~self.runtime_mask & ~self.calibration_mask()
        if not newly.any():
            return 0
        self.runtime_mask |= newly
        self._sync_masks(persist=True)   # keep Rust-side bad_flat in sync + save DB
        return int(newly.sum())

    def remove_runtime_bad(self, clear_mask: np.ndarray) -> int:
        """Remove runtime pixels that no longer behave like bad pixels."""
        if self.runtime_mask is None:
            return 0
        removed = clear_mask & self.runtime_mask
        if not removed.any():
            return 0
        self.runtime_mask = self.runtime_mask & ~removed
        self._sync_masks(persist=True)
        return int(removed.sum())

    @property
    def progress(self): return len(self._buf)/self.N if self.calibrating else 0.0
    @property
    def frames_captured(self): return len(self._buf)


class TemporalSmoother:
    """EMA smoothing — reduces per-frame random noise."""
    ALPHA = 0.35
    def __init__(self): self.smooth=None; self.enabled=False
    def update(self, f32):
        if not self.enabled: return f32
        if self.smooth is None or self.smooth.shape != f32.shape:
            self.smooth=f32.copy(); return self.smooth
        self.smooth = self.ALPHA*f32 + (1-self.ALPHA)*self.smooth
        return self.smooth

# ── Video recorder ────────────────────────────────────────────────────────────

class VideoRecorder:
    """Records the colourised thermal video, optionally with microphone audio.

    Video and audio are captured independently, then muxed together at stop.
    This keeps the microphone capture path isolated from the video encoder so
    transient issues in one path are less likely to glitch the other.  Falls
    back to video-only if no usable audio backend is available.

    Platform audio input:
      macOS   — AVAudioRecorder (preferred), sounddevice fallback, ffmpeg fallback
      Linux   — PulseAudio    (-f pulse -i default)
      Windows — DirectShow    (-f dshow -i audio=default)
    """
    _mac_audio_input_spec: Optional[str] = None
    _audio_input_index: Optional[int] = None
    _audio_input_label: str = ""

    def __init__(self):
        self._writer    : Optional[cv2.VideoWriter] = None
        self._ffmpeg_proc = None   # optional ffmpeg video-only encoder
        self._audio_proc = None    # optional ffmpeg audio-only capture
        self._audio_recorder = None
        self._audio_stream = None  # optional sounddevice capture stream
        self._audio_wave = None    # optional wave writer for sounddevice capture
        self._audio_write_q: Optional["queue.Queue[Optional[bytes]]"] = None
        self._audio_write_thr: Optional[threading.Thread] = None
        self._audio_backend = ""
        self._write_q   : Optional["queue.Queue[Optional[np.ndarray]]"] = None
        self._write_thr : Optional[threading.Thread] = None
        self.path       = ""
        self._vid_tmp   = ""
        self._audio_tmp = ""
        self._t0        = 0.0
        self._video_t0  = 0.0
        self._video_frame_t0 = 0.0
        self._audio_t0  = 0.0
        self._audio_actual_t0 = 0.0
        self._n         = 0
        self._dropped   = 0
        self.has_audio  = False
        self.target_fps = 0.0
        self._ffmpeg_err = ""
        self._audio_final_args: List[str] = ['-c:a', 'copy']
        self._sleep_guard_proc = None
        self._sleep_ping_thr: Optional[threading.Thread] = None
        self._sleep_ping_alive = False
        self.last_result_note = ""

    @property
    def recording(self) -> bool:
        return (
            self._writer is not None or
            (self._ffmpeg_proc is not None and self._ffmpeg_proc.poll() is None)
        )

    @property
    def keeping_screen_awake(self) -> bool:
        return self._sleep_guard_proc is not None and self._sleep_guard_proc.poll() is None

    def _sleep_ping_loop(self) -> None:
        caffeinate = shutil.which('caffeinate')
        if not caffeinate:
            return
        while self._sleep_ping_alive:
            try:
                subprocess.run(
                    [caffeinate, '-u', '-t', '5'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                return

    def _start_sleep_guard(self) -> None:
        if sys.platform != 'darwin':
            return
        caffeinate = shutil.which('caffeinate')
        if not caffeinate:
            return
        self._stop_sleep_guard()
        try:
            self._sleep_guard_proc = subprocess.Popen(
                [caffeinate, '-dims'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._sleep_ping_alive = True
            self._sleep_ping_thr = threading.Thread(
                target=self._sleep_ping_loop, daemon=True)
            self._sleep_ping_thr.start()
        except Exception:
            self._sleep_guard_proc = None
            self._sleep_ping_alive = False
            self._sleep_ping_thr = None

    def _stop_sleep_guard(self) -> None:
        self._sleep_ping_alive = False
        if self._sleep_ping_thr is not None:
            self._sleep_ping_thr.join(timeout=1.5)
            self._sleep_ping_thr = None
        proc = self._sleep_guard_proc
        self._sleep_guard_proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass
        except Exception:
            pass

    @staticmethod
    def _path_bytes(path: str) -> int:
        try:
            return os.path.getsize(path) if path and os.path.exists(path) else 0
        except Exception:
            return 0

    def _mux_space_budget(self, video_path: str, audio_path: str) -> Tuple[int, int]:
        if not video_path or not audio_path:
            return 0, 0
        base_dir = os.path.dirname(self.path or video_path) or '.'
        try:
            free_bytes = shutil.disk_usage(base_dir).free
        except Exception:
            return 0, 0
        need_bytes = (
            self._path_bytes(video_path) +
            self._path_bytes(audio_path) +
            128 * 1024 * 1024
        )
        return free_bytes, need_bytes

    def current_mux_space_warning(self) -> str:
        if not self.recording or not self.has_audio:
            return ""
        free_bytes, need_bytes = self._mux_space_budget(
            self._vid_tmp or self.path, self._audio_tmp)
        if not need_bytes or free_bytes >= need_bytes:
            return ""
        gib = 1024 ** 3
        return f"Low disk: {free_bytes / gib:.1f}G free, need ~{need_bytes / gib:.1f}G"

    @classmethod
    def _sounddevice_inputs(cls) -> List[dict]:
        if not _HAS_SD:
            return []
        try:
            devices = []
            for idx, d in enumerate(sd.query_devices()):
                max_in = int(d.get('max_input_channels', 0))
                if max_in > 0:
                    devices.append({
                        'index': idx,
                        'name': str(d.get('name', f'Input {idx}')),
                        'channels': max_in,
                        'samplerate': int(d.get('default_samplerate', 48000) or 48000),
                    })
            return devices
        except Exception:
            return []

    @classmethod
    def _mac_default_input_label(cls) -> str:
        try:
            sp = subprocess.run(
                ['system_profiler', 'SPAudioDataType'],
                capture_output=True, text=True, timeout=5)
        except Exception:
            return ""

        default_source = ""
        default_device = ""
        current_name = ""
        for line in sp.stdout.splitlines():
            m = re.match(r'^\s{8}(.+):\s*$', line)
            if m:
                current_name = m.group(1)
                continue
            if 'Input Source: Default' in line and current_name:
                default_source = current_name
            if 'Default Input Device: Yes' in line and current_name:
                default_device = current_name
        return default_source or default_device

    @classmethod
    def _choose_sounddevice_input(cls, advance: int = 0) -> dict:
        inputs = cls._sounddevice_inputs()
        if not inputs:
            raise RuntimeError("no CoreAudio input devices found")

        indices = [d['index'] for d in inputs]
        if cls._audio_input_index in indices:
            pos = indices.index(cls._audio_input_index)
            choice = inputs[(pos + advance) % len(inputs)]
        else:
            default_idx = None
            try:
                default_idx = int(sd.default.device[0])
            except Exception:
                pass
            choice = None
            # Prefer an external hardware mic over the built-in or virtual inputs.
            for d in inputs:
                lname = d['name'].lower()
                if 'teams' in lname:
                    continue
                if 'dji' in lname or ('macbook' not in lname and 'built-in' not in lname):
                    choice = d
                    break
            if choice is None and default_idx in indices:
                choice = inputs[indices.index(default_idx)]
            if choice is None:
                choice = inputs[0]

        cls._audio_input_index = int(choice['index'])
        cls._audio_input_label = f"{choice['name']} (sd:{choice['index']})"
        return choice

    @classmethod
    def cycle_audio_input(cls) -> str:
        if sys.platform == 'darwin' and _HAS_AVAUDIORECORDER:
            return ""
        if sys.platform == 'darwin' and _HAS_SD:
            try:
                cls._choose_sounddevice_input(advance=1)
                return cls._audio_input_label
            except Exception:
                return ""
        return ""

    @classmethod
    def _audio_input_args(cls) -> list:
        """ffmpeg input flags for the default microphone on this platform."""
        if sys.platform == 'darwin':
            return cls._mac_audio_input_args()
        elif sys.platform == 'win32':
            cls._audio_input_label = "default DirectShow audio device"
            return ['-f', 'dshow', '-i', 'audio=default']
        else:   # Linux / BSD
            cls._audio_input_label = "default PulseAudio input"
            return ['-f', 'pulse', '-i', 'default']

    @classmethod
    def _mac_audio_input_args(cls) -> list:
        if cls._mac_audio_input_spec is not None:
            return ['-f', 'avfoundation', '-i', cls._mac_audio_input_spec]

        default_name = ""
        try:
            sp = subprocess.run(
                ['system_profiler', 'SPAudioDataType'],
                capture_output=True, text=True, timeout=5)
            current_name = ""
            for line in sp.stdout.splitlines():
                m = re.match(r'^\s{8}(.+):\s*$', line)
                if m:
                    current_name = m.group(1)
                    continue
                if 'Default Input Device: Yes' in line and current_name:
                    default_name = current_name
                    break
        except Exception:
            pass

        devices = []
        try:
            r = subprocess.run(
                ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
                capture_output=True, text=True, timeout=5)
            out = (r.stderr or '') + '\n' + (r.stdout or '')
            in_audio = False
            for line in out.splitlines():
                if 'AVFoundation audio devices:' in line:
                    in_audio = True
                    continue
                if not in_audio:
                    continue
                m = re.search(r'\[(\d+)\]\s+(.+)$', line)
                if m:
                    devices.append((m.group(1), m.group(2).strip()))
        except Exception:
            pass

        spec = ':0'
        selected_name = ""
        if default_name:
            for idx, name in devices:
                if name == default_name:
                    spec = f':{idx}'
                    selected_name = name
                    break
        if spec == ':0':
            for idx, name in devices:
                if 'Microphone' in name:
                    spec = f':{idx}'
                    selected_name = name
                    break
        if not selected_name and devices:
            audio_idx = spec[1:] if spec.startswith(':') else spec
            for idx, name in devices:
                if idx == audio_idx:
                    selected_name = name
                    break
        cls._mac_audio_input_spec = spec
        cls._audio_input_label = f"{selected_name} ({spec})" if selected_name else spec
        return ['-f', 'avfoundation', '-i', spec]

    @classmethod
    def audio_input_label(cls) -> str:
        if cls._audio_input_label:
            return cls._audio_input_label
        try:
            cls.refresh_audio_input()
        except Exception:
            pass
        return cls._audio_input_label

    @classmethod
    def refresh_audio_input(cls) -> None:
        cls._audio_input_label = ""
        if sys.platform == 'darwin' and _HAS_AVAUDIORECORDER:
            label = cls._mac_default_input_label()
            if label:
                cls._audio_input_label = f"{label} (system default)"
                return
        if sys.platform == 'darwin' and _HAS_SD:
            try:
                cls._choose_sounddevice_input()
                return
            except Exception:
                pass
        cls._mac_audio_input_spec = None
        try:
            cls._audio_input_args()
        except Exception:
            pass

    @staticmethod
    def probe_audio() -> bool:
        """Return True if ffmpeg can open the default microphone right now.

        Starts a real capture subprocess, waits 400 ms, checks it is still
        running, then kills it.  This is the only reliable way to detect
        macOS permission denials, missing devices, etc.
        """
        if sys.platform == 'darwin' and _HAS_AVAUDIORECORDER:
            path = os.path.join('/tmp', f'codex_probe_{os.getpid()}.m4a')
            try:
                url = Foundation.NSURL.fileURLWithPath_(path)
                settings = {
                    AVFoundation.AVFormatIDKey: AVFoundation.kAudioFormatMPEG4AAC,
                    AVFoundation.AVSampleRateKey: 48000.0,
                    AVFoundation.AVNumberOfChannelsKey: 1,
                    AVFoundation.AVEncoderBitRateKey: 128000,
                    AVFoundation.AVEncoderAudioQualityKey: AVFoundation.AVAudioQualityHigh,
                }
                rec, err = AVFoundation.AVAudioRecorder.alloc().initWithURL_settings_error_(
                    url, settings, None)
                if rec is None:
                    return False
                if not rec.prepareToRecord() or not rec.record():
                    return False
                time.sleep(0.2)
                rec.stop()
                time.sleep(0.05)
                VideoRecorder.refresh_audio_input()
                return True
            except Exception:
                return False
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
        if sys.platform == 'darwin' and _HAS_SD:
            try:
                choice = VideoRecorder._choose_sounddevice_input()
                channels = max(1, min(2, int(choice['channels'])))
                stream = sd.RawInputStream(
                    samplerate=int(choice['samplerate']),
                    blocksize=1024,
                    channels=channels,
                    dtype='int16',
                    device=int(choice['index']),
                )
                stream.start()
                time.sleep(0.2)
                stream.stop()
                stream.close()
                return True
            except Exception:
                return False
        if not shutil.which('ffmpeg'):
            return False
        try:
            VideoRecorder.refresh_audio_input()
            proc = subprocess.Popen(
                ['ffmpeg', '-y'] + VideoRecorder._audio_input_args() +
                ['-ar', '44100', '-ac', '1', '-t', '1', '-f', 'null', '-'],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.4)
            rc = proc.poll()
            proc.terminate()
            try: proc.wait(timeout=2)
            except Exception: pass
            # If it exited in <0.4 s it almost certainly failed immediately
            return rc is None
        except Exception:
            return False

    @staticmethod
    def _ffmpeg_video_arg_sets() -> List[List[str]]:
        if sys.platform == 'darwin':
            return [
                # Prefer the hardware encoder on macOS so 60 fps capture stays
                # reliable, but give it substantially more bitrate than before.
                ['-c:v', 'h264_videotoolbox', '-allow_sw', '1',
                 '-b:v', '24M', '-maxrate', '32M', '-bufsize', '48M'],
                ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14'],
                ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '16'],
                ['-c:v', 'mpeg4', '-q:v', '2'],
            ]
        return [
            ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14'],
            ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '16'],
            ['-c:v', 'mpeg4', '-q:v', '2'],
        ]

    @staticmethod
    def _read_process_error(proc) -> str:
        if proc is None or proc.stderr is None:
            return ""
        try:
            err = proc.stderr.read().decode(errors='replace').strip()
        except Exception:
            return ""
        return err[-400:] if err else ""

    def _audio_writer_loop(self):
        while self._audio_write_q is not None:
            chunk = self._audio_write_q.get()
            if chunk is None:
                break
            try:
                if self._audio_wave is not None:
                    self._audio_wave.writeframesraw(chunk)
            except Exception:
                break

    def start(self, w: int, h: int, fps: float, with_audio: bool = True):
        d = os.path.expanduser("~/Desktop/BosonCaptures")
        os.makedirs(d, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path        = os.path.join(d, f"thermal_{ts}.mp4")
        self._vid_tmp    = ""
        self._audio_tmp  = ""
        self._writer = None
        self._ffmpeg_proc = None
        self._audio_proc = None
        self._audio_recorder = None
        self._audio_stream = None
        self._audio_wave = None
        self._audio_write_q = None
        self._audio_write_thr = None
        self._audio_backend = ""
        self._write_q = None
        self._write_thr = None
        self._t0 = time.time(); self._n = 0
        self._video_t0 = 0.0
        self._video_frame_t0 = 0.0
        self._audio_t0 = 0.0
        self._audio_actual_t0 = 0.0
        self._dropped    = 0
        self.has_audio   = False
        self.target_fps  = max(float(fps), 1.0)
        self._ffmpeg_err = ""
        self._audio_final_args = ['-c:a', 'copy']
        self.last_result_note = ""

        def _start_async_writer():
            self._write_q = queue.Queue(maxsize=32)
            self._write_thr = threading.Thread(target=self._writer_loop, daemon=True)
            self._write_thr.start()

        def _open_writer(path: str):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(path, fourcc, self.target_fps, (w, h))
            if not self._writer.isOpened():
                self._writer = None
                raise RuntimeError(f"VideoWriter failed to open for {path}")
            self._video_t0 = time.perf_counter()
            _start_async_writer()

        def _open_video_ffmpeg_writer(path: str):
            if not shutil.which('ffmpeg'):
                raise FileNotFoundError("ffmpeg not found in PATH")

            last_err = ""
            for video_args in self._ffmpeg_video_arg_sets():
                cmd = [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-thread_queue_size', '512',
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-video_size', f'{w}x{h}',
                    '-framerate', f'{self.target_fps:.3f}',
                    '-i', 'pipe:0',
                ] + video_args + [
                    '-pix_fmt', 'yuv420p',
                    '-an',
                    path,
                ]
                proc = None
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        bufsize=0,
                    )
                    time.sleep(0.25)
                    rc = proc.poll()
                    if rc is not None:
                        last_err = self._read_process_error(proc)
                        continue
                    if proc.stdin is None:
                        raise RuntimeError("ffmpeg stdin unavailable")
                    self._ffmpeg_proc = proc
                    self._video_t0 = time.perf_counter()
                    _start_async_writer()
                    return
                except Exception as e:
                    last_err = f"{e.__class__.__name__}: {e}"
                finally:
                    if proc is not None and proc is not self._ffmpeg_proc and proc.poll() is None:
                        try: proc.terminate()
                        except Exception: pass
            raise RuntimeError(last_err or "ffmpeg video pipeline failed to start")

        def _open_audio_capture(path: str):
            if sys.platform == 'darwin' and _HAS_AVAUDIORECORDER:
                url = Foundation.NSURL.fileURLWithPath_(path)
                settings = {
                    AVFoundation.AVFormatIDKey: AVFoundation.kAudioFormatMPEG4AAC,
                    AVFoundation.AVSampleRateKey: 48000.0,
                    AVFoundation.AVNumberOfChannelsKey: 1,
                    AVFoundation.AVEncoderBitRateKey: 160000,
                    AVFoundation.AVEncoderAudioQualityKey: AVFoundation.AVAudioQualityHigh,
                }
                rec, err = AVFoundation.AVAudioRecorder.alloc().initWithURL_settings_error_(
                    url, settings, None)
                if rec is None:
                    raise RuntimeError(f"AVAudioRecorder failed to open for {path}")
                if not rec.prepareToRecord():
                    raise RuntimeError("AVAudioRecorder prepareToRecord failed")
                if not rec.record():
                    raise RuntimeError("AVAudioRecorder record() failed")
                self._audio_recorder = rec
                self._audio_backend = 'avaudiorecorder'
                self._audio_t0 = time.perf_counter()
                self._audio_actual_t0 = self._audio_t0
                self._audio_final_args = ['-c:a', 'copy']
                self.refresh_audio_input()
                return

            if sys.platform == 'darwin' and _HAS_SD:
                choice = self._choose_sounddevice_input()
                channels = max(1, min(2, int(choice['channels'])))
                samplerate = int(choice['samplerate']) or 48000
                try:
                    self._audio_wave = wave.open(path, 'wb')
                    self._audio_wave.setnchannels(channels)
                    self._audio_wave.setsampwidth(2)
                    self._audio_wave.setframerate(samplerate)
                    self._audio_write_q = queue.Queue(maxsize=256)
                    self._audio_write_thr = threading.Thread(
                        target=self._audio_writer_loop, daemon=True)
                    self._audio_write_thr.start()

                    def _cb(indata, frames, time_info, status):
                        try:
                            if self._audio_write_q is not None:
                                self._audio_write_q.put_nowait(bytes(indata))
                        except queue.Full:
                            pass

                    self._audio_stream = sd.RawInputStream(
                        samplerate=samplerate,
                        blocksize=1024,
                        channels=channels,
                        dtype='int16',
                        device=int(choice['index']),
                        callback=_cb,
                    )
                    self._audio_stream.start()
                except Exception:
                    if self._audio_stream is not None:
                        try: self._audio_stream.close()
                        except Exception: pass
                        self._audio_stream = None
                    if self._audio_wave is not None:
                        try: self._audio_wave.close()
                        except Exception: pass
                        self._audio_wave = None
                    raise
                self._audio_backend = 'sounddevice'
                self._audio_t0 = time.perf_counter()
                self._audio_actual_t0 = self._audio_t0
                self._audio_final_args = ['-c:a', 'aac', '-b:a', '160k']
                return

            if not shutil.which('ffmpeg'):
                raise FileNotFoundError("ffmpeg not found in PATH")

            self.refresh_audio_input()
            cmd = (
                ['ffmpeg', '-y', '-loglevel', 'error', '-thread_queue_size', '512']
                + self._audio_input_args()
                + ['-ac', '1', '-c:a', 'aac', '-b:a', '128k', path]
            )
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.25)
            rc = proc.poll()
            if rc is not None:
                err = self._read_process_error(proc)
                raise RuntimeError(err or f"ffmpeg audio capture exited with code {rc}")
            self._audio_proc = proc
            self._audio_backend = 'ffmpeg'
            self._audio_t0 = time.perf_counter()
            self._audio_actual_t0 = self._audio_t0
            self._audio_final_args = ['-c:a', 'copy']

        def _open_video_output(path: str):
            try:
                _open_video_ffmpeg_writer(path)
            except Exception as e:
                # Keep recording even if ffmpeg video encode is unavailable.
                self._ffmpeg_err = f"{e.__class__.__name__}: {e}"
                _open_writer(path)

        target_video = self.path
        if with_audio:
            self._vid_tmp = os.path.join(d, f"thermal_{ts}_vid.mp4")
            if sys.platform == 'darwin' and _HAS_AVAUDIORECORDER:
                self._audio_tmp = os.path.join(d, f"thermal_{ts}_audio.m4a")
            elif sys.platform == 'darwin' and _HAS_SD:
                self._audio_tmp = os.path.join(d, f"thermal_{ts}_audio.wav")
            else:
                self._audio_tmp = os.path.join(d, f"thermal_{ts}_audio.m4a")
            target_video = self._vid_tmp
            try:
                _open_audio_capture(self._audio_tmp)
                self.has_audio = True
            except Exception as e:
                hint = ""
                msg = str(e)
                if sys.platform == 'darwin' and 'permission' in msg.lower():
                    hint = " — grant Microphone access to Terminal in System Settings"
                print(f"  Audio unavailable ({e.__class__.__name__}: {e}){hint} — recording video only")

        _open_video_output(target_video)
        self._start_sleep_guard()

        if not with_audio:
            print(f"  Recording → {self.path}  @{self.target_fps:.0f}fps")
            return
        audio_tag = ""
        if self.has_audio:
            label = self.audio_input_label()
            audio_tag = f" + audio [{label}]" if label else " + audio"
        print(f"  Recording → {self.path}{audio_tag}  @{self.target_fps:.0f}fps")

    def _writer_loop(self):
        while self._write_q is not None:
            item = self._write_q.get()
            if item is None:
                break
            frame, frame_time = item
            try:
                if frame_time and self._video_frame_t0 == 0.0:
                    self._video_frame_t0 = frame_time
                if self._ffmpeg_proc is not None:
                    if self._ffmpeg_proc.stdin is None:
                        raise BrokenPipeError("ffmpeg stdin closed")
                    self._ffmpeg_proc.stdin.write(memoryview(frame).cast('B'))
                elif self._writer is not None:
                    self._writer.write(frame)
                else:
                    break
                self._n += 1
            except Exception as e:
                self._dropped += 1
                if self._ffmpeg_proc is not None and not self._ffmpeg_err:
                    self._ffmpeg_err = f"{e.__class__.__name__}: {e}"
                break

    def write(self, frame: np.ndarray, copy_frame: bool = True,
              frame_time: float = 0.0):
        if self._write_q is not None and self.recording:
            queued = frame.copy() if copy_frame else frame
            if not queued.flags.c_contiguous:
                queued = np.ascontiguousarray(queued)
            try:
                self._write_q.put_nowait((queued, frame_time))
            except queue.Full:
                self._dropped += 1

    def stop(self) -> Tuple[str, int, float]:
        dur = time.time() - self._t0
        sync_error = ""
        audio_ok = self.has_audio

        if self._write_q is not None:
            self._write_q.put(None)
        if self._write_thr is not None:
            self._write_thr.join(timeout=30)
            self._write_thr = None
        self._write_q = None

        if self._writer:
            self._writer.release(); self._writer = None

        ffmpeg_proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        if ffmpeg_proc is not None:
            try:
                if ffmpeg_proc.stdin is not None:
                    ffmpeg_proc.stdin.close()
            except Exception:
                pass
            try:
                rc = ffmpeg_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try: ffmpeg_proc.terminate()
                except Exception: pass
                try: rc = ffmpeg_proc.wait(timeout=5)
                except Exception: rc = -1
            err = self._read_process_error(ffmpeg_proc)
            if err and not self._ffmpeg_err:
                self._ffmpeg_err = err
            if rc != 0:
                sync_error = self._ffmpeg_err or f"ffmpeg exited with code {rc}"
                audio_ok = False

        if self._audio_backend == 'avaudiorecorder':
            try:
                if self._audio_recorder is not None and self._audio_t0:
                    wall_elapsed = time.perf_counter() - self._audio_t0
                    recorded_elapsed = float(self._audio_recorder.currentTime())
                    startup_lag = max(0.0, wall_elapsed - recorded_elapsed)
                    self._audio_actual_t0 = self._audio_t0 + startup_lag
                if self._audio_recorder is not None:
                    self._audio_recorder.stop()
                # Give AVAudioRecorder a moment to finalize container metadata.
                time.sleep(0.1)
            except Exception as e:
                sync_error = f"{e.__class__.__name__}: {e}"
                audio_ok = False
            self._audio_recorder = None
        elif self._audio_backend == 'sounddevice':
            try:
                if self._audio_stream is not None:
                    self._audio_stream.stop()
                    self._audio_stream.close()
            except Exception as e:
                sync_error = f"{e.__class__.__name__}: {e}"
                audio_ok = False
            self._audio_stream = None
            if self._audio_write_q is not None:
                self._audio_write_q.put(None)
            if self._audio_write_thr is not None:
                self._audio_write_thr.join(timeout=10)
                self._audio_write_thr = None
            self._audio_write_q = None
            if self._audio_wave is not None:
                try:
                    self._audio_wave.close()
                except Exception as e:
                    sync_error = f"{e.__class__.__name__}: {e}"
                    audio_ok = False
                self._audio_wave = None
        else:
            audio_proc = self._audio_proc
            self._audio_proc = None
            if audio_proc is not None:
                try:
                    if audio_proc.stdin is not None:
                        audio_proc.stdin.write(b'q')
                        audio_proc.stdin.flush()
                        audio_proc.stdin.close()
                    rc = audio_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try: audio_proc.terminate()
                    except Exception: pass
                    try: rc = audio_proc.wait(timeout=5)
                    except Exception: rc = -1
                except Exception:
                    try: audio_proc.terminate()
                    except Exception: pass
                    try: rc = audio_proc.wait(timeout=5)
                    except Exception: rc = -1
                err = self._read_process_error(audio_proc)
                if rc != 0:
                    sync_error = err or f"audio capture exited with code {rc}"
                    audio_ok = False
            self._audio_proc = None
        self._audio_backend = ""

        video_path = self._vid_tmp if self._vid_tmp else self.path
        video_ok = os.path.exists(video_path)
        audio_ok = audio_ok and bool(self._audio_tmp) and os.path.exists(self._audio_tmp)

        if video_ok and video_path != self.path and not audio_ok:
            shutil.move(video_path, self.path)
            video_path = self.path

        if video_ok and audio_ok:
            free_bytes, need_bytes = self._mux_space_budget(video_path, self._audio_tmp)
            if need_bytes and free_bytes < need_bytes:
                sync_error = (f"low disk space ({free_bytes / (1024 * 1024):.0f} MiB free, "
                              f"need about {need_bytes / (1024 * 1024):.0f} MiB for mux)")
                audio_ok = False

        if video_ok and audio_ok:
            video_start = self._video_frame_t0 or self._video_t0
            audio_start = self._audio_actual_t0 or self._audio_t0
            offset = audio_start - video_start if video_start and audio_start else 0.0
            if offset > 0:
                # Audio started after the first captured video frame.  Advance
                # the audio timestamps so the delayed audio content lines up
                # with the already-recorded video instead of keeping dead air.
                mux_inputs = ['-i', video_path, '-itsoffset', f'{-offset:.6f}', '-i', self._audio_tmp]
            elif offset < 0:
                mux_inputs = ['-i', video_path, '-ss', f'{-offset:.6f}', '-i', self._audio_tmp]
            else:
                mux_inputs = ['-i', video_path, '-i', self._audio_tmp]
            try:
                r = subprocess.run(
                    ['ffmpeg', '-y'] + mux_inputs + [
                        '-map', '0:v:0',
                        '-map', '1:a:0',
                        '-c:v', 'copy',
                    ] + self._audio_final_args + [
                        '-movflags', '+faststart',
                        '-shortest',
                        self.path,
                    ],
                    capture_output=True, timeout=120
                )
                if r.returncode == 0:
                    if video_path != self.path and os.path.exists(video_path):
                        os.remove(video_path)
                    if self._audio_tmp and os.path.exists(self._audio_tmp):
                        os.remove(self._audio_tmp)
                else:
                    sync_error = r.stderr.decode(errors='replace')[-300:] or "ffmpeg mux failed"
                    audio_ok = False
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                sync_error = f"{e.__class__.__name__}: {e}"
                audio_ok = False
            if not audio_ok:
                if video_path != self.path and os.path.exists(video_path):
                    shutil.move(video_path, self.path)
                    video_path = self.path

        drop_note = f", dropped {self._dropped}" if self._dropped else ""
        if sync_error and not os.path.exists(self.path):
            self.last_result_note = "Recording failed — final file was not written"
            print(f"  Recording failed (audio/video combine error: {sync_error})")
        elif audio_ok:
            self.last_result_note = ""
            print(f"  Saved — {self._n} frames, {dur:.1f}s{drop_note}, muxed audio → {self.path}")
        elif self.has_audio and sync_error:
            if 'low disk space' in sync_error.lower():
                self.last_result_note = "Saved video only — low disk space blocked audio mux"
            else:
                self.last_result_note = "Saved video only — audio mux failed"
            extra = f" (raw audio kept at {self._audio_tmp})" if self._audio_tmp and os.path.exists(self._audio_tmp) else ""
            print(f"  Saved (audio mux failed: {sync_error}) — "
                  f"{self._n} frames, {dur:.1f}s{drop_note} → {self.path}{extra}")
        else:
            self.last_result_note = ""
            print(f"  Saved — {self._n} frames, {dur:.1f}s{drop_note} → {self.path}")
        self._stop_sleep_guard()
        return self.path, self._n, dur

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0 if self.recording else 0.0

# ── Threaded grabber ──────────────────────────────────────────────────────────

class FrameGrabber:
    """Reads camera on a background thread.

    Live preview can pull only the newest frame for low latency, while the
    recorder can drain every pending frame so short UI stalls do not silently
    drop sensor cadence.
    """
    def __init__(self, cap):
        self._cap    = cap
        self._frames = deque(maxlen=32)
        self._lock   = threading.Lock()
        self._alive  = True
        self.cam_fps: float = 0.0   # delivery rate measured in background thread
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        t0 = time.perf_counter(); cn = 0
        while self._alive:
            ret, f = self._cap.read()
            if ret and f is not None:
                ft = time.perf_counter()
                with self._lock:
                    self._frames.append((ft, f))
                cn += 1
                t1 = time.perf_counter()
                if t1 - t0 >= 1.0:
                    self.cam_fps = cn / (t1 - t0)
                    cn = 0; t0 = t1

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Returns only the newest pending frame and drops older ones."""
        with self._lock:
            if not self._frames:
                return False, None, 0.0
            frame_t, frame = self._frames[-1]
            self._frames.clear()
            return True, frame, frame_t

    def read_all(self) -> List[Tuple[float, np.ndarray]]:
        """Returns all pending frames in capture order."""
        with self._lock:
            if not self._frames:
                return []
            frames = list(self._frames)
            self._frames.clear()
            return frames

    def release(self):
        self._alive = False; self._cap.release()

# ── PyObjC 60fps pre-config ───────────────────────────────────────────────────

def try_force_60fps():
    try:
        import AVFoundation as avf, CoreMedia as cm
        boson_sizes = {
            (s.w, s.h)
            for s in SENSOR_PROFILES
            if "Boson" in s.name and s.fps >= 60
        }
        for dev in avf.AVCaptureDevice.devicesWithMediaType_(avf.AVMediaTypeVideo):
            name = str(dev.localizedName())
            if "FLIR" not in name and "Boson" not in name: continue
            best = None
            best_range = None
            best_rate = 0.0
            best_area = -1
            for fmt in dev.formats():
                d = cm.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
                dims = (int(d.width), int(d.height))
                if dims not in boson_sizes:
                    continue
                fmt_best_range = None
                max_rate = 0.0
                for r in fmt.videoSupportedFrameRateRanges():
                    rate = float(r.maxFrameRate())
                    if rate > max_rate:
                        max_rate = rate
                        fmt_best_range = r
                if max_rate < 60.0:
                    continue
                area = dims[0] * dims[1]
                if best is None or (area, max_rate) > (best_area, best_rate):
                    best = fmt
                    best_range = fmt_best_range
                    best_rate = max_rate
                    best_area = area
            if best is None:
                continue
            locked = dev.lockForConfiguration_(None)
            if isinstance(locked, tuple):
                locked = locked[0]
            if not locked:
                continue
            try:
                dev.setActiveFormat_(best)
                if best_range is not None:
                    try:
                        min_dur = best_range.minFrameDuration()
                        max_dur = best_range.maxFrameDuration()
                        dev.setActiveVideoMinFrameDuration_(min_dur)
                        dev.setActiveVideoMaxFrameDuration_(max_dur)
                    except Exception:
                        pass
            finally:
                dev.unlockForConfiguration()
            print(f"  PyObjC: 60fps set on '{name}'")
    except Exception: pass

# ── Global state ──────────────────────────────────────────────────────────────
markers    : List[dict] = []
flip_h     = flip_v = False
cm_idx     = 0
view_mode  = 0
frozen     = False

norm8     : Optional[np.ndarray] = None
raw_f     : Optional[np.ndarray] = None
frame_stats = {"min":0.,"max":0.,"mean":0.,"delta":0.}

nuc      = MotionGatedNUC()
flatf    = FlatFieldNUC()
smoother = TemporalSmoother()
recorder = VideoRecorder()
denoise  = True

# Probe microphone availability once at startup.
# Only takes ~400 ms and lets us hide the audio toggle entirely on machines
# where audio is unavailable (no backend, no mic, or macOS permission denied).
print("  Probing microphone…", end=" ", flush=True)
_AUDIO_AVAILABLE: bool = VideoRecorder.probe_audio()
if _AUDIO_AVAILABLE:
    _audio_label = VideoRecorder.audio_input_label()
    print(f"available ({_audio_label})" if _audio_label else "available")
else:
    print("unavailable (video-only)")
record_audio: bool = _AUDIO_AVAILABLE   # user-toggled; only relevant when True

# Persistent flat arrays for thermal_core (allocated lazily after sensor detected)
_tc_nuc_offset : Optional[np.ndarray] = None   # float32, shape (CAM_H*CAM_W,)
_tc_prev_frame : Optional[np.ndarray] = None   # float32, shape (CAM_H*CAM_W,)
_tc_out_buf    : Optional[np.ndarray] = None   # uint8,   shape (IH*IW*3,) — Rust writes here

# ── Runtime dead-pixel discovery ───────────────────────────────────────────────
# Every _RTBP_INTERVAL frames we run a quick spatial scan.  A pixel must be
# flagged suspicious in _RTBP_CONFIRM separate scans (spread over many seconds)
# before it is confirmed as bad.  Already-confirmed runtime pixels must then
# survive _RTBP_RECOVER clearly-healthy scans before they are removed again.
_RTBP_INTERVAL = 180   # frames between scans  (~3 s at 60 fps)
_RTBP_CONFIRM  = 4     # suspicious hits needed (~12 s of evidence minimum)
_RTBP_RECOVER  = 8     # clearly-healthy hits needed (~24 s) to forgive runtime px
_rtbp_suspicion: Optional[np.ndarray] = None   # int16 (CAM_H, CAM_W) confidence
_rtbp_recovery : Optional[np.ndarray] = None   # int16 (CAM_H, CAM_W) healthy-hit streak
_rtbp_frame_ctr: int   = 0

def _rtbp_update(gray: np.ndarray) -> None:
    """Runtime bad-pixel scan. Called every frame; expensive work runs only
    every _RTBP_INTERVAL frames so the amortised per-frame cost is trivial.

    Algorithm:
      1. Compute local 5×5 neighbourhood mean of the raw gray frame.
      2. A pixel is 'suspicious' only if it is both a strong global outlier
         and much worse than the local deviation field around it.  That keeps
         real scene edges from being misidentified as dead pixels.
      3. Suspicion decays when a candidate stops looking isolated, so good
         pixels do not accumulate toward confirmation forever.
      4. Confirmed runtime pixels are periodically revalidated and removed
         after enough clearly-healthy scans.
    """
    global _rtbp_suspicion, _rtbp_recovery, _rtbp_frame_ctr
    _rtbp_frame_ctr += 1
    if _rtbp_frame_ctr % _RTBP_INTERVAL != 0:
        return

    if _rtbp_suspicion is None or _rtbp_suspicion.shape != (CAM_H, CAM_W):
        _rtbp_suspicion = np.zeros((CAM_H, CAM_W), np.int16)
    if _rtbp_recovery is None or _rtbp_recovery.shape != (CAM_H, CAM_W):
        _rtbp_recovery = np.zeros((CAM_H, CAM_W), np.int16)

    f32     = gray.astype(np.float32)
    nbr     = cv2.blur(f32, (5, 5))
    dev     = f32 - nbr                          # signed deviation from neighbours
    abs_dev = np.abs(dev)
    gstd    = float(abs_dev.std())
    if gstd < 0.5:                               # featureless / saturated frame
        return

    # Dead pixels are isolated outliers.  Real scene edges often have large
    # deviations too, but their neighbours are also "busy".  Require the pixel
    # to beat both the global scene threshold and the local deviation field.
    local_abs = cv2.blur(abs_dev, (5, 5))
    suspicious = (
        (abs_dev > gstd * 6.0) &
        (abs_dev > 4.0) &
        (abs_dev > (local_abs * 3.0 + 2.0))
    )
    clearly_normal = abs_dev < max(gstd * 2.0, 2.5)

    calib_bad = flatf.calibration_mask()
    runtime_bad = flatf.runtime_mask if flatf.runtime_mask is not None else np.zeros_like(calib_bad)
    candidates = ~calib_bad
    suspicious &= candidates
    clearly_normal &= candidates

    new_candidates = candidates & ~runtime_bad
    if new_candidates.any():
        pos = new_candidates & suspicious
        neg = new_candidates & ~suspicious
        if pos.any():
            _rtbp_suspicion[pos] = np.minimum(_rtbp_suspicion[pos] + 1, _RTBP_CONFIRM)
        if neg.any():
            _rtbp_suspicion[neg] = np.maximum(_rtbp_suspicion[neg] - 1, 0)

    runtime_candidates = candidates & runtime_bad
    if runtime_candidates.any():
        still_bad = runtime_candidates & suspicious
        healthy   = runtime_candidates & clearly_normal
        uncertain = runtime_candidates & ~suspicious & ~clearly_normal
        if still_bad.any():
            _rtbp_recovery[still_bad] = 0
        if healthy.any():
            _rtbp_recovery[healthy] = np.minimum(_rtbp_recovery[healthy] + 1, _RTBP_RECOVER)
        if uncertain.any():
            _rtbp_recovery[uncertain] = np.maximum(_rtbp_recovery[uncertain] - 1, 0)

    newly_confirmed = new_candidates & (_rtbp_suspicion >= _RTBP_CONFIRM)
    if newly_confirmed.any():
        n_added = flatf.add_runtime_bad(newly_confirmed)
        if n_added:
            _rtbp_suspicion[newly_confirmed] = 0
            _rtbp_recovery[newly_confirmed] = 0
            print(f"  Runtime scan: +{n_added} bad px confirmed  "
                  f"(total {flatf.n_bad})")

    newly_recovered = runtime_candidates & (_rtbp_recovery >= _RTBP_RECOVER)
    if newly_recovered.any():
        n_removed = flatf.remove_runtime_bad(newly_recovered)
        if n_removed:
            _rtbp_suspicion[newly_recovered] = 0
            _rtbp_recovery[newly_recovered] = 0
            print(f"  Runtime scan: -{n_removed} runtime px removed  "
                  f"(total {flatf.n_bad})")

status_msg   = "Ready"
nuc_auto_t   = 0.0
mouse_x = mouse_y = 0
_last_saved_path = ""

# Sidebar click hitboxes — cleared and rebuilt each frame
_hitboxes: List[Tuple[int,int,int,int,object]] = []  # (abs_x,y,w,h,action)

# ── Per-frame profiler (set PROFILE=True to print stage timings) ──────────────
PROFILE = False
_pt: List[float] = [0.0] * 5   # [proc, sidebar, compose, show, key]
_pn = 0; _pt0 = 0.0

def _prof_tick(slot: int):
    global _pt0
    if not PROFILE: return
    _pt[slot] += time.perf_counter() - _pt0
    _pt0 = time.perf_counter()

def _prof_start():
    global _pt0
    if PROFILE: _pt0 = time.perf_counter()

# Camera FPS (set by FrameGrabber background thread, read in sidebar)
cam_fps: float = 0.0

def current_record_fps() -> float:
    if cam_fps > 0:
        return cam_fps
    if sensor and sensor.fps > 0:
        return float(sensor.fps)
    return max(fps_current, 1.0)

# Sidebar frame cache — rebuild at most every _SB_EVERY frames
_SB_EVERY  = 3
_sb_tick   = 0
_sb_cache: Optional[np.ndarray] = None

# ── Pre-allocated display canvas (avoids 3 MB allocation per frame) ───────────
_canvas: Optional[np.ndarray] = None

def _ensure_canvas():
    global _canvas
    if _canvas is None or _canvas.shape[:2] != (WIN_H, WIN_W):
        _canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

# ── Colormap strip cache (rebuild only when colormap changes) ─────────────────
_cmap_strip_cache: dict = {}

def _cmap_strip(w: int, h: int) -> np.ndarray:
    """Return a (h,w,3) colorized gradient strip, cached per (cm_idx, w, h)."""
    key = (cm_idx, w, h)
    if key not in _cmap_strip_cache:
        _cmap_strip_cache.clear()
        bar = np.linspace(0, 255, w, dtype=np.uint8).reshape(1, -1)
        _cmap_strip_cache[key] = cv2.resize(colorize(bar, cm_idx), (w, h))
    return _cmap_strip_cache[key]

# ── Sidebar mouse interaction ─────────────────────────────────────────────────

def reg_hb(sb_x: int, y: int, w: int, h: int, action):
    """Register a hitbox in sidebar coords (will be offset by IW for abs)."""
    _hitboxes.append((IW + sb_x, y, w, h, action))

def hovering(sb_x: int, y: int, w: int, h: int) -> bool:
    ax = IW + sb_x
    return ax <= mouse_x <= ax+w and y <= mouse_y <= y+h

def to_cam(dx, dy):
    cx = int(dx/SCALE); cy = int(dy/SCALE)
    if flip_h: cx = CAM_W-1-cx
    if flip_v: cy = CAM_H-1-cy
    return max(0,min(cx,CAM_W-1)), max(0,min(cy,CAM_H-1))

def mouse_cb(event, x, y, flags, _):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

    if x >= IW:
        # Sidebar click
        if event == cv2.EVENT_LBUTTONDOWN:
            for ax, ay, aw, ah, action in _hitboxes:
                if ax <= x <= ax+aw and ay <= y <= ay+ah:
                    action()
                    return
    else:
        # Camera image click
        cx,cy = to_cam(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            markers.append({"x":cx,"y":cy,
                            "color":MARKER_COLORS[len(markers)%len(MARKER_COLORS)]})
        elif event == cv2.EVENT_RBUTTONDOWN and markers:
            markers.pop(min(range(len(markers)),
                            key=lambda i:abs(markers[i]["x"]-cx)+abs(markers[i]["y"]-cy)))

def _open_path(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(
                ['/usr/bin/open', path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == 'win32':
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            opener = shutil.which('xdg-open')
            if not opener:
                return False
            subprocess.Popen(
                [opener, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except Exception:
        return False

def _reveal_path(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(
                ['/usr/bin/open', '-R', path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == 'win32':
            subprocess.Popen(['explorer', f'/select,{os.path.normpath(path)}'])
        else:
            folder = os.path.dirname(path) or '.'
            opener = shutil.which('xdg-open')
            if not opener:
                return False
            subprocess.Popen(
                [opener, folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except Exception:
        return False

# ── Drawing helpers ───────────────────────────────────────────────────────────

def put(img, text, x, y, color=C_BRIGHT, scale=0.40, bold=False):
    # bold=True kept as parameter for API compat but always renders at weight 1
    # (HERSHEY at small sizes looks better thin; bold at ≤0.5 scale is unreadable)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)

def card(sb, x, y, w, h, fill=BG_CARD, border=BORDER):
    cv2.rectangle(sb,(x,y),(x+w,y+h), fill,-1)
    cv2.rectangle(sb,(x,y),(x+w,y+h), border, 1)

def hline(sb, y, x0=6, x1=None):
    cv2.line(sb,(x0,y),((x1 or SB_W-6),y), BORDER, 1)

def section(sb, x, y, title, color=C_ACCENT) -> int:
    cv2.rectangle(sb,(x,y-10),(x+3,y+4), color,-1)
    put(sb, title, x+7, y, C_MED, 0.34)
    return y + 16

def toggle_row(sb, x, y, label, shortcut, on: bool, action,
               on_color=None, off_label="OFF") -> int:
    """Draw a full-width toggleable row (26 px tall). Returns new y."""
    on_col = on_color or C_GREEN
    ROW_H  = 26
    hover  = hovering(x, y, SB_W - x - 4, ROW_H)
    row_bg = (36, 36, 36) if hover else (22, 22, 22)
    cv2.rectangle(sb, (x,    y), (SB_W-4, y+ROW_H), row_bg, -1)
    cv2.rectangle(sb, (x,    y), (SB_W-4, y+ROW_H),
                  BORDER_HI if hover else BORDER, 1)

    # Badge
    bc = on_col if on else (36, 36, 36)
    if hover: bc = tuple(min(255, c+22) for c in bc)
    tc_b = (255, 255, 255) if on else (62, 62, 62)
    bx1, by1, bx2, by2 = x+4, y+5, x+38, y+21
    cv2.rectangle(sb, (bx1, by1), (bx2, by2), bc, -1)
    cv2.rectangle(sb, (bx1, by1), (bx2, by2),
                  BORDER_HI if hover else BORDER, 1)
    put(sb, "ON" if on else off_label, bx1+4, by2-3, tc_b, 0.29)

    tx_y = y + ROW_H - 7
    put(sb, label,           x+44,     tx_y, C_BRIGHT if on else C_DIM, 0.35)
    put(sb, f"({shortcut})", SB_W-30,  tx_y, C_DIM, 0.28)

    reg_hb(x, y, SB_W - x - 4, ROW_H, action)
    return y + ROW_H + 2

def draw_histogram(sb, x, y, w, h_px):
    card(sb, x, y, w, h_px, fill=(15,15,15))
    if norm8 is None: return
    hist = cv2.calcHist([norm8],[0],None,[w],[0,256]).flatten()
    if hist.max() == 0: return
    bar_h = (hist / hist.max() * (h_px - 3)).astype(np.int32)   # shape (w,)

    # Build a full-size colored strip (cached) then mask above each bar — fully vectorised
    colored = _cmap_strip(w, h_px).copy()    # copy so we can mask it in-place
    row_idx = np.arange(h_px, dtype=np.int32).reshape(-1, 1)     # (h_px, 1)
    mask    = row_idx < (h_px - bar_h).reshape(1, -1)            # (h_px, w)
    colored[mask] = (15, 15, 15)
    sb[y:y+h_px, x:x+w] = colored

# ── Main sidebar ──────────────────────────────────────────────────────────────

def draw_sidebar(fps: float) -> np.ndarray:
    _hitboxes.clear()
    sb = np.full((IH, SB_W, 3), BG, dtype=np.uint8)
    X  = 8
    BW = SB_W - 16   # standard card / button width

    # ── Header ────────────────────────────────────────────────────────────────
    card(sb, 0, 0, SB_W, 48, fill=(18, 18, 18))
    sname = sensor.name if sensor else "Thermal Camera"
    put(sb, sname, X, 15, C_BRIGHT, 0.44)
    fps_col = C_ACCENT if fps >= 55 else C_BLUE if fps >= 25 else C_RED
    # Show cam delivery / display fps so bottleneck is instantly visible
    cfps = f"cam {cam_fps:.0f}/" if cam_fps > 0 else ""
    line2 = f"{cfps}{fps:.0f} fps  {CAM_W}x{CAM_H}"
    if frozen:             line2 += "  [FROZEN]";               fps_col = C_FROST
    if recorder.recording:
        audio_tag = "+MIC" if recorder.has_audio else ""
        line2 += f"  [REC{audio_tag}] {recorder.elapsed:.0f}s"; fps_col = C_REC
    put(sb, line2, X, 33, fps_col, 0.35)
    y = 52

    # ── Three large action buttons: FREEZE / REC·STOP / SAVE ──────────────────
    hline(sb, y); y += 6
    ABH  = 34                       # action button height
    _gap = 3
    bw3  = (BW - _gap * 2) // 3    # ≈82 px each

    def _abtn(bi, label, active, hi_col, fn, dot=None):
        """dot: None | 'circle' | 'square' — draws a coloured shape before the label."""
        bx  = X + bi * (bw3 + _gap)
        hov = hovering(bx, y, bw3, ABH)
        if active:
            bg = tuple(int(c * 0.45) for c in hi_col)
            tc, bc = hi_col, hi_col
        elif hov:
            bg, tc, bc = (44, 44, 44), C_BRIGHT, BORDER_HI
        else:
            bg, tc, bc = (22, 22, 22), C_DIM, BORDER
        cv2.rectangle(sb, (bx, y), (bx+bw3, y+ABH), bg, -1)
        cv2.rectangle(sb, (bx, y), (bx+bw3, y+ABH), bc,  1)
        dot_w = 12 if dot else 0
        tw    = len(label) * 7 + dot_w
        tx    = bx + max(4, (bw3 - tw) // 2)
        text_y = y + ABH//2 + 6
        if dot:
            dot_col = hi_col if (active or hov) else (65, 65, 65)
            dcx, dcy = tx + 4, text_y - 5
            if dot == 'circle':
                cv2.circle(sb, (dcx, dcy), 4, dot_col, -1, cv2.LINE_AA)
            else:   # square
                cv2.rectangle(sb, (dcx-4, dcy-4), (dcx+4, dcy+4), dot_col, -1)
            tx += 12
        put(sb, label, tx, text_y, tc, 0.35)
        reg_hb(bx, y, bw3, ABH, fn)

    _abtn(0, "FREEZE", frozen, C_FROST, _action_freeze)
    _abtn(1, "STOP" if recorder.recording else "REC",
          recorder.recording, C_REC, _action_toggle_record,
          dot='square' if recorder.recording else 'circle')
    _abtn(2, "SAVE", False, C_ACCENT, _action_save_snapshot)

    y += ABH + 8

    def _mini_btn(bx: int, by: int, bw: int, bh: int, label: str, action,
                  accent=C_ACCENT) -> None:
        hov = hovering(bx, by, bw, bh)
        bg  = tuple(min(255, c + 24) for c in accent) if hov else tuple(int(c * 0.22) for c in accent)
        bd  = accent if hov else BORDER
        cv2.rectangle(sb, (bx, by), (bx+bw, by+bh), bg, -1)
        cv2.rectangle(sb, (bx, by), (bx+bw, by+bh), bd, 1)
        tw = len(label) * 6
        put(sb, label, bx + max(5, (bw - tw) // 2), by + bh - 7, C_BRIGHT, 0.31)
        reg_hb(bx, by, bw, bh, action)

    if recorder.recording:
        awake_col = C_GREEN if recorder.keeping_screen_awake else C_ORANGE
        awake_msg = ("Display kept awake while recording"
                     if recorder.keeping_screen_awake
                     else "Recording active")
        put(sb, awake_msg, X+2, y, awake_col, 0.28)
        y += 11
        disk_warn = recorder.current_mux_space_warning()
        if disk_warn:
            put(sb, disk_warn, X+2, y, C_ORANGE, 0.27)
            y += 11

    # Mic-audio toggle — only drawn when a working microphone was detected at startup
    if _AUDIO_AVAILABLE:
        y = toggle_row(sb, X, y, "Record with mic", "M", record_audio,
                       _action_toggle_record_audio,
                       C_ORANGE, "OFF")
        label = VideoRecorder.audio_input_label()
        if label and len(label) > 32:
            label = label[:29] + "..."
        if label:
            put(sb, label, X+2, y, C_MED, 0.27); y += 11
        if recorder.recording:
            put(sb, "Stop recording to change", X+2, y, C_DIM, 0.27)
        else:
            if sys.platform == 'darwin' and _HAS_AVAUDIORECORDER:
                put(sb, "Use macOS Sound settings to change input", X+2, y, C_DIM, 0.27)
            else:
                put(sb, "A = change input  |  Mute = video only", X+2, y, C_DIM, 0.27)
        y += 11

    if _last_saved_path:
        last_exists = os.path.exists(_last_saved_path)
        hline(sb, y); y += 8
        y = section(sb, X, y, "LAST SAVE", C_BLUE if last_exists else C_ORANGE)
        card(sb, X, y, BW, 58, fill=(20, 24, 30) if last_exists else (28, 20, 20))
        name = os.path.basename(_last_saved_path)
        if len(name) > 34:
            name = name[:31] + "..."
        put(sb, name, X+6, y+14, C_BRIGHT if last_exists else C_ORANGE, 0.32)
        if last_exists:
            put(sb, "Click name or OPEN to launch file", X+6, y+27, C_DIM, 0.27)
            reg_hb(X, y, BW, 30, _action_open_last_saved)
            btn_y = y + 34
            btn_w = (BW - 6) // 2
            _mini_btn(X, btn_y, btn_w, 18, "OPEN", _action_open_last_saved, C_GREEN)
            _mini_btn(X + btn_w + 6, btn_y, btn_w, 18, "REVEAL", _action_reveal_last_saved, C_BLUE)
        else:
            put(sb, "File moved or deleted", X+6, y+27, C_DIM, 0.27)
        y += 64

    hline(sb, y); y += 8

    # ── Histogram ─────────────────────────────────────────────────────────────
    y = section(sb, X, y, "THERMAL DISTRIBUTION")
    HW, HH = BW, 52
    draw_histogram(sb, X, y, HW, HH)
    y += HH + 3
    put(sb, "Cold", X,       y, C_DIM, 0.29)
    put(sb, "Hot",  X+BW-22, y, C_DIM, 0.29)
    y += 13

    # ── Frame stats ───────────────────────────────────────────────────────────
    if frame_stats["delta"] > 0:
        mn, mx, me, dl = (frame_stats["min"], frame_stats["max"],
                          frame_stats["mean"], frame_stats["delta"])
        card(sb, X, y, BW, 28, fill=(19, 19, 19))
        put(sb, f"Min {mn:.0f}%  Max {mx:.0f}%  dT {dl:.0f}%",
            X+5, y+13, C_MED, 0.32)
        put(sb, f"Mean {me:.0f}%", X+5, y+24, C_DIM, 0.29)
        y += 32

    # ── Colormap ──────────────────────────────────────────────────────────────
    hline(sb, y); y += 8
    y = section(sb, X, y, "COLORMAP")
    hov_cm = hovering(X, y, BW, 18)
    sb[y:y+18, X:X+BW] = _cmap_strip(BW, 18)
    cv2.rectangle(sb, (X, y), (X+BW, y+18),
                  BORDER_HI if hov_cm else BORDER, 1)
    reg_hb(X, y, BW, 18, _action_cycle_cmap)
    y += 22
    put(sb, CMAPS[cm_idx][0], X,       y, C_BRIGHT, 0.40)
    put(sb, "click or C",     X+BW-64, y, C_DIM,    0.30)
    y += 14

    # ── View mode tabs ────────────────────────────────────────────────────────
    hline(sb, y); y += 8
    y = section(sb, X, y, "VIEW MODE")
    tab_w = BW // 3
    tab_h = 24
    for vi, vname in enumerate(VIEW_MODES):
        bx      = X + vi * tab_w
        active  = (vi == view_mode)
        hov_tab = hovering(bx, y, tab_w - 2, tab_h)
        bg  = (52, 100, 152) if active else ((38, 38, 38) if hov_tab else (26, 26, 26))
        tc  = (255,255,255) if active else ((190,190,190) if hov_tab else (70, 70, 70))
        cv2.rectangle(sb, (bx, y), (bx+tab_w-2, y+tab_h), bg, -1)
        cv2.rectangle(sb, (bx, y), (bx+tab_w-2, y+tab_h),
                      BORDER_HI if hov_tab else BORDER, 1)
        tw = len(vname) * 6
        tx = bx + max(3, (tab_w - tw) // 2)
        put(sb, vname, tx, y+tab_h-7, tc, 0.30)
        reg_hb(bx, y, tab_w-2, tab_h, lambda v=vi: _action_set_view(v))
    y += tab_h + 4
    desc, dc = {
        0: ("Flat-field + motion-NUC corrections applied",                    C_DIM),
        1: ("Per-pixel offset map  (red=stuck · cyan=hot/cold)"
            if flatf.correction is not None
            else "Not calibrated yet — press F to calibrate",
            C_ORANGE if flatf.correction is None else C_DIM),
        2: ("Raw sensor output — no corrections applied",                     C_DIM),
    }[view_mode]
    put(sb, desc, X, y, dc, 0.27)
    y += 13

    # ── Flat-field NUC ────────────────────────────────────────────────────────
    hline(sb, y); y += 8
    calib_col = C_ORANGE if flatf.calibrating else (C_GREEN if flatf.enabled else C_RED)
    y = section(sb, X, y, "FLAT-FIELD NUC  (F / Shift+F)", calib_col)

    if flatf.calibrating:
        mode_label = "FULL" if flatf._mode == 'full' else "COLOUR"
        put(sb, f"CAPTURING… [{mode_label}]", X, y, C_ORANGE, 0.36); y += 16
        pw = BW
        cv2.rectangle(sb, (X, y), (X+pw,              y+14), (26, 26, 26), -1)
        cv2.rectangle(sb, (X, y), (X+int(pw*flatf.progress), y+14), (0,160,55), -1)
        cv2.rectangle(sb, (X, y), (X+pw,              y+14), BORDER, 1)
        put(sb, f"{flatf.frames_captured}/{flatf.N}", X+pw//2-16, y+11, C_BRIGHT, 0.32)
        y += 20
        if flatf._mode == 'full':
            put(sb, "Slowly slide over wall / sky", X, y, C_DIM, 0.30); y += 13
        else:
            put(sb, "Hold still or slide — colour only", X, y, C_DIM, 0.30); y += 13
    elif flatf.enabled:
        # breakdown: stuck + hot/cold + runtime
        stuck_s   = f"{flatf.n_stuck} stuck"
        outlier_s = f"{flatf.n_outlier} hot/cold"
        parts = [stuck_s, outlier_s]
        if flatf.n_runtime > 0:
            parts.append(f"+{flatf.n_runtime} runtime")
        detail = "  ".join(parts)
        card(sb, X, y, BW, 46, fill=(18, 24, 18))
        cv2.circle(sb, (X+8, y+10), 4, C_GREEN, -1, cv2.LINE_AA)
        put(sb, "Calibrated",              X+18, y+14, C_GREEN, 0.36)
        put(sb, f"{flatf.n_bad} bad px corrected", X+6, y+27, C_DIM, 0.30)
        put(sb, detail,                    X+6,  y+39, C_DIM,   0.27)
        y += 50
        put(sb, "N = NUC Map  |  F = colour  Shift+F = full", X, y, C_DIM, 0.27)
        y += 13
        rt_col = C_ACCENT if flatf.n_runtime > 0 else C_DIM
        put(sb, "Runtime scan: active", X, y, rt_col, 0.27)
        y += 13
    else:
        card(sb, X, y, BW, 73, fill=(20, 18, 18))
        put(sb, "Not calibrated",                         X+6, y+13, C_DIM,    0.34)
        put(sb, "1. Point at blank wall or sky",           X+6, y+27, C_MED,    0.30)
        put(sb, "2. Slowly slide camera  (or hold still)", X+6, y+41, C_MED,    0.30)
        put(sb, "F = colour only  (~1 s)",                 X+6, y+55, C_ORANGE, 0.30)
        put(sb, "Shift+F = full detection  (~4 s)",        X+6, y+68, C_ACCENT, 0.29)
        y += 77

    # ── Corrections ───────────────────────────────────────────────────────────
    hline(sb, y); y += 8
    y = section(sb, X, y, "CORRECTIONS")
    y = toggle_row(sb, X, y, "Motion NUC",      "D", denoise,          _action_toggle_denoise)
    put(sb, "Removes per-pixel fixed-pattern sensor noise", X+2, y, C_DIM, 0.27); y += 11
    y = toggle_row(sb, X, y, "Temporal smooth",  "T", smoother.enabled, _action_toggle_temporal)
    put(sb, "Frame-blend to reduce shot noise (adds motion blur)", X+2, y, C_DIM, 0.27); y += 11
    y = toggle_row(sb, X, y, "Flip horizontal",  "H", flip_h,           _action_toggle_fliph,  C_BLUE, "OFF")
    y = toggle_row(sb, X, y, "Flip vertical",    "V", flip_v,           _action_toggle_flipv,  C_BLUE, "OFF")

    # ── Markers ───────────────────────────────────────────────────────────────
    hline(sb, y); y += 8
    y = section(sb, X, y, f"MARKERS  ({len(markers)} placed)")
    if not markers:
        put(sb, "Click image to place a marker",  X, y, C_DIM, 0.30); y += 14
        put(sb, "Right-click to remove nearest",  X, y, C_DIM, 0.30)
    else:
        for i, m in enumerate(markers):
            val   = f"{norm8[m['y'], m['x']] / 255 * 100:.0f}%" if norm8 is not None else "-"
            hov_m = hovering(X, y, BW, 22)
            if hov_m:
                cv2.rectangle(sb, (X, y), (X+BW, y+22), (30, 30, 30), -1)
            cv2.circle(sb, (X+9, y+11), 5, m["color"], -1, cv2.LINE_AA)
            put(sb, f"M{i+1}", X+20, y+15, C_BRIGHT, 0.38)
            put(sb, val,       X+46, y+15, m["color"],  0.38)
            if hov_m:
                put(sb, "click to remove", X+86, y+15, (80, 80, 80), 0.27)
            i_cap = i
            reg_hb(X, y, BW, 22,
                   lambda ic=i_cap: markers.pop(ic) if ic < len(markers) else None)
            y += 24
        if y < IH - 54:
            put(sb, "Right-click on image to remove nearest", X, y, C_DIM, 0.27)

    # ── Bottom shortcuts ──────────────────────────────────────────────────────
    bot = IH - 44
    hline(sb, bot)
    put(sb, "S = snapshot    Q / Esc = quit",         X, bot+15, C_DIM, 0.31)
    last_hint = "OPEN LAST after save" if _last_saved_path else "Recordings > ~/Desktop/BosonCaptures/"
    put(sb, last_hint,  X, bot+29, C_DIM, 0.27)

    return sb

# ── Action callbacks (used by keyboard + mouse hitboxes) ─────────────────────

def _action_cycle_cmap():
    global cm_idx, status_msg
    cm_idx = (cm_idx+1) % len(CMAPS)
    status_msg = f"Colormap: {CMAPS[cm_idx][0]}"

def _action_set_view(v: int):
    global view_mode, nuc_auto_t, status_msg
    nuc_auto_t = 0.0; view_mode = v
    status_msg = f"View: {VIEW_MODES[v]}"

def _action_toggle_denoise():
    global denoise, status_msg
    denoise = not denoise
    status_msg = f"Motion NUC {'ON' if denoise else 'OFF'}"

def _action_toggle_temporal():
    global status_msg
    smoother.enabled = not smoother.enabled; smoother.smooth = None
    status_msg = f"Temporal smooth {'ON' if smoother.enabled else 'OFF'}"

def _action_toggle_fliph():
    global flip_h, status_msg
    flip_h = not flip_h
    status_msg = f"Flip H {'ON' if flip_h else 'OFF'}"

def _action_toggle_flipv():
    global flip_v, status_msg
    flip_v = not flip_v
    status_msg = f"Flip V {'ON' if flip_v else 'OFF'}"

def _action_toggle_record():
    global status_msg, _last_saved_path
    if recorder.recording:
        path, n, dur = recorder.stop()
        _last_saved_path = path if path and os.path.exists(path) else ""
        if recorder.last_result_note:
            status_msg = f"{recorder.last_result_note}  {os.path.basename(path)}"
            if _last_saved_path:
                status_msg += "  — click OPEN LAST"
        elif _last_saved_path:
            status_msg = (f"Saved {n} frames ({dur:.1f}s)  "
                          f"{os.path.basename(path)}  — click OPEN LAST")
        else:
            status_msg = f"Saved {n} frames ({dur:.1f}s)  {os.path.basename(path)}"
    else:
        recorder.start(IW, IH, current_record_fps(), with_audio=record_audio)
        audio_note = " + audio" if record_audio and _AUDIO_AVAILABLE else ""
        status_msg = f"Recording{audio_note}: {os.path.basename(recorder.path)}"

def _action_toggle_record_audio():
    global record_audio, status_msg
    if recorder.recording: return   # don't switch mid-recording
    record_audio = not record_audio
    status_msg = "Recording audio: ON" if record_audio else "Recording audio: OFF"

def _action_cycle_audio_input():
    global status_msg
    if recorder.recording:
        return
    label = VideoRecorder.cycle_audio_input()
    if label:
        status_msg = f"Mic input: {label}"
    elif sys.platform == 'darwin' and _HAS_AVAUDIORECORDER:
        status_msg = "Change the default mic in macOS Sound settings"

def _action_freeze():
    global frozen, status_msg
    frozen = not frozen
    status_msg = "Frame frozen - SPACE to resume" if frozen else "Resumed"

def _action_start_calibrate(mode: str = 'offset'):
    global status_msg, _rtbp_suspicion, _rtbp_recovery, _rtbp_frame_ctr
    if not flatf.calibrating:
        _rtbp_suspicion = None
        _rtbp_recovery = None
        _rtbp_frame_ctr = 0
        flatf.start(mode); nuc.reset()
        _action_set_view(0)
        if mode == 'full':
            status_msg = "Full calibration — slide slowly over wall / sky"
        else:
            status_msg = "Colour calibration — hold still or slide over surface"

def _action_save_snapshot():
    global status_msg, _last_saved_path
    d  = os.path.expanduser("~/Desktop/BosonCaptures")
    os.makedirs(d, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(d, f"thermal_{ts}.png")
    if _canvas is not None:
        cv2.imwrite(fn, _canvas)
        _last_saved_path = fn if os.path.exists(fn) else ""
        status_msg = f"Saved: {os.path.basename(fn)}  — click OPEN LAST"

def _action_open_last_saved():
    global status_msg
    if not _last_saved_path or not os.path.exists(_last_saved_path):
        status_msg = "Last saved file is missing"
        return
    if _open_path(_last_saved_path):
        status_msg = f"Opened: {os.path.basename(_last_saved_path)}"
    else:
        status_msg = f"Could not open: {os.path.basename(_last_saved_path)}"

def _action_reveal_last_saved():
    global status_msg
    if not _last_saved_path or not os.path.exists(_last_saved_path):
        status_msg = "Last saved file is missing"
        return
    if _reveal_path(_last_saved_path):
        status_msg = f"Revealed: {os.path.basename(_last_saved_path)}"
    else:
        status_msg = f"Could not reveal: {os.path.basename(_last_saved_path)}"

# ── Camera image builder ──────────────────────────────────────────────────────

def build_cam_image(f32_corr: np.ndarray, f32_raw: np.ndarray) -> np.ndarray:
    if view_mode == 1:
        if flatf.correction is not None:
            src = flatf.nuc_map_image(cm_idx)
        else:
            src = np.zeros((CAM_H,CAM_W,3), np.uint8)
    elif view_mode == 2:
        n   = cv2.normalize(f32_raw,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        src = colorize(n, cm_idx)
    else:
        n   = cv2.normalize(f32_corr,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        src = colorize(n, cm_idx)
    return cv2.resize(src,(IW,IH),interpolation=cv2.INTER_NEAREST)

def decorate_camera_image(base_img: np.ndarray, copy_frame: bool = False) -> np.ndarray:
    need_markers = bool(markers) and view_mode != 1 and norm8 is not None
    img = base_img.copy() if copy_frame or need_markers else base_img

    if need_markers:
        for i, m in enumerate(markers):
            cx = int((CAM_W-1-m["x"] if flip_h else m["x"])*SCALE)
            cy = int((CAM_H-1-m["y"] if flip_v else m["y"])*SCALE)
            c  = m["color"]
            cv2.line(img,(cx-18,cy),(cx+18,cy),c,2,cv2.LINE_AA)
            cv2.line(img,(cx,cy-18),(cx,cy+18),c,2,cv2.LINE_AA)
            cv2.circle(img,(cx,cy),4,c,-1,cv2.LINE_AA)
            cv2.putText(img,f"M{i+1}",(cx+9,cy-9),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(img,f"M{i+1}",(cx+9,cy-9),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,c,1,cv2.LINE_AA)

    if frozen:               border_c, bw = C_FROST, 4
    elif recorder.recording: border_c, bw = C_REC,   3
    elif flatf.calibrating:  border_c, bw = C_ORANGE, 2
    elif view_mode == 1:     border_c, bw = (80,138,198), 2
    else:                    border_c, bw = (20,20,20), 1
    cv2.rectangle(img,(0,0),(IW-1,IH-1), border_c, bw)
    return img

# ── Camera finder ─────────────────────────────────────────────────────────────

def find_camera() -> Tuple[Optional[cv2.VideoCapture], Optional[SensorProfile]]:
    """Open the first thermal camera found and match to a sensor profile.

    Selects the best OpenCV backend per platform:
      macOS   — CAP_AVFOUNDATION
      Windows — CAP_DSHOW
      Linux   — CAP_V4L2 (also tries default backend as fallback)
    """
    try_force_60fps()

    if sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    elif sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:  # Linux and others
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        for idx in range(12):
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if not cap.isOpened():
                cap.release(); continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release(); continue

            h, w = frame.shape[:2]
            s    = match_sensor(w, h)

            # Skip obvious built-in webcams (high-res, common laptop sizes).
            # Thermal sensors are always <= 640 × 512.
            if w > 1280 or h > 1024:
                cap.release(); continue

            # Apply optimal format / fps settings (best-effort — may be ignored)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'I420'))
            cap.set(cv2.CAP_PROP_FPS, s.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  Camera {idx}: {s.name}  {w}x{h}  {actual_fps:.0f} fps  "
                  f"(backend {'AVFOUNDATION' if backend==cv2.CAP_AVFOUNDATION else 'DSHOW' if backend==cv2.CAP_DSHOW else 'V4L2' if backend==cv2.CAP_V4L2 else 'default'})")
            return cap, s

    return None, None

# ── Main ──────────────────────────────────────────────────────────────────────
fps_current = 30.0   # updated every second for UI / profiler
_ESC_KEY_CODES = {27, 0x10001B, 0xFF1B}

def _read_ui_key(delay_ms: int = 1) -> Tuple[int, int]:
    """Return (raw_key, normalized_key) from OpenCV HighGUI.

    macOS / Qt backends sometimes return backend-specific escape codes such as
    0x10001B or 0xFF1B instead of plain 27.  Normalize those to 27 while
    keeping the low ASCII byte for regular keys.
    """
    raw = cv2.waitKeyEx(delay_ms) if hasattr(cv2, "waitKeyEx") else cv2.waitKey(delay_ms)
    if raw < 0:
        return raw, -1
    key = raw & 0xFF
    if raw in _ESC_KEY_CODES or key == 27:
        return raw, 27
    return raw, key

def main():
    global norm8, raw_f, frame_stats, status_msg, nuc_auto_t
    global view_mode, fps_current
    global cam_fps, _sb_tick, _sb_cache, _pn, _pt
    global _tc_nuc_offset, _tc_prev_frame, _tc_out_buf

    print("Thermal Viewer — searching for camera…")
    cap, s = find_camera()
    if cap is None:
        hint = {
            "darwin":  "macOS: System Settings → Privacy & Security → Camera → grant Terminal",
            "win32":   "Windows: make sure the camera is not in use by another app",
        }.get(sys.platform, "Linux: check 'ls /dev/video*' and camera permissions")
        print(f"ERROR: no thermal camera found.\n{hint}"); return

    setup_layout(s)
    flatf.load_if_needed()

    grabber = FrameGrabber(cap)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    fps_t = time.time(); fps_n = 0
    last_img = np.zeros((IH,IW,3), np.uint8)

    print(f"  Window: {WIN_W}x{WIN_H}  (PROFILE={'ON' if PROFILE else 'OFF — set PROFILE=True for stage timing'})")
    print("  SPACE=freeze  R=record  C=colormap  D=NUC  F=colour-cal  Shift+F=full-cal")
    print("  T=smooth  H/V=flip  N=view  S=save  A=audio input  Q=quit")
    print("  Sidebar buttons are clickable with mouse")
    if PROFILE:
        print("  Profiler: proc / sidebar / compose / show / key  [ms avg]")

    while True:
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1: break

        now = time.time()
        cam_fps = grabber.cam_fps          # camera delivery rate from background thread
        if frozen:
            pending_frames = grabber.read_all()   # discard while frozen; resume stays current
        elif recorder.recording:
            pending_frames = grabber.read_all()   # preserve sensor cadence for recording
        else:
            ret, frame, frame_t = grabber.read_latest()
            pending_frames = [(frame_t, frame)] if ret and frame is not None else []

        # ── Frame processing ───────────────────────────────────────────────
        _prof_start()
        if not frozen:
            n_pending = len(pending_frames)
            for i, (frame_t, frame) in enumerate(pending_frames):
                if frame is None:
                    continue
                is_last = (i == n_pending - 1)
                gray = frame[:, :, 0] if frame.ndim == 3 else frame

                if flatf.calibrating:
                    # f32 required only for calibration feed — don't compute it otherwise
                    f32 = gray.astype(np.float32)
                    done = flatf.feed(f32)
                    if done:
                        nuc_auto_t = now + 1.0
                        _action_set_view(1)
                        status_msg = (f"Flat-field done - {flatf.n_bad} bad px "
                                      f"- showing NUC Map")
                else:
                    # Runtime dead-pixel discovery (skipped during calibration so
                    # the uniform-surface frames don't pollute the suspicion map).
                    _rtbp_update(gray)

                # ── Fast path: Rust handles NUC + normalize + LUT ──────────────
                if _HAS_TC and denoise and view_mode == 0 and cm_idx == 0:
                    n = CAM_W * CAM_H
                    if _tc_nuc_offset is None or _tc_nuc_offset.shape[0] != n:
                        _tc_nuc_offset = np.zeros(n, np.float32)
                        _tc_prev_frame = np.zeros(n, np.float32)
                        nuc.O    = None
                        nuc.prev = None
                    # Allocate persistent output buffer lazily (once, reused every frame)
                    if _tc_out_buf is None or _tc_out_buf.shape[0] != IH * IW * 3:
                        _tc_out_buf = np.empty(IH * IW * 3, np.uint8)
                    # gray_flat: frame[:,:,0] has stride 3 → ravel() makes a contiguous copy
                    gray_flat = gray.ravel()   # 1-D uint8, already contiguous after ravel
                    # corr_flat / bad_flat are pre-built on FlatFieldNUC — never recomputed here
                    _tc.process_frame(
                        gray_flat, CAM_W, CAM_H,
                        _tc_nuc_offset, _tc_prev_frame,
                        nuc.mu, nuc.thresh,
                        IRON_LUT_FLAT,
                        _tc_out_buf,           # Rust writes directly into this buffer
                        IW, IH,
                        flatf.corr_flat, flatf.bad_flat,
                    )
                    # View into _tc_out_buf — no copy.  If we need to draw on it we
                    # copy lazily below (only when markers are present).
                    last_img = _tc_out_buf.reshape(IH, IW, 3)
                    if is_last:
                        # norm8 at display resolution: used for marker lookups
                        # and stats.  Its values track temperature via Iron.
                        norm8 = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)
                        mn_v, mx_v, _, _ = cv2.minMaxLoc(norm8[::4, ::4])
                        me_v = float(cv2.mean(norm8[::4, ::4])[0])
                        frame_stats = {"min": mn_v / 2.55, "max": mx_v / 2.55,
                                       "mean": me_v / 2.55,
                                       "delta": (mx_v - mn_v) / 2.55}

                # ── Fallback: pure Python (other colormaps / views / no Rust) ──
                else:
                    f32   = gray.astype(np.float32)
                    raw_f = f32.copy()
                    if denoise:
                        f32 = flatf.apply(f32)
                        f32 = nuc.update(f32)
                    else:
                        nuc.prev = f32.copy()
                    f32 = smoother.update(f32)
                    if is_last:
                        norm8 = cv2.normalize(f32, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        mn = float(norm8.min()) / 255 * 100
                        mx = float(norm8.max()) / 255 * 100
                        me = float(norm8.mean()) / 255 * 100
                        frame_stats = {"min": mn, "max": mx, "mean": me, "delta": mx - mn}
                    last_img = build_cam_image(f32, raw_f)

                if flip_h: last_img = cv2.flip(last_img, 1)
                if flip_v: last_img = cv2.flip(last_img, 0)
                if recorder.recording:
                    recorder.write(decorate_camera_image(last_img, copy_frame=True),
                                   copy_frame=False,
                                   frame_time=frame_t)
                fps_n += 1
        _prof_tick(0)    # slot 0: frame processing

        # Auto-revert NUC map view
        if view_mode==1 and nuc_auto_t>0 and now>nuc_auto_t:
            view_mode=0; nuc_auto_t=0.0
            status_msg = "Returned to Live view - N to see NUC Map again"

        # FPS counter
        if now - fps_t >= 1.0:
            fps_current = fps_n/(now-fps_t); fps_n=0; fps_t=now

        # ── Compose display ────────────────────────────────────────────────
        img = decorate_camera_image(last_img)

        # ── Sidebar — rebuilt at most every _SB_EVERY frames (~20fps) ────
        # Positions never change frame-to-frame so cached hitboxes stay valid.
        _sb_tick += 1
        if _sb_tick % _SB_EVERY == 0 or _sb_cache is None:
            _sb_cache = draw_sidebar(fps_current)
        _prof_tick(1)    # slot 1: sidebar

        # ── Canvas assembly ────────────────────────────────────────────────
        _ensure_canvas()
        _canvas[:IH, :IW]  = img
        _canvas[:IH, IW:]  = _sb_cache
        _canvas[IH:, :]    = (9, 9, 9)
        cv2.line(_canvas, (0,IH), (WIN_W,IH), (38,38,38), 1)
        vm_col = {0:C_DIM, 1:C_ORANGE, 2:C_DIM}[view_mode]
        put(_canvas, f"[{VIEW_MODES[view_mode].upper()}]", 8, IH+21, vm_col, 0.36)
        if frozen:
            put(_canvas, "FROZEN - SPACE to resume", 80, IH+21, C_FROST, 0.38)
        elif flatf.calibrating:
            put(_canvas, f"Calibrating  {int(flatf.progress*100)}%  "
                        f"({flatf.frames_captured}/{flatf.N}) — slowly slide over surface",
               80, IH+21, C_ORANGE, 0.38)
        elif recorder.recording:
            cv2.rectangle(_canvas, (82, IH+12), (90, IH+20), C_REC, -1)
            audio_tag2 = " + MIC" if recorder.has_audio else ""
            put(_canvas, f"RECORDING{audio_tag2}  {recorder.elapsed:.0f}s", 96, IH+21, C_REC, 0.38)
        else:
            put(_canvas, status_msg, 80, IH+21, (148,148,148), 0.36)
        _prof_tick(2)    # slot 2: canvas compose + status bar

        # ── Display ────────────────────────────────────────────────────────
        cv2.imshow(WIN_NAME, _canvas)
        _prof_tick(3)    # slot 3: imshow

        # waitKeyEx(1) every frame: macOS Core Animation only commits imshow()
        # frames to the screen when the Cocoa event loop is drained.  Using
        # waitKeyEx keeps backend-specific escape codes intact so we can
        # normalize them reliably instead of hoping they survive masking.
        raw_key, key = _read_ui_key(1)
        _prof_tick(4)    # slot 4: key poll

        # ── Profiler output every 2 s ─────────────────────────────────────
        if PROFILE:
            _pn += 1
            if _pn >= int(fps_current * 2) and fps_current > 0:
                n = max(1, _pn)
                print(f"FPS disp={fps_current:.1f} cam={cam_fps:.1f}  "
                      f"proc={_pt[0]/n*1e3:.1f}ms  side={_pt[1]/n*1e3:.1f}ms  "
                      f"compose={_pt[2]/n*1e3:.1f}ms  show={_pt[3]/n*1e3:.1f}ms  "
                      f"key={_pt[4]/n*1e3:.1f}ms  "
                      f"TOTAL={sum(_pt)/n*1e3:.1f}ms")
                _pt = [0.0]*5; _pn = 0

        # ── Keyboard ───────────────────────────────────────────────────────
        if   key in (ord('q'), ord('Q'), 27): break
        elif key == ord(' '):           _action_freeze()
        elif key == ord('r'):           _action_toggle_record()
        elif key == ord('m') and _AUDIO_AVAILABLE: _action_toggle_record_audio()
        elif key in (ord('a'), ord('A')) and _AUDIO_AVAILABLE: _action_cycle_audio_input()
        elif key == ord('c'):           _action_cycle_cmap()
        elif key == ord('n'):           _action_set_view((view_mode+1)%len(VIEW_MODES))
        elif key == ord('h'):           _action_toggle_fliph()
        elif key == ord('v'):           _action_toggle_flipv()
        elif key == ord('d'):           _action_toggle_denoise()
        elif key == ord('t'):           _action_toggle_temporal()
        elif key == ord('f'):           _action_start_calibrate('offset')
        elif key == ord('F'):           _action_start_calibrate('full')
        elif key == ord('s'):           _action_save_snapshot()

    if recorder.recording: recorder.stop()
    grabber.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
