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
  D                  Motion NUC toggle   F             Flat-field calibrate
  T                  Temporal smooth     H / V         Flip H / V
  S                  Save snapshot       Q / Esc       Quit
"""

import cv2, numpy as np, time, os, threading
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
    SensorProfile("Generic UVC 640×480",   640, 480, 30, ""),
    SensorProfile("Generic UVC 320×240",   320, 240, 30, ""),
]

def match_sensor(w: int, h: int) -> SensorProfile:
    for s in SENSOR_PROFILES:
        if s.w == w and s.h == h:
            return s
    return SensorProfile(f"Unknown {w}×{h}", w, h, 30, "")

# ── Layout (computed after sensor detection) ──────────────────────────────────
TARGET_IW = 960     # target display width for camera image
SB_W      = 268     # sidebar width
BAR_H     = 32      # status bar height

# These are initialised in setup_layout():
sensor    : Optional[SensorProfile] = None
CAM_W = CAM_H = 640, 512   # overwritten
SCALE = 1.5
IW    = 960
IH    = 768
WIN_W = IW + SB_W
WIN_H = IH + BAR_H

def setup_layout(s: SensorProfile):
    global sensor, CAM_W, CAM_H, SCALE, IW, IH, WIN_W, WIN_H
    sensor = s
    CAM_W, CAM_H = s.w, s.h
    SCALE  = TARGET_IW / s.w
    IW     = int(s.w * SCALE)
    IH     = int(s.h * SCALE)
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
IRON_LUT = _build_iron_lut()

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
                desired = cv2.GaussianBlur(corrected, self._ksize, self.sigma,
                                           borderType=cv2.BORDER_REFLECT)
                self.O  += self.mu * (corrected - desired)
                self.updates += 1
        self.prev = f32.copy()
        return corrected

    def reset(self):
        self.O = None; self.prev = None; self.updates = 0


class FlatFieldNUC:
    """One-point flat-field calibration. Point at uniform surface, press F.

    Dead-pixel detection uses two independent methods so nothing is missed:

    Method A — Stuck / frozen pixels
        Temporal std is far below the scene average.  A truly stuck pixel
        never responds to scene changes, so its std ≈ 0.  Threshold: < 20 %
        of the median std across all pixels.

    Method B — Persistent spatial outliers (hot / cold pixels)
        Even on a uniform scene some pixels are always much brighter or darker
        than their immediate neighbours.  We compare each pixel's mean to a
        9×9 Gaussian-blurred version of the mean; pixels in the top 0.8 % of
        that local-deviation map (or > 3 σ above its mean) are flagged.

    Both masks are OR'd together so any pixel caught by either method is
    corrected with the 3×3 neighbourhood average.
    """
    N = 64

    def __init__(self):
        self.correction  : Optional[np.ndarray] = None
        self.bad_mask    : Optional[np.ndarray] = None   # all bad pixels combined
        self.stuck_mask  : Optional[np.ndarray] = None   # method A only
        self.outlier_mask: Optional[np.ndarray] = None   # method B only
        self.n_bad = self.n_stuck = self.n_outlier = 0
        self.enabled     = False
        self.calibrating = False
        self._buf: List[np.ndarray] = []
        self._done_t: float = 0
        self._load()

    @property
    def savepath(self) -> str:
        d = os.path.expanduser("~/.thermal_viewer")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"flatfield_{CAM_W}x{CAM_H}.npz")

    def start(self):
        self._buf=[]; self.calibrating=True; self._done_t=0

    def feed(self, f32: np.ndarray) -> bool:
        if not self.calibrating: return False
        self._buf.append(f32.copy())
        if len(self._buf) >= self.N: self._finish(); return True
        return False

    def _finish(self):
        stack = np.stack(self._buf, 0)
        mean  = stack.mean(0).astype(np.float32)
        std   = stack.std(0).astype(np.float32)

        # Per-pixel offset correction (subtracts the fixed-pattern offset)
        self.correction = (mean.mean() - mean)

        # ── Method A: stuck / frozen pixels ─────────────────────────────────
        med_std = float(np.median(std))
        self.stuck_mask = std < (med_std * 0.20)   # < 20 % of median std

        # ── Method B: persistent spatial outliers (hot / cold pixels) ────────
        # Compare each pixel's mean to its local neighbourhood average
        local_avg = cv2.GaussianBlur(mean, (9, 9), 2.0)
        local_dev = np.abs(mean - local_avg)
        dev_p     = float(np.percentile(local_dev, 99.2))   # top 0.8 %
        dev_sigma = float(local_dev.mean() + 3.0 * local_dev.std())
        self.outlier_mask = local_dev > max(dev_p, dev_sigma)

        # ── Combine ──────────────────────────────────────────────────────────
        self.bad_mask  = self.stuck_mask | self.outlier_mask
        self.n_stuck   = int(self.stuck_mask.sum())
        self.n_outlier = int((self.outlier_mask & ~self.stuck_mask).sum())  # non-overlap
        self.n_bad     = int(self.bad_mask.sum())

        self.calibrating = False; self.enabled = True; self._done_t = time.time()
        self._buf = []
        np.savez(self.savepath,
                 c=self.correction,
                 b=self.bad_mask.astype(np.uint8),
                 bs=self.stuck_mask.astype(np.uint8),
                 bo=self.outlier_mask.astype(np.uint8))
        print(f"  Flat-field done — {self.n_bad} bad px "
              f"({self.n_stuck} stuck, {self.n_outlier} hot/cold)")

    def apply(self, f32: np.ndarray) -> np.ndarray:
        if not self.enabled or self.correction is None: return f32
        out = f32 + self.correction
        if self.n_bad > 0:
            # Replace every bad pixel with the mean of its 3×3 neighbours.
            # cv2.blur() (box-filter mean) runs over the whole image in one
            # O(n) integral-image pass; we then copy only at bad locations.
            neighbour_mean = cv2.blur(out, (3, 3))
            out[self.bad_mask] = neighbour_mean[self.bad_mask]
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

        n   = cv2.normalize(self.correction, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
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
            self.bad_mask     = d["b"].astype(bool)
            self.stuck_mask   = d["bs"].astype(bool) if "bs" in d else self.bad_mask
            self.outlier_mask = d["bo"].astype(bool) if "bo" in d else np.zeros_like(self.bad_mask)
            self.n_stuck      = int(self.stuck_mask.sum())
            self.n_outlier    = int((self.outlier_mask & ~self.stuck_mask).sum())
            self.n_bad        = int(self.bad_mask.sum())
            self.enabled      = True
            print(f"  Loaded flat-field ({CAM_W}×{CAM_H}) — "
                  f"{self.n_bad} bad px ({self.n_stuck} stuck, {self.n_outlier} hot/cold)")
        except Exception as e:
            print(f"  Could not load flat-field: {e}")

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
    """Records the colorized camera image to an .mp4 file."""
    def __init__(self):
        self._writer: Optional[cv2.VideoWriter] = None
        self.path = ""
        self._t0  = 0.0
        self._n   = 0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def start(self, w: int, h: int, fps: float):
        d = os.path.expanduser("~/Desktop/BosonCaptures")
        os.makedirs(d, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(d, f"thermal_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self.path, fourcc, max(fps,10.0), (w,h))
        self._t0 = time.time(); self._n = 0
        print(f"  Recording → {self.path}")

    def write(self, frame: np.ndarray):
        if self._writer: self._writer.write(frame); self._n += 1

    def stop(self) -> Tuple[str,int,float]:
        if self._writer: self._writer.release(); self._writer = None
        dur = time.time() - self._t0
        print(f"  Stopped — {self._n} frames, {dur:.1f}s → {self.path}")
        return self.path, self._n, dur

    @property
    def elapsed(self) -> float:
        return time.time()-self._t0 if self.recording else 0.0

# ── Threaded grabber ──────────────────────────────────────────────────────────

class FrameGrabber:
    """Reads camera on a background thread — UI loop never blocks on cap.read().
    Tracks a 'new' flag so the main loop only processes each camera frame once,
    giving accurate FPS measurement and no wasted NUC computation on duplicates."""
    def __init__(self, cap):
        self._cap   = cap
        self._frame : Optional[np.ndarray] = None
        self._new   = False
        self._lock  = threading.Lock()
        self._alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._alive:
            ret, f = self._cap.read()
            if ret and f is not None:
                with self._lock:
                    self._frame = f
                    self._new   = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns (True, frame) only when a NEW frame has arrived since last call."""
        with self._lock:
            if not self._new or self._frame is None: return False, None
            self._new = False
            return True, self._frame   # caller must not modify in-place

    def release(self):
        self._alive = False; self._cap.release()

# ── PyObjC 60fps pre-config ───────────────────────────────────────────────────

def try_force_60fps():
    try:
        import AVFoundation as avf, CoreMedia as cm
        for dev in avf.AVCaptureDevice.devicesWithMediaType_(avf.AVMediaTypeVideo):
            name = str(dev.localizedName())
            if "FLIR" not in name and "Boson" not in name: continue
            best = None
            for fmt in dev.formats():
                d = cm.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
                if d.width == CAM_W and d.height == CAM_H:
                    for r in fmt.videoSupportedFrameRateRanges():
                        if r.maxFrameRate() >= 60: best=fmt; break
                if best: break
            if not best: return
            if dev.lockForConfiguration_(None): return
            dev.setActiveFormat_(best)
            t = cm.CMTimeMake(1,60)
            dev.setActiveVideoMinFrameDuration_(t)
            dev.setActiveVideoMaxFrameDuration_(t)
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

status_msg   = "Ready"
nuc_auto_t   = 0.0
mouse_x = mouse_y = 0

# Sidebar click hitboxes — cleared and rebuilt each frame
_hitboxes: List[Tuple[int,int,int,int,object]] = []  # (abs_x,y,w,h,action)

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
    global mouse_x, mouse_y, status_msg
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
    global status_msg
    _hitboxes.clear()
    sb = np.full((IH, SB_W, 3), BG, dtype=np.uint8)
    X  = 8
    BW = SB_W - 16   # standard card / button width

    # ── Header ────────────────────────────────────────────────────────────────
    card(sb, 0, 0, SB_W, 48, fill=(18, 18, 18))
    sname = sensor.name if sensor else "Thermal Camera"
    put(sb, sname, X, 15, C_BRIGHT, 0.44)
    fps_col = C_ACCENT if fps >= 55 else C_BLUE if fps >= 25 else C_RED
    line2 = f"{fps:.0f} fps  ·  {CAM_W}×{CAM_H}"
    if frozen:             line2 += "  ● FROZEN";         fps_col = C_FROST
    if recorder.recording: line2 += f"  ● REC {recorder.elapsed:.0f}s"; fps_col = C_REC
    put(sb, line2, X, 33, fps_col, 0.35)
    y = 52

    # ── Three large action buttons: FREEZE / REC·STOP / SAVE ──────────────────
    hline(sb, y); y += 6
    ABH  = 34                       # action button height
    _gap = 3
    bw3  = (BW - _gap * 2) // 3    # ≈82 px each

    def _abtn(bi, label, active, hi_col, fn):
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
        tw = len(label) * 7
        tx = bx + max(4, (bw3 - tw) // 2)
        put(sb, label, tx, y + ABH//2 + 6, tc, 0.35)
        reg_hb(bx, y, bw3, ABH, fn)

    _abtn(0, "FREEZE", frozen, C_FROST, _action_freeze)
    _abtn(1, "■ STOP" if recorder.recording else "● REC",
          recorder.recording, C_REC, _action_toggle_record)
    _abtn(2, "SAVE", False, C_ACCENT, _action_save_snapshot)

    y += ABH + 8
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
        put(sb, f"Min {mn:.0f}%   Max {mx:.0f}%   Δ {dl:.0f}%",
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
    desc = {0: "Corrected live feed",
            1: "Flat-field noise map" if flatf.correction is not None else "No calibration yet",
            2: "Raw uncorrected feed"}[view_mode]
    dc = C_ORANGE if (view_mode == 1 and flatf.correction is None) else C_DIM
    put(sb, desc, X, y, dc, 0.29)
    y += 14

    # ── Flat-field NUC ────────────────────────────────────────────────────────
    hline(sb, y); y += 8
    calib_col = C_ORANGE if flatf.calibrating else (C_GREEN if flatf.enabled else C_RED)
    y = section(sb, X, y, "FLAT-FIELD NUC  (F)", calib_col)

    if flatf.calibrating:
        put(sb, "CAPTURING — hold still", X, y, C_ORANGE, 0.36); y += 16
        pw = BW
        cv2.rectangle(sb, (X, y), (X+pw,              y+14), (26, 26, 26), -1)
        cv2.rectangle(sb, (X, y), (X+int(pw*flatf.progress), y+14), (0,160,55), -1)
        cv2.rectangle(sb, (X, y), (X+pw,              y+14), BORDER, 1)
        put(sb, f"{flatf.frames_captured}/{flatf.N}", X+pw//2-16, y+11, C_BRIGHT, 0.32)
        y += 20
        put(sb, "Point at wall / sky / lens cap", X, y, C_DIM, 0.30)
        y += 14
    elif flatf.enabled:
        card(sb, X, y, BW, 38, fill=(18, 24, 18))
        put(sb, "● Calibrated",                      X+6, y+14, C_GREEN, 0.36)
        put(sb, f"{flatf.n_bad} bad pixels corrected", X+6, y+27, C_DIM,   0.29)
        y += 42
        put(sb, "N → NUC Map  |  F = recalibrate", X, y, C_DIM, 0.28)
        y += 14
    else:
        card(sb, X, y, BW, 52, fill=(20, 18, 18))
        put(sb, "Not calibrated",                X+6, y+13, C_DIM,    0.34)
        put(sb, "1. Point at blank wall or sky",  X+6, y+27, C_MED,    0.30)
        put(sb, "2. Press F — hold 2 seconds",    X+6, y+40, C_ORANGE, 0.32)
        y += 56

    # ── Corrections ───────────────────────────────────────────────────────────
    hline(sb, y); y += 8
    y = section(sb, X, y, "CORRECTIONS")
    y = toggle_row(sb, X, y, "Motion NUC",      "D", denoise,          _action_toggle_denoise)
    y = toggle_row(sb, X, y, "Temporal smooth",  "T", smoother.enabled, _action_toggle_temporal)
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
            val   = f"{norm8[m['y'], m['x']] / 255 * 100:.0f}%" if norm8 is not None else "—"
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
    put(sb, "Recordings → ~/Desktop/BosonCaptures/",  X, bot+29, C_DIM, 0.27)

    return sb

# ── Action callbacks (used by keyboard + mouse hitboxes) ─────────────────────

def _action_cycle_cmap():
    global cm_idx, status_msg
    cm_idx = (cm_idx+1) % len(CMAPS)
    status_msg = f"Colormap → {CMAPS[cm_idx][0]}"

def _action_set_view(v: int):
    global view_mode, nuc_auto_t, status_msg
    nuc_auto_t = 0.0; view_mode = v
    status_msg = f"View → {VIEW_MODES[v]}"

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
    global status_msg
    if recorder.recording:
        path, n, dur = recorder.stop()
        status_msg = f"Saved {n} frames ({dur:.1f}s) → {os.path.basename(path)}"
    else:
        recorder.start(IW, IH, fps_current)
        status_msg = f"Recording started → {os.path.basename(recorder.path)}"

def _action_freeze():
    global frozen, status_msg
    frozen = not frozen
    status_msg = "Frame frozen — SPACE to resume" if frozen else "Resumed"

def _action_start_calibrate():
    global status_msg
    if not flatf.calibrating:
        flatf.start(); nuc.reset()
        _action_set_view(0)
        status_msg = "Flat-field calibration started — aim at uniform surface"

def _action_save_snapshot():
    global status_msg
    d  = os.path.expanduser("~/Desktop/BosonCaptures")
    os.makedirs(d, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(d, f"thermal_{ts}.png")
    if _canvas is not None:
        cv2.imwrite(fn, _canvas)
        status_msg = f"Saved → {os.path.basename(fn)}"

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

# ── Camera finder ─────────────────────────────────────────────────────────────

def find_camera() -> Tuple[Optional[cv2.VideoCapture], Optional[SensorProfile]]:
    """Open the first thermal camera found and match to a sensor profile."""
    try_force_60fps()

    for idx in range(8):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened(): cap.release(); continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if not ret or frame is None: cap.release(); continue

        h, w = frame.shape[:2]
        s    = match_sensor(w, h)

        # Apply optimal format settings
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'I420'))
        cap.set(cv2.CAP_PROP_FPS, s.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Camera {idx}: {s.name}  {w}×{h}  {actual_fps:.0f} fps")
        return cap, s

    return None, None

# ── Main ──────────────────────────────────────────────────────────────────────
fps_current = 30.0   # updated every second, used by recorder

def main():
    global norm8, raw_f, frame_stats, status_msg, nuc_auto_t
    global frozen, view_mode, flip_h, flip_v, cm_idx, fps_current

    print("Thermal Viewer — searching for camera…")
    cap, s = find_camera()
    if cap is None:
        print("ERROR: no thermal camera found.\n"
              "Check System Settings → Privacy → Camera → Terminal"); return

    setup_layout(s)
    flatf.load_if_needed()

    grabber = FrameGrabber(cap)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    fps_t = time.time(); fps_n = 0
    last_img = np.zeros((IH,IW,3), np.uint8)

    print(f"  Window: {WIN_W}×{WIN_H}")
    print("  SPACE=freeze  R=record  C=colormap  D=NUC  F=calibrate")
    print("  T=smooth  H/V=flip  N=view  S=save  Q=quit")
    print("  Sidebar buttons are clickable with mouse")

    while True:
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1: break

        now = time.time()
        ret, frame = grabber.read()

        if not frozen and ret and frame is not None:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
            f32  = gray.astype(np.float32)
            raw_f = f32.copy()

            if flatf.calibrating:
                done = flatf.feed(f32)
                if done:
                    nuc_auto_t = now + 1.0   # show NUC map for 1 second then return to Live
                    _action_set_view(1)
                    status_msg = (f"Flat-field done — {flatf.n_bad} bad px "
                                  f"— showing NUC Map")

            if denoise:
                f32 = flatf.apply(f32)
                f32 = nuc.update(f32)
            else:
                nuc.prev = f32.copy()

            f32   = smoother.update(f32)
            norm8 = cv2.normalize(f32,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

            mn = float(norm8.min())/255*100
            mx = float(norm8.max())/255*100
            me = float(norm8.mean())/255*100
            frame_stats = {"min":mn,"max":mx,"mean":me,"delta":mx-mn}

            last_img = build_cam_image(f32, raw_f)
            if flip_h: last_img = cv2.flip(last_img,1)
            if flip_v: last_img = cv2.flip(last_img,0)
            fps_n += 1

        # Auto-revert NUC map view
        if view_mode==1 and nuc_auto_t>0 and now>nuc_auto_t:
            view_mode=0; nuc_auto_t=0.0
            status_msg = "Returned to Live view — N to see NUC Map again"

        # FPS counter
        if now - fps_t >= 1.0:
            fps_current = fps_n/(now-fps_t); fps_n=0; fps_t=now

        # ── Compose display ────────────────────────────────────────────────
        # Only copy if we need to draw markers on top; otherwise use directly
        need_draw = bool(markers) and view_mode != 1 and norm8 is not None
        img = last_img.copy() if need_draw else last_img

        # Marker crosshairs
        if need_draw:
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

        # Camera image border signals state
        if frozen:
            border_c, bw = C_FROST, 4
        elif recorder.recording:
            border_c, bw = C_REC, 3
        elif flatf.calibrating:
            border_c, bw = C_ORANGE, 2
        elif view_mode == 1:
            border_c, bw = (80,138,198), 2
        else:
            border_c, bw = (20,20,20), 1
        cv2.rectangle(img,(0,0),(IW-1,IH-1), border_c, bw)

        # Record camera image (before adding sidebar)
        if recorder.recording:
            recorder.write(img)

        sidebar = draw_sidebar(fps_current)

        # Assemble into pre-allocated canvas — zero extra allocation per frame
        _ensure_canvas()
        _canvas[:IH, :IW]  = img
        _canvas[:IH, IW:]  = sidebar
        # Status bar (fill then draw text)
        _canvas[IH:, :]    = (9, 9, 9)
        cv2.line(_canvas, (0,IH), (WIN_W,IH), (38,38,38), 1)
        vm_col = {0:C_DIM, 1:C_ORANGE, 2:C_DIM}[view_mode]
        put(_canvas, f"[{VIEW_MODES[view_mode].upper()}]", 8, IH+21, vm_col, 0.36)
        if frozen:
            put(_canvas, "FROZEN — SPACE to resume", 80, IH+21, C_FROST, 0.38)
        elif flatf.calibrating:
            put(_canvas, f"Calibrating  {int(flatf.progress*100)}%  "
                        f"({flatf.frames_captured}/{flatf.N}) — hold still",
               80, IH+21, C_ORANGE, 0.38)
        elif recorder.recording:
            put(_canvas, f"● RECORDING  {recorder.elapsed:.0f}s", 80, IH+21, C_REC, 0.38)
        else:
            put(_canvas, status_msg, 80, IH+21, (148,148,148), 0.36)

        cv2.imshow(WIN_NAME, _canvas)

        # ── Keyboard ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if   key in (ord('q'), 27):     break
        elif key == ord(' '):           _action_freeze()
        elif key == ord('r'):           _action_toggle_record()
        elif key == ord('c'):           _action_cycle_cmap()
        elif key == ord('n'):           _action_set_view((view_mode+1)%len(VIEW_MODES))
        elif key == ord('h'):           _action_toggle_fliph()
        elif key == ord('v'):           _action_toggle_flipv()
        elif key == ord('d'):           _action_toggle_denoise()
        elif key == ord('t'):           _action_toggle_temporal()
        elif key == ord('f'):           _action_start_calibrate()
        elif key == ord('s'):
            d  = os.path.expanduser("~/Desktop/BosonCaptures")
            os.makedirs(d, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(d, f"thermal_{ts}.png")
            cv2.imwrite(fn, _canvas)
            status_msg = f"Saved → {os.path.basename(fn)}"

    if recorder.recording: recorder.stop()
    grabber.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
