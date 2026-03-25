#!/usr/bin/env python3
"""
fps_bare.py — stripped to the bone to isolate the FPS ceiling.

Stages tested (uncomment one DISPLAY_MODE at a time):
  0 = capture only, no display           (what does the camera deliver?)
  1 = capture + cv2.imshow raw BGR       (add display cost)
  2 = capture + gray + imshow            (add color conversion)
  3 = capture + gray + colormap + imshow (add LUT / resize)

Run:
  python3 fps_bare.py
Press Q or Esc to quit.
"""

import cv2, numpy as np, time, threading

DISPLAY_MODE = 3   # 0 = capture-only, 1 = raw, 2 = gray, 3 = colormap+resize

# ── Find camera (same logic as main viewer) ───────────────────────────────────
def try_force_60fps():
    try:
        import AVFoundation as avf, CoreMedia as cm
        boson_sizes = {(640, 512), (320, 256)}
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
            print(f"  PyObjC: 60fps locked on '{name}'")
    except Exception as e:
        print(f"  PyObjC skip: {e}")

try_force_60fps()

# Known thermal sensor resolutions — anything else (webcam, etc.) is skipped
_THERMAL_SIZES = {(640,512),(320,256),(160,120),(256,192),(320,240)}

cap = None
print("Scanning cameras:")
for idx in range(8):
    c = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not c.isOpened(): c.release(); continue
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, f = c.read()
    if not ret or f is None: c.release(); continue
    h, w = f.shape[:2]
    is_thermal = (w, h) in _THERMAL_SIZES
    print(f"  [{idx}] {w}x{h}  {'<-- thermal' if is_thermal else '(skipped)'}")
    if not is_thermal: c.release(); continue
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'I420'))
    c.set(cv2.CAP_PROP_FPS, 60)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"  Using camera {idx}: {w}x{h}  reported={c.get(cv2.CAP_PROP_FPS):.0f}fps")
    cap = c; break

if cap is None:
    print("No thermal camera found (is it plugged in? check System Settings -> Camera)")
    raise SystemExit

# ── Threaded grabber with _new flag ───────────────────────────────────────────
_frame = None; _new = False; _lock = threading.Lock(); _alive = True
_cam_n = 0; _cam_t = time.perf_counter(); _cam_fps = 0.0

def _grab():
    global _frame, _new, _cam_n, _cam_t, _cam_fps
    while _alive:
        ret, f = cap.read()
        if ret and f is not None:
            with _lock:
                _frame = f; _new = True
            _cam_n += 1
            now = time.perf_counter()
            if now - _cam_t >= 1.0:
                _cam_fps = _cam_n / (now - _cam_t)
                _cam_n = 0; _cam_t = now

threading.Thread(target=_grab, daemon=True).start()

# ── Iron LUT ──────────────────────────────────────────────────────────────────
def _build_iron():
    stops = [(0,(0,0,0)),(64,(102,0,102)),(128,(0,0,204)),
             (192,(0,153,255)),(255,(255,255,255))]
    lut = np.zeros((256,1,3), np.uint8)
    for i in range(256):
        for j in range(len(stops)-1):
            v0,c0=stops[j]; v1,c1=stops[j+1]
            if v0<=i<=v1:
                f=(i-v0)/(v1-v0)
                lut[i,0]=[int(c0[k]+f*(c1[k]-c0[k])) for k in range(3)]
                break
    return lut
IRON      = _build_iron()
IRON_FLAT = IRON[:, 0, :]   # (256, 3) — for cv2.LUT (faster than numpy fancy indexing)

WIN = "fps_bare"
if DISPLAY_MODE > 0:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    h, w = 768, 960
    cv2.resizeWindow(WIN, w, h)

disp_n = 0; disp_t = time.perf_counter()
t_proc_acc = 0.0; t_show_acc = 0.0; t_key_acc = 0.0

print(f"\nDISPLAY_MODE={DISPLAY_MODE}  (0=capture-only 1=raw 2=gray 3=colormap+resize)")
print("Press Q/Esc to quit, 0-3 to change mode at runtime\n")

while True:
    # --- grab ---
    with _lock:
        if not _new or _frame is None:
            frame = None
        else:
            frame = _frame; _new = False

    t0 = time.perf_counter()

    if frame is not None:
        if DISPLAY_MODE >= 2:
            # Single-channel slice is faster than cvtColor for a camera that
            # outputs identical BGR channels (FLIR Boson with I420 format).
            gray = frame[:, :, 0] if frame.ndim == 3 else frame
        if DISPLAY_MODE >= 3:
            # convertScaleAbs uses SIMD; avoids two-pass NORM_MINMAX overhead.
            mn = int(gray.min()); mx = int(gray.max())
            alpha = 255.0 / (mx - mn) if mx > mn else 1.0
            n8   = cv2.convertScaleAbs(gray, alpha=alpha, beta=-mn * alpha)
            # IRON_FLAT[n8] is numpy fancy-indexing into a (256,3) table;
            # equivalent to IRON[n8,0] but avoids the unit-dimension axis.
            col  = IRON_FLAT[n8]
            # Don't resize in Python — WINDOW_NORMAL lets Core Animation
            # GPU-scale the native 640×512 image to the 960×768 window for
            # free.  Skipping this saves ~1.5ms of proc time.
            disp = col
        elif DISPLAY_MODE == 2:
            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif DISPLAY_MODE == 1:
            disp = frame
        disp_n += 1

    t_proc_acc += time.perf_counter() - t0

    if DISPLAY_MODE > 0 and frame is not None:
        t1 = time.perf_counter()
        cv2.imshow(WIN, disp)
        t_show_acc += time.perf_counter() - t1

    t2 = time.perf_counter()
    # waitKey(1) every frame: drains the macOS Cocoa event loop so Core
    # Animation actually commits each imshow() to screen.  With only 4ms of
    # processing ahead of it, waitKey syncs to the next VSync boundary
    # (~16.7ms) for free — we were going to wait for the next camera frame
    # anyway.  Skipping this (or using pollKey every 6 frames) causes macOS to
    # buffer imshow calls and only push them to screen every pollKey call,
    # making the window look like ~10 fps even though disp_n counts 60/s.
    key = cv2.waitKey(1) & 0xFF
    t_key_acc += time.perf_counter() - t2

    now = time.perf_counter()
    if now - disp_t >= 1.0:
        disp_fps = disp_n / (now - disp_t)
        n = max(1, disp_n)
        print(f"cam={_cam_fps:5.1f}fps  disp={disp_fps:5.1f}fps  "
              f"proc={t_proc_acc/n*1e3:5.2f}ms  "
              f"show={t_show_acc/n*1e3:5.2f}ms  "
              f"key={t_key_acc/n*1e3:4.2f}ms  "
              f"total={( t_proc_acc+t_show_acc+t_key_acc)/n*1e3:5.2f}ms")
        disp_n = 0; disp_t = now
        t_proc_acc = t_show_acc = t_key_acc = 0.0

    if   key in (ord('q'), 27): break
    elif key == ord('0'): DISPLAY_MODE = 0; print("Mode 0: capture only")
    elif key == ord('1'): DISPLAY_MODE = 1; print("Mode 1: raw BGR")
    elif key == ord('2'): DISPLAY_MODE = 2; print("Mode 2: gray")
    elif key == ord('3'): DISPLAY_MODE = 3; print("Mode 3: colormap+resize")

_alive = False
cap.release()
if DISPLAY_MODE > 0:
    cv2.destroyAllWindows()
print("Done")
