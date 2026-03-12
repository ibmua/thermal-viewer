# Thermal Viewer

Open-source Python thermal camera viewer for USB UVC thermal cameras.
Clean sidebar UI, noise correction, flat-field calibration, video recording.

---

## Supported cameras

| Camera | Resolution | FPS |
|---|---|---|
| FLIR Boson 640 | 640 × 512 | 60 |
| FLIR Boson 320 | 320 × 256 | 60 |
| FLIR Lepton 3.x (via PureThermal) | 160 × 120 | 9 |
| InfiRay / Xinfrared | 256 × 192 | 25 |
| Seek Compact | 320 × 240 | 15 |
| Generic UVC thermal | various | 30 |

Camera is **auto-detected** by resolution on startup. No configuration needed.

---

## Requirements

- Python 3.9+
- macOS (primary target; Linux/Windows should work with minor changes)

```bash
pip install opencv-python numpy
```

On macOS, grant **Terminal** camera access in
System Settings → Privacy & Security → Camera.

---

## Usage

```bash
python3 thermal_viewer.py
```

### Controls

| Key / Mouse | Action |
|---|---|
| **Left-click** image | Place a marker |
| **Right-click** image | Remove nearest marker |
| **Click** sidebar buttons | Toggle modes, cycle colormap |
| **SPACE** | Freeze / unfreeze current frame |
| **R** | Start / stop video recording |
| **C** | Cycle colormap (Iron / Inferno / Turbo / Jet / Hot / Gray) |
| **N** | Cycle view: Live → NUC Map → Raw |
| **D** | Toggle Motion NUC denoising |
| **F** | Flat-field calibration (point at uniform surface first) |
| **T** | Toggle temporal smoothing |
| **H** | Flip horizontal |
| **V** | Flip vertical |
| **S** | Save PNG snapshot |
| **Q / Esc** | Quit |

---

## UI layout

```
┌─────────────────────────────────────────────────────────┐
│  Camera image (clean — no overlays)       │  Sidebar    │
│                                           │  ─ sensor   │
│  Only marker crosshairs drawn on image    │  ─ histogram│
│  (tiny M1, M2 labels)                     │  ─ stats    │
│                                           │  ─ colormap │
│                                           │  ─ view mode│
│                                           │  ─ flat-NUC │
│                                           │  ─ modes    │
│                                           │  ─ recording│
│                                           │  ─ markers  │
├───────────────────────────────────────────────────────── │
│  Status bar                                              │
└─────────────────────────────────────────────────────────┘
```

Colormap, scale, mode badges, markers, and controls are all in the **sidebar** — nothing overlaid on the thermal image except small marker labels.

---

## Noise correction pipeline

Three independent, stackable corrections:

1. **Flat-field NUC** (one-point calibration)
   Point at any uniform surface → press **F** → hold 2 seconds.
   Computes per-pixel offset correction and detects stuck/dead pixels.
   Calibration is saved to `~/.thermal_viewer/flatfield_WxH.npz` and loaded automatically next run.

2. **Motion-gated NUC** (scene-based, always running in background)
   Learns the fixed-pattern noise from scene motion using an LMS algorithm.
   No calibration target needed. Toggle with **D**.

3. **Temporal EMA smoothing**
   Exponential moving average (α = 0.35) to reduce per-frame random noise (NETD).
   Toggle with **T**.

### NUC Map view

Press **N** to switch to **NUC Map** — shows the flat-field correction as a colorized heatmap. Bright = pixels that were consistently too bright (corrected down). Dark = pixels too dark (corrected up). Useful for verifying calibration quality.

---

## Video recording

Press **R** to start recording. The colorized camera image is saved as `.mp4` to `~/Desktop/BosonCaptures/`. Press **R** again to stop. File size and duration are printed to the terminal.

---

## Notes on FLIR Boson FPS

The Boson 640 supports 60 fps but USB 2.0 bandwidth limits raw BGRA at that rate. This viewer pre-configures the device via PyObjC (macOS AVFoundation) before OpenCV opens the session, and requests I420 (YUV planar) pixel format which fits in ~28 MB/s. Expect 50–60 fps reported.

If PyObjC is not available the viewer falls back gracefully to whatever rate OpenCV negotiates (typically 30 fps).

---

## License

MIT
