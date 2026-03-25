# Release Checklist

This project has two release surfaces:

- the Python app package (`thermal-viewer`)
- the optional Rust accelerator wheel (`thermal_core`)

Use this checklist before tagging a release.

## Automated checks

Run from the repo root:

```bash
python3 -m py_compile thermal_viewer.py fps_bare.py setup.py
python3 -m pyflakes thermal_viewer.py fps_bare.py
python3 -m pip install --force-reinstall --no-deps .
```

Run from `thermal_core/`:

```bash
cargo clippy --release -- -D warnings
python3 -m maturin build --release
python3 -m pip install --force-reinstall target/wheels/thermal_core-*.whl
python3 - <<'EOF'
import numpy as np, thermal_core as tc
gray   = np.random.randint(0, 256, 80*60, dtype=np.uint8)
nuc_o  = np.zeros(80*60, dtype=np.float32)
prev_f = np.zeros(80*60, dtype=np.float32)
lut    = np.zeros(768, dtype=np.uint8)
out    = np.empty(80*60*3, dtype=np.uint8)
tc.process_frame(gray, 80, 60, nuc_o, prev_f, 0.05, 50.0, lut, out)
assert out.shape == (80*60*3,)
print("thermal_core smoke-test PASSED")
EOF
```

## Manual checks on real hardware

These cannot be proven in CI and must be checked on a machine with a real camera.

1. Launch the app and confirm the correct sensor profile and FPS are detected.
2. Verify fullscreen opens by default.
3. Record a short clip with microphone audio enabled.
4. Confirm the screen stays awake while recording.
5. Stop the recording and confirm:
   - the save status is explicit
   - `LAST SAVE` shows `OPEN` and `REVEAL`
   - the saved file opens correctly
6. Run one fresh `Shift+F` flat-field calibration on a uniform scene and confirm the Live view does not show the old square artifact.
7. Press `Esc` and `Q` to confirm both quit paths work.

## Publishing

1. Update versions in [pyproject.toml](/Users/sharpy/thermal-viewer/pyproject.toml) and [thermal_core/Cargo.toml](/Users/sharpy/thermal-viewer/thermal_core/Cargo.toml).
2. Commit the version bump.
3. Tag the release: `git tag vX.Y.Z`
4. Push the tag: `git push origin vX.Y.Z`
5. Wait for `.github/workflows/build-wheels.yml` to build and attach wheel artifacts to the GitHub Release.
6. Copy any released wheels you want bundled in-repo into [thermal_core/wheels](/Users/sharpy/thermal-viewer/thermal_core/wheels).
7. Commit the bundled wheel updates if needed.

## Notes

- `ffmpeg` is required for microphone audio to end up in the final MP4.
- On macOS, `pip install .` and `install.sh` install the native audio helper dependencies.
- If disk space is too low at stop-time, the app preserves video and leaves the raw audio sidecar on disk for recovery instead of silently pretending audio succeeded.
