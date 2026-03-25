# Contributing to Thermal Viewer

Thank you for considering contributing! Here's everything you need to get started.

---

## Development setup

```bash
git clone https://github.com/ibmua/thermal-viewer
cd thermal-viewer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy opencv-python

# Build the Rust extension in development mode (requires Rust ≥ 1.75)
cd thermal_core && maturin develop --release && cd ..

python thermal_viewer.py
```

---

## Project layout

```
thermal_viewer.py          Main Python application
fps_bare.py                Standalone frame-rate benchmark
thermal_core/
  src/lib.rs               Rust hot-loop (flat-field, NUC, LUT, resize)
  Cargo.toml
  wheels/                  Pre-built .whl files for distribution
.github/
  workflows/
    ci.yml                 Lint + Rust check on every PR
    build-wheels.yml       Cross-platform wheel build + GitHub Release on tags
install.sh / install.bat   End-user install scripts
pyproject.toml
```

---

## Making changes to the Rust extension

After editing `thermal_core/src/lib.rs`:

```bash
cd thermal_core
maturin develop --release   # rebuild and reinstall into current venv
```

Run the smoke-test to verify:

```bash
python - <<'EOF'
import numpy as np, thermal_core as tc
gray  = np.random.randint(0, 256, 80*60, dtype=np.uint8)
nuc_o = np.zeros(80*60, dtype=np.float32)
prev  = np.zeros(80*60, dtype=np.float32)
lut   = np.zeros(768,   dtype=np.uint8)
out   = tc.process_frame(gray, 80, 60, nuc_o, prev, 0.05, 50.0, lut)
assert len(out) == 80*60*3
print("OK")
EOF
```

---

## Submitting a pull request

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make your changes.
3. Run `cargo clippy` in `thermal_core/` and fix any warnings.
4. Open a PR against `main` — CI will run automatically.

---

## Adding a new camera

Edit `find_camera()` in `thermal_viewer.py`:
- Add your camera's expected resolution to the detection heuristic.
- Open a PR with a note about the camera model, resolution, and USB VID:PID if known.

---

## Releasing a new version

1. Update `version` in `pyproject.toml` and `thermal_core/Cargo.toml`.
2. Commit: `git commit -am "chore: bump version to X.Y.Z"`
3. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. The `build-wheels.yml` workflow builds wheels for all platforms and creates a GitHub Release automatically.
5. Copy the release wheels into `thermal_core/wheels/` and commit so `install.sh` can find them.

---

## Code style

- Python: PEP 8, type hints where practical, keep hot-loop code comment-heavy.
- Rust: `cargo fmt` before committing; no `unsafe` unless unavoidable.

---

## License

By contributing you agree that your changes will be licensed under the MIT License.
