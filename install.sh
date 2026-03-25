#!/usr/bin/env bash
# thermal-viewer installer — macOS / Linux
# For Windows use install.bat instead.
set -e

REPO="ibmua/thermal-viewer"
GITHUB_RELEASE_URL="https://github.com/${REPO}/releases/latest/download"

echo "=== thermal-viewer installer ==="

# ── 1. Python deps ─────────────────────────────────────────────────────────────
echo "Installing Python package and dependencies..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install . --quiet
echo "  thermal-viewer + Python dependencies OK"

# ── 2. Identify the right wheel ────────────────────────────────────────────────
WHEEL_DIR="thermal_core/wheels"
PYTHON_TAG=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
ARCH=$(uname -m)
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')

if   [ "$ARCH" = "arm64"  ] && [ "$PLATFORM" = "darwin" ]; then
    WHEEL_GLOB="thermal_core-*-${PYTHON_TAG}-*-macosx_*_arm64.whl"
elif [ "$ARCH" = "x86_64" ] && [ "$PLATFORM" = "darwin" ]; then
    WHEEL_GLOB="thermal_core-*-${PYTHON_TAG}-*-macosx_*_x86_64.whl"
elif [ "$ARCH" = "x86_64" ] && [ "$PLATFORM" = "linux" ]; then
    WHEEL_GLOB="thermal_core-*-${PYTHON_TAG}-*-manylinux*_x86_64.whl"
elif [ "$ARCH" = "aarch64" ] && [ "$PLATFORM" = "linux" ]; then
    WHEEL_GLOB="thermal_core-*-${PYTHON_TAG}-*-manylinux*_aarch64.whl"
else
    WHEEL_GLOB=""
fi

INSTALLED_TC=0

# ── 3a. Try local pre-built wheel ──────────────────────────────────────────────
if [ -n "$WHEEL_GLOB" ]; then
    WHEEL=$(ls "$WHEEL_DIR"/$WHEEL_GLOB 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo "Installing pre-built thermal_core from $WHEEL ..."
        python3 -m pip install "$WHEEL" --force-reinstall --quiet
        INSTALLED_TC=1
        echo "  thermal_core (Rust) OK — 60fps fast path enabled"
    fi
fi

# ── 3b. Try GitHub Release download ────────────────────────────────────────────
if [ "$INSTALLED_TC" -eq 0 ] && [ -n "$WHEEL_GLOB" ] && command -v curl &>/dev/null; then
    WHEEL_NAME=$(python3 -c "
import sys, platform
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
arch = platform.machine()
plat = sys.platform
if plat == 'darwin':
    mac_ver = platform.mac_ver()[0].replace('.','_').split('_')
    mac_maj = mac_ver[0]
    arch_tag = 'arm64' if arch == 'arm64' else 'x86_64'
    print(f'thermal_core-0.1.0-{tag}-{tag}-macosx_{mac_maj}_0_{arch_tag}.whl')
elif plat == 'linux':
    arch_tag = 'aarch64' if arch == 'aarch64' else 'x86_64'
    print(f'thermal_core-0.1.0-{tag}-{tag}-manylinux_2_17_{arch_tag}.manylinux2014_{arch_tag}.whl')
" 2>/dev/null)

    if [ -n "$WHEEL_NAME" ]; then
        TMP_WHEEL="/tmp/$WHEEL_NAME"
        echo "Trying to download $WHEEL_NAME from GitHub Releases..."
        if curl -fsSL "${GITHUB_RELEASE_URL}/${WHEEL_NAME}" -o "$TMP_WHEEL" 2>/dev/null; then
            python3 -m pip install "$TMP_WHEEL" --force-reinstall --quiet
            cp "$TMP_WHEEL" "$WHEEL_DIR/" 2>/dev/null || true
            INSTALLED_TC=1
            echo "  thermal_core (Rust) OK — 60fps fast path enabled"
        else
            echo "  (no pre-built wheel found on GitHub Releases for this platform)"
        fi
    fi
fi

# ── 3c. Fall back: build from source ───────────────────────────────────────────
if [ "$INSTALLED_TC" -eq 0 ]; then
    if command -v cargo &>/dev/null; then
        if ! command -v maturin &>/dev/null; then
            echo "Installing maturin..."
            python3 -m pip install maturin --quiet
        fi
        echo "Building thermal_core from source (this takes ~30 seconds)..."
        pushd thermal_core > /dev/null
        python3 -m maturin build --release --quiet
        WHEEL=$(ls target/wheels/thermal_core-*.whl 2>/dev/null | head -1)
        if [ -n "$WHEEL" ]; then
            python3 -m pip install "$WHEEL" --force-reinstall --quiet
            INSTALLED_TC=1
            echo "  thermal_core (Rust) OK — 60fps fast path enabled"
        fi
        popd > /dev/null
    fi
fi

if [ "$INSTALLED_TC" -eq 0 ]; then
    echo ""
    echo "  ⚠  thermal_core not available — viewer will run in pure-Python mode (~55fps)"
    echo "     To unlock 60fps:"
    echo "       1. Install Rust:  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "       2. Re-run this script."
fi

echo ""
if command -v ffmpeg &>/dev/null; then
    echo "  ffmpeg OK — microphone recording can be muxed into MP4"
else
    echo "  ⚠  ffmpeg not found — video recording works, but microphone audio"
    echo "     cannot be finalized into the saved MP4 until ffmpeg is installed."
fi

echo ""
echo "=== Install complete ==="
echo ""
if [ "$(uname -s)" = "Darwin" ]; then
    echo "macOS: if the camera doesn't open, grant Terminal camera access:"
    echo "  System Settings → Privacy & Security → Camera"
    echo "macOS: for microphone recording, also grant Terminal microphone access."
    echo ""
fi
echo "Run with:  python3 thermal_viewer.py"
if command -v thermal-viewer &>/dev/null; then
    echo "      or:  thermal-viewer"
else
    USER_BIN=$(python3 -c "import site; print(site.USER_BASE + '/bin')")
    echo "Console script installed at:  $USER_BIN/thermal-viewer"
fi
