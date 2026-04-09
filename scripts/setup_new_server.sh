#!/bin/bash
# ============================================================
# Cosmos-Policy + Adaptive TTC-WAM: New Server Setup Script
# ============================================================
# Tested on: Ubuntu 22.04, CUDA 12.x, Python 3.10
# Target: B200 multi-GPU server
#
# Prerequisites:
#   - CUDA 12.x drivers installed
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - gh CLI installed (for cloning private repos)
#   - cmake installed (apt install cmake)
#   - HuggingFace access to nvidia/Cosmos-Policy-LIBERO-Predict2-2B
#
# Usage:
#   bash scripts/setup_new_server.sh [--with-robocasa] [--with-libero] [--skip-model-download]
#
# This script automates all the environment hacks discovered during
# the initial setup (see notes/exp0zero_phase_a_progress.md Section 5).
# ============================================================

set -euo pipefail

# ==========================================
# Parse arguments
# ==========================================
WITH_ROBOCASA=false
WITH_LIBERO=false
SKIP_MODEL=false
for arg in "$@"; do
    case $arg in
        --with-robocasa) WITH_ROBOCASA=true ;;
        --with-libero) WITH_LIBERO=true ;;
        --skip-model-download) SKIP_MODEL=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo "Cosmos-Policy Setup"
echo "  Repo root: $REPO_ROOT"
echo "  RoboCasa:  $WITH_ROBOCASA"
echo "  LIBERO:    $WITH_LIBERO"
echo "  Date:      $(date)"
echo "============================================"

# ==========================================
# Step 1: Create virtual environment
# ==========================================
echo ""
echo "[Step 1/6] Creating virtual environment..."

if [ ! -d ".venv" ]; then
    uv venv --python 3.10 .venv
    echo "  Created .venv with Python 3.10"
else
    echo "  .venv already exists, skipping"
fi

PYTHON="$REPO_ROOT/.venv/bin/python3"
PIP="$REPO_ROOT/.venv/bin/pip"

# ==========================================
# Step 2: Install dependencies
# ==========================================
echo ""
echo "[Step 2/6] Installing dependencies..."

# Install cmake first (needed by egl-probe, a transitive dep of libero)
if ! command -v cmake &>/dev/null; then
    echo "  WARNING: cmake not found. Install it: sudo apt install cmake"
    echo "  Some dependencies (egl-probe) will fail without cmake."
fi

# Base install
EXTRAS=""
if $WITH_LIBERO; then
    EXTRAS="$EXTRAS --group libero"
fi
if $WITH_ROBOCASA; then
    EXTRAS="$EXTRAS --group robocasa"
fi

echo "  Running: uv sync $EXTRAS"
uv sync $EXTRAS

echo "  Dependencies installed!"

# ==========================================
# Step 3: NVIDIA library shims
# ==========================================
echo ""
echo "[Step 3/6] Setting up NVIDIA library shims..."

NVIDIA_BASE="$REPO_ROOT/.venv/lib/python3.10/site-packages/nvidia"
SHIM_DIR="$REPO_ROOT/.venv/lib/nvidia-libs"
mkdir -p "$SHIM_DIR"

# Find all nvidia lib directories
ALL_NVIDIA_LIBS=$(find "$NVIDIA_BASE" -name "lib" -type d 2>/dev/null | paste -sd: || echo "")

if [ -z "$ALL_NVIDIA_LIBS" ]; then
    echo "  WARNING: No nvidia packages found in venv. GPU features may not work."
else
    # Create unversioned symlinks for libraries that ctypes.CDLL loads
    # transformer_engine loads "libcudnn.so" but only libcudnn.so.9 exists
    for lib in libcudnn libcudnn_graph libcudnn_engines_runtime_compiled libcudnn_engines_precompiled libcudnn_heuristic libcudnn_adv libnvrtc libnvrtc-builtins; do
        versioned=$(find "$NVIDIA_BASE" -name "${lib}.so.*" -type f 2>/dev/null | head -1)
        if [ -n "$versioned" ] && [ ! -e "$SHIM_DIR/${lib}.so" ]; then
            ln -sf "$versioned" "$SHIM_DIR/${lib}.so"
            echo "  Linked: ${lib}.so -> $(basename $versioned)"
        fi
    done

    # Create fake ldconfig script
    # transformer_engine uses `ldconfig -p | grep libnvrtc` which ignores LD_LIBRARY_PATH
    cat > "$SHIM_DIR/ldconfig" << 'LDCONFIG_EOF'
#!/bin/bash
# Shim ldconfig: appends nvidia lib paths from venv to real ldconfig output
REAL_LDCONFIG=$(which -a ldconfig 2>/dev/null | grep -v "$0" | head -1)
if [ -z "$REAL_LDCONFIG" ]; then
    REAL_LDCONFIG=/sbin/ldconfig
fi

if [[ "$*" == *"-p"* ]]; then
    $REAL_LDCONFIG "$@" 2>/dev/null || true
    # Append nvidia libs from the venv
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    NVIDIA_BASE="$REPO_ROOT/.venv/lib/python3.10/site-packages/nvidia"
    for dir in $(find "$NVIDIA_BASE" -name "lib" -type d 2>/dev/null); do
        for so in "$dir"/*.so*; do
            [ -f "$so" ] && echo "	$(basename $so) (libc6,x86-64) => $so"
        done
    done
else
    $REAL_LDCONFIG "$@"
fi
LDCONFIG_EOF
    chmod +x "$SHIM_DIR/ldconfig"
    echo "  Created fake ldconfig in shim dir"
fi

# ==========================================
# Step 4: sitecustomize.py (robosuite log fix)
# ==========================================
echo ""
echo "[Step 4/6] Installing sitecustomize.py..."

SITE_PACKAGES="$REPO_ROOT/.venv/lib/python3.10/site-packages"
cat > "$SITE_PACKAGES/sitecustomize.py" << 'SITECUSTOM_EOF'
import os
import logging

# Fix: robosuite hardcodes /tmp/robosuite.log which may be owned by another user
os.environ.setdefault("ROBOSUITE_LOG", f"/tmp/robosuite_{os.getuid()}.log")

_orig_FileHandler = logging.FileHandler
class _PatchedFileHandler(logging.FileHandler):
    def __init__(self, filename, *args, **kwargs):
        if filename == "/tmp/robosuite.log":
            filename = f"/tmp/robosuite_{os.getuid()}.log"
        super().__init__(filename, *args, **kwargs)

logging.FileHandler = _PatchedFileHandler
SITECUSTOM_EOF
echo "  Installed sitecustomize.py (robosuite log redirect)"

# ==========================================
# Step 5: LIBERO config (if needed)
# ==========================================
if $WITH_LIBERO; then
    echo ""
    echo "[Step 5/6] Setting up LIBERO config..."

    LIBERO_PKG=$(find "$SITE_PACKAGES" -path "*/libero/libero" -type d | head -1)
    if [ -n "$LIBERO_PKG" ]; then
        mkdir -p ~/.libero
        cat > ~/.libero/config.yaml << LIBERO_EOF
assets: ${LIBERO_PKG}/assets
bddl_files: ${LIBERO_PKG}/bddl_files
benchmark_root: ${LIBERO_PKG}
datasets: ${LIBERO_PKG}/../datasets
init_states: ${LIBERO_PKG}/init_files
LIBERO_EOF
        echo "  Created ~/.libero/config.yaml"
    else
        echo "  WARNING: libero package not found in site-packages"
    fi
else
    echo ""
    echo "[Step 5/6] Skipping LIBERO config (use --with-libero to enable)"
fi

# ==========================================
# Step 6: Download model weights
# ==========================================
if ! $SKIP_MODEL; then
    echo ""
    echo "[Step 6/6] Downloading model weights..."
    echo "  Downloading nvidia/Cosmos-Policy-LIBERO-Predict2-2B..."
    $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos-Policy-LIBERO-Predict2-2B', local_dir='nvidia/Cosmos-Policy-LIBERO-Predict2-2B')
print('  Model downloaded!')
" 2>&1 || echo "  WARNING: Model download failed. You may need to run 'huggingface-cli login' first."
else
    echo ""
    echo "[Step 6/6] Skipping model download (--skip-model-download)"
fi

# ==========================================
# Print environment setup commands
# ==========================================
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To run experiments, source the environment first:"
echo ""
echo "  # Add to your shell or run before experiments:"
echo "  export PATH=\"$SHIM_DIR:\$PATH\""
echo "  export LD_LIBRARY_PATH=\"$SHIM_DIR:$ALL_NVIDIA_LIBS\""
echo ""
echo "  # Then run (example: Phase A on GPU 0):"
echo "  cd $REPO_ROOT"
echo "  CUDA_VISIBLE_DEVICES=0 $PYTHON -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \\"
echo "      --config cosmos_predict2_2b_480p_libero__inference_only \\"
echo "      --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \\"
echo "      ... (see scripts/run_experiment.sh)"
echo ""
echo "Quick test:"
echo "  $PYTHON -c 'import cosmos_policy; print(\"OK\")'"
echo ""
