#!/usr/bin/env bash
# Usage: source ./actenv.sh [-e ENV_NAME] [-c CUDA_VER] [-g GCC_VER] [--no-clear]
# Example: source ./actenv.sh -e styleGANenv -c 12.5 -g 11

set -euo pipefail

# ---------- Defaults ----------
ENV_NAME="styleGANenv"
CUDA_VER="12.5"
GCC_VER="11"
CLEAR_CACHE=1

# ---------- Parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env)   ENV_NAME="$2"; shift 2 ;;
    -c|--cuda)  CUDA_VER="$2"; shift 2 ;;
    -g|--gcc)   GCC_VER="$2"; shift 2 ;;
    --no-clear) CLEAR_CACHE=0; shift ;;
    *) echo "Unknown arg: $1"; return 2 2>/dev/null || exit 2 ;;
  esac
done

echo "[actenv] Config: env=${ENV_NAME} cuda=${CUDA_VER} gcc=${GCC_VER} clear_cache=${CLEAR_CACHE}"

# ---------- Modules (if available) ----------
if command -v module &>/dev/null; then
  echo "[actenv] Loading modules..."
  module purge
  module load "gcc/${GCC_VER}"
  module load "cuda/${CUDA_VER}"
else
  echo "[actenv] No 'module' command; skipping module loads."
fi

# ---------- Conda activate ----------
if [[ -z "${CONDA_EXE:-}" ]]; then
  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "[actenv] Could not find conda.sh; ensure Conda is installed" >&2
    return 1 2>/dev/null || exit 1
  fi
fi
conda activate "${ENV_NAME}"

# ---------- CUDA paths ----------
CUDA_HOME="/usr/local/cuda-${CUDA_VER}"
if [[ ! -d "${CUDA_HOME}" ]]; then
  echo "[actenv] ERROR: ${CUDA_HOME} not found on this node." >&2
  return 1 2>/dev/null || exit 1
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "[actenv] CUDA_HOME=${CUDA_HOME}"

# ---------- Clean caches (optional) ----------
if [[ "${CLEAR_CACHE}" -eq 1 ]]; then
  echo "[actenv] Clearing torch caches..."
  rm -rf "${HOME}/.cache/torch_extensions" "${HOME}/.cache/torch/inductor" || true
fi

# ---------- Pin compilers & flags ----------
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="${CXX}"
unset _GLIBCXX_USE_CXX11_ABI || true   # let PyTorch set ABI (0/1)
export CXXFLAGS="-std=c++17"

# ---------- Set arch list from actual GPUs ----------
if command -v nvidia-smi &>/dev/null; then
  export TORCH_CUDA_ARCH_LIST="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    | awk '{print $1}' | awk '!seen[$0]++' | paste -sd, -
  )"
else
  echo "[actenv] WARNING: nvidia-smi not found; TORCH_CUDA_ARCH_LIST not set."
fi

# ---------- Optional: verbose builds ----------
export TORCH_CUDA_VERBOSE_BUILD=1
export MAX_JOBS=1

# ---------- Print summary ----------
echo "[actenv] CC=$(which gcc)"
echo "[actenv] CXX=$(which g++)"
echo "[actenv] nvcc=$(command -v nvcc || echo 'nvcc not on PATH')"
echo "[actenv] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}"
python - <<'PY'
import torch, sys
print("[actenv] python:", sys.executable)
print("[actenv] torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
print("[actenv] ABI(_GLIBCXX_USE_CXX11_ABI):", torch._C._GLIBCXX_USE_CXX11_ABI)
print("[actenv] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[actenv] device:", torch.cuda.get_device_name(0), "cap=", torch.cuda.get_device_capability(0))
PY

echo "[actenv] Environment ready."
