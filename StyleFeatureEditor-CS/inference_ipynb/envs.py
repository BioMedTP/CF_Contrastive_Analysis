# envs.py
import os, shutil, subprocess, sys
from pathlib import Path
from typing import Optional, Tuple
import gc
import torch
import pynvml

# envs.py

def make_gpu_mem_reporter(gpu_index=0, clear_cache_on_setup=True):
    """
    Create a GPU memory reporter closure.

    Reports:
      - torch.cuda.memory_allocated / reserved (current process)
      - NVML total GPU used / total (all processes)

    Args:
        gpu_index: NVML GPU index (0 -> GPU:0)
        clear_cache_on_setup: run gc.collect() + torch.cuda.empty_cache() once
    """
    import gc
    import torch

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception as e:
        handle = None
        pynvml = None
        nvml_err = e

    if clear_cache_on_setup and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    def print_memory(prefix=""):
        if not torch.cuda.is_available():
            print(f"[{prefix}] CUDA not available.\n")
            return

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved() / 1024**3

        print(f"[{prefix}]")
        print(f"  - PyTorch Allocated : {allocated:.2f} GB")
        print(f"  - PyTorch Reserved  : {reserved:.2f} GB")

        if handle is None:
            print(f"  - Total GPU Used    : NVML unavailable ({nvml_err})\n")
            return

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used  = mem_info.used / 1024**3
        total = mem_info.total / 1024**3
        print(f"  - Total GPU Used    : {used:.2f} GB / {total:.2f} GB\n")

    return print_memory




def _which(cmd: str) -> Optional[str]:
    p = shutil.which(cmd)
    return p if p else None

def _dedup(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def actenv_jupyter(cuda_version: str = "12.5",
                   gcc_version: Optional[str] = None,
                   clear_cache: bool = True,
                   set_verbose: bool = True) -> dict:
    """
    Configure the Jupyter kernel env for PyTorch C++/CUDA extensions.
    - cuda_version: e.g. "12.5", uses /usr/local/cuda-<version>
    - gcc_version:  e.g. "11" to prefer g++-11/gcc-11; if None, auto-pick a modern g++
    - clear_cache:  remove ~/.cache/torch_extensions & ~/.cache/torch/inductor
    - set_verbose:  enable verbose build logs and single-threaded builds
    NOTE: Run BEFORE importing torch.
    Returns a dict summary of the effective settings.
    """

    # # 0) Warn if torch already imported
    # if "torch" in sys.modules:
    #     print("[actenv_jupyter] WARNING: torch already imported; "
    #           "restart kernel and run this first for cleanest results.")

    # 1) CUDA paths
    cuda_home = f"/usr/local/cuda-{cuda_version}"
    if not os.path.isdir(cuda_home):
        raise FileNotFoundError(f"{cuda_home} not found on this node.")
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH","")
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH","")
    print(f"[actenv_jupyter] CUDA_HOME={cuda_home}")

    # 2) TORCH_CUDA_ARCH_LIST from actual GPUs (handles multi-type nodes)
    try:
        caps = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip().splitlines()
        caps = [c.strip() for c in caps if c.strip()]
        if caps:
            arch_list = ",".join(_dedup(caps))
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
            print(f"[actenv_jupyter] TORCH_CUDA_ARCH_LIST={arch_list}")
    except Exception as e:
        print(f"[actenv_jupyter] WARN: could not query nvidia-smi compute_cap ({e}). "
              "You can set TORCH_CUDA_ARCH_LIST manually if needed.")

    # 3) Pin host compilers (critical). Prefer requested gcc_version if provided.
    cxx_candidates = []
    cc_candidates  = []
    if gcc_version:
        cxx_candidates += [f"g++-{gcc_version}"]
        cc_candidates  += [f"gcc-{gcc_version}"]
    # fallbacks in descending preference
    cxx_candidates += ["g++-12","g++-11","g++-10","g++"]
    cc_candidates  += ["gcc-12","gcc-11","gcc-10","gcc"]

    cxx = next((p for c in cxx_candidates if (p:=_which(c))), None)
    cc  = next((p for c in cc_candidates  if (p:=_which(c))), None)

    if cxx:
        os.environ["CXX"] = cxx
        os.environ["CUDAHOSTCXX"] = cxx  # ensure nvcc uses this g++
    if cc:
        os.environ["CC"] = cc

    # C++ standard for PyTorch headers; let Torch decide ABI itself (0/1)
    os.environ["CXXFLAGS"] = "-std=c++17"
    os.environ.pop("_GLIBCXX_USE_CXX11_ABI", None)

    print(f"[actenv_jupyter] CXX={os.environ.get('CXX')} | CC={os.environ.get('CC')}")

    # 4) Optional cache clear
    if clear_cache:
        for p in (Path.home()/".cache/torch_extensions", Path.home()/".cache/torch/inductor"):
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception as e:
                print(f"[actenv_jupyter] WARN: failed to remove {p}: {e}")
        print("[actenv_jupyter] Cleared torch caches.")

    # 5) Optional verbose build flags
    if set_verbose:
        os.environ["TORCH_CUDA_VERBOSE_BUILD"] = "1"
        os.environ["MAX_JOBS"] = "1"  # increase later for speed once stable

    # 6) Summary (no torch import)
    summary = {
        "CUDA_HOME": os.environ["CUDA_HOME"],
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "CC": os.environ.get("CC"),
        "CXX": os.environ.get("CXX"),
        "CUDAHOSTCXX": os.environ.get("CUDAHOSTCXX"),
        "CXXFLAGS": os.environ.get("CXXFLAGS"),
        "PATH_head": ":".join(os.environ["PATH"].split(":")[:3]),
        "LD_LIBRARY_PATH_head": ":".join(os.environ.get("LD_LIBRARY_PATH","").split(":")[:3]),
        "TORCH_CUDA_VERBOSE_BUILD": os.environ.get("TORCH_CUDA_VERBOSE_BUILD"),
        "MAX_JOBS": os.environ.get("MAX_JOBS"),
    }
    print("[actenv_jupyter] Ready.")
    return summary

def change_exp_dir(path: str):
    os.chdir(path)
    print("cwd:", os.getcwd())
    return os.getcwd()