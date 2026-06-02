"""pipeline_enhancer.py — 4x-UltraSharp ESRGAN upscaler (High Quality mode).

Downloads 4x-UltraSharp.pth from HuggingFace Hub on first use (~67 MB).
Uses the custom RRDB architecture in esrgan_model.py (handles old-arch format).
Runs on MPS (Apple Silicon) — fast, no diffusion, no system hang risk.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# ── Device ────────────────────────────────────────────────────────────────────
_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_DTYPE  = torch.float32

# ── Checkpoint ────────────────────────────────────────────────────────────────
_ESRGAN_REPO     = "philz1337x/upscaler"
_ESRGAN_FILENAME = "4x-UltraSharp.pth"
_ESRGAN_REVISION = "011deacac8270114eb7d2eeff4fe6fa9a837be70"
_LOCAL_WEIGHTS   = Path(__file__).parent / "weights" / _ESRGAN_FILENAME
_upscaler_cache: dict = {}


def _load_upscaler():
    if "upscaler" in _upscaler_cache:
        return _upscaler_cache["upscaler"]

    from esrgan_model import UpscalerESRGAN

    print("[HQ] Resolving 4x-UltraSharp checkpoint...")
    if _LOCAL_WEIGHTS.exists():
        weight_path = _LOCAL_WEIGHTS
        print(f"[HQ] Found local weights at {weight_path}")
    else:
        weight_path = Path(hf_hub_download(
            repo_id=_ESRGAN_REPO,
            filename=_ESRGAN_FILENAME,
            revision=_ESRGAN_REVISION,
        ))
    print(f"[HQ] Loading onto {_DEVICE}...")
    upscaler = UpscalerESRGAN(weight_path, device=_DEVICE, dtype=_DTYPE)
    _upscaler_cache["upscaler"] = upscaler
    print("[HQ] Ready.")
    return upscaler


def pipeline_enhance(
    image: Image.Image,
    prompt: str = "",          # unused — kept for API compatibility with app.py
    **_kwargs,
) -> Image.Image:
    """Upscale with 4x-UltraSharp ESRGAN. Fast on Apple Silicon MPS."""
    upscaler = _load_upscaler()
    result = upscaler.upscale_with_tiling(image.convert("RGB"))
    if _DEVICE.type == "mps":
        torch.mps.empty_cache()
    return result


def predownload_all() -> None:
    """Pre-download the 4x-UltraSharp weight to the HuggingFace cache."""
    print("[HQ] Downloading 4x-UltraSharp (~67 MB)...")
    hf_hub_download(
        repo_id=_ESRGAN_REPO,
        filename=_ESRGAN_FILENAME,
        revision=_ESRGAN_REVISION,
    )
    print("[HQ] Done. Run 'make run' to start the app.")


if __name__ == "__main__":
    predownload_all()
