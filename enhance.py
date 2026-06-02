import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet

# ── Device selection ──────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ── Local weights directory ───────────────────────────────────────────────────
WEIGHTS_DIR = Path(__file__).parent / "weights"

# ── Background upsampler model registry ──────────────────────────────────────────
BG_UPSAMPLER_CHOICES = [
    "RealESRGAN x4plus",
    "RealESRGAN x2plus",
    "realesr-general-x4v3",
    "4x-NMKD-Superscale-SP",
]

# ── AnimeGANv3 style registry ─────────────────────────────────────────────────
ANIMEGAN_STYLES = ["Hayao (Ghibli)", "Shinkai (Vivid)"]

_ANIMEGAN_FILES = {
    "Hayao (Ghibli)": "AnimeGANv3_Hayao_36.onnx",
    "Shinkai (Vivid)": "AnimeGANv3_Shinkai_37.onnx",
}

_animegan_cache: dict = {}


def load_animegan(style: str):
    if style in _animegan_cache:
        return _animegan_cache[style]
    fname = _ANIMEGAN_FILES.get(style)
    if fname is None:
        raise ValueError(f"Unknown AnimeGAN style: {style}")
    model_path = WEIGHTS_DIR / fname
    if not model_path.exists():
        raise FileNotFoundError(
            f"AnimeGAN model '{fname}' not found. Run: make weights"
        )
    import onnxruntime as ort  # lazy import — not needed for other modes
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    _animegan_cache[style] = session
    return session


def apply_animegan(img_bgr: np.ndarray, style: str) -> np.ndarray:
    """Photo → anime style transfer using AnimeGANv3 ONNX. In/out: BGR uint8."""
    session = load_animegan(style)
    h, w = img_bgr.shape[:2]

    def to_8s(x):
        return 256 if x < 256 else x - x % 8

    rw, rh = to_8s(w), to_8s(h)
    img_in = cv2.resize(img_bgr, (rw, rh))
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    img_in = np.expand_dims(img_in, axis=0)  # (1, H, W, 3)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run(None, {input_name: img_in})[0]

    result = (np.squeeze(result) + 1.0) / 2.0 * 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return cv2.resize(result_bgr, (w, h))


_BG_MODEL_CFGS = {
    "RealESRGAN x4plus": dict(
        model_fn=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_block=23, num_grow_ch=32, scale=4),
        scale=4, weight="RealESRGAN_x4plus.pth",
    ),
    "4x-NMKD-Superscale-SP": dict(
        model_fn=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_block=23, num_grow_ch=32, scale=4),
        scale=4, weight="4x_NMKD-Superscale-SP_178000_G.pth",
    ),
    "RealESRGAN x2plus": dict(
        model_fn=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_block=23, num_grow_ch=32, scale=2),
        scale=2, weight="RealESRGAN_x2plus.pth",
    ),
    "RealESRGAN anime 6B": dict(
        model_fn=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_block=6, num_grow_ch=32, scale=4),
        scale=4, weight="RealESRGAN_x4plus_anime_6B.pth",
    ),
    "realesr-general-x4v3": dict(
        model_fn=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                         num_conv=32, upscale=4, act_type="prelu"),
        scale=4, weight="realesr-general-x4v3.pth",
        wdn_weight="realesr-general-wdn-x4v3.pth",
    ),
}

_bg_cache: dict = {}


# ── Wrapper to give UpscalerESRGAN the same .enhance() interface as RealESRGANer ──
class _CustomESRGANWrapper:
    """Wraps esrgan_model.UpscalerESRGAN to match the RealESRGANer.enhance() interface."""

    def __init__(self, weight_path: Path, tile: int = 400):
        from esrgan_model import UpscalerESRGAN
        self.upscaler = UpscalerESRGAN(weight_path, device=torch.device(device), dtype=torch.float32)
        self.tile = tile  # accepted for API compatibility; tiling is always on

    def enhance(self, img_bgr: np.ndarray, outscale: int = 4):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(img_rgb)
        pil_out = self.upscaler.upscale_with_tiling(pil_in)
        if outscale != 4:
            new_w = int(img_bgr.shape[1] * outscale)
            new_h = int(img_bgr.shape[0] * outscale)
            pil_out = pil_out.resize((new_w, new_h), Image.LANCZOS)
        out_bgr = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGB2BGR)
        if device == "mps":
            torch.mps.empty_cache()
        return out_bgr, None  # (output_bgr, img_mode) — matches RealESRGANer tuple


_CUSTOM_ESRGAN_MODELS = {"4x-NMKD-Superscale-SP"}


def _normalize_esrgan_weight(weight_path: Path) -> None:
    """Wrap a bare state-dict .pth so RealESRGANer can load it.
    Community models (e.g. 4x-UltraSharp) ship without the 'params' envelope
    that RealESRGANer expects. This patches the file in-place, once."""
    d = torch.load(str(weight_path), map_location="cpu", weights_only=True)
    if isinstance(d, dict) and ("params" in d or "params_ema" in d):
        return  # already in the expected format
    torch.save({"params": d}, str(weight_path))
    print(f"[enhance] Normalized bare checkpoint → {weight_path.name}")


def make_bg_upsampler(model_name: str, denoise_strength: float = 0.5,
                      tile: int = 400) -> RealESRGANer:
    cfg = _BG_MODEL_CFGS[model_name]
    weight_path = WEIGHTS_DIR / cfg["weight"]
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Missing weight for '{model_name}': {cfg['weight']}. "
            "Run: make weights  (or make weights-general)"
        )
    # Community old-arch models (NMKD etc.) need the custom loader
    if model_name in _CUSTOM_ESRGAN_MODELS:
        return _CustomESRGANWrapper(weight_path, tile=tile)
    _normalize_esrgan_weight(weight_path)
    model = cfg["model_fn"]()
    model_path = str(weight_path)
    dni_weight = None
    if "wdn_weight" in cfg:
        wdn_path = WEIGHTS_DIR / cfg["wdn_weight"]
        if wdn_path.exists() and denoise_strength != 1.0:
            model_path = [str(weight_path), str(wdn_path)]
            dni_weight = [denoise_strength, 1 - denoise_strength]
    return RealESRGANer(
        scale=cfg["scale"],
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )


def get_bg_upsampler(model_name: str, denoise_strength: float = 0.5) -> RealESRGANer:
    """Return a cached upsampler, building it on first use per (model, denoise) pair."""
    cache_key = (model_name, denoise_strength if model_name == "realesr-general-x4v3" else None)
    if cache_key not in _bg_cache:
        _bg_cache[cache_key] = make_bg_upsampler(model_name, denoise_strength)
    return _bg_cache[cache_key]


# ── Model loader (called once at app startup) ─────────────────────────────────
def load_models(upscale: int = 2):
    _check_weights()
    bg_upsampler = make_bg_upsampler("RealESRGAN x4plus")
    face_restorer = GFPGANer(
        model_path=str(WEIGHTS_DIR / "GFPGANv1.4.pth"),
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
    )
    return face_restorer, bg_upsampler


# ── Anime upsampler (loaded lazily on first use) ───────────────────────────────
def load_anime_upsampler() -> RealESRGANer:
    anime_path = WEIGHTS_DIR / "RealESRGAN_x4plus_anime_6B.pth"
    if not anime_path.exists():
        raise FileNotFoundError(
            "Anime model not found. Run: make weights-anime"
        )
    return make_bg_upsampler("RealESRGAN anime 6B")


# ── CodeFormer lazy loader ────────────────────────────────────────────────────
_codeformer_inference_fn = None


def _setup_codeformer_symlinks():
    """Create CodeFormer/weights/* symlinks needed by module-level checks at import."""
    base = Path(__file__).parent
    mappings = {
        base / "CodeFormer" / "weights" / "CodeFormer" / "codeformer.pth": WEIGHTS_DIR / "codeformer.pth",
        base / "CodeFormer" / "weights" / "facelib" / "detection_Resnet50_Final.pth": WEIGHTS_DIR / "detection_Resnet50_Final.pth",
        base / "CodeFormer" / "weights" / "facelib" / "parsing_parsenet.pth": WEIGHTS_DIR / "parsing_parsenet.pth",
        base / "CodeFormer" / "weights" / "realesrgan" / "RealESRGAN_x2plus.pth": WEIGHTS_DIR / "RealESRGAN_x2plus.pth",
    }
    for link, target in mappings.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target.resolve())


def _patch_codeformer_load_file_from_url():
    """Monkey-patch all codeformer load_file_from_url references to serve from
    our local weights/ directory, avoiding any SSL / network calls."""
    from urllib.parse import urlparse as _urlparse

    _local = {
        "codeformer.pth":                  WEIGHTS_DIR / "codeformer.pth",
        "detection_Resnet50_Final.pth":    WEIGHTS_DIR / "detection_Resnet50_Final.pth",
        "parsing_parsenet.pth":            WEIGHTS_DIR / "parsing_parsenet.pth",
        "RealESRGAN_x2plus.pth":           WEIGHTS_DIR / "RealESRGAN_x2plus.pth",
    }

    import codeformer.basicsr.utils.download_util as _dl
    _orig = _dl.load_file_from_url

    def _patched(url, model_dir=None, progress=True, file_name=None):
        filename = file_name or os.path.basename(_urlparse(url).path)
        local_path = _local.get(filename)
        if local_path and local_path.exists():
            return str(local_path)
        return _orig(url, model_dir=model_dir, progress=progress, file_name=file_name)

    # Patch the source module and every module that captured a local reference
    import codeformer.facelib.utils.misc as _facelib_misc
    import codeformer.facelib.detection as _facelib_det
    import codeformer.facelib.parsing as _facelib_parse
    import codeformer.facelib.utils.face_restoration_helper as _frh
    import codeformer.basicsr.utils.realesrgan_utils as _realesr

    for _mod in (_dl, _facelib_misc, _facelib_det, _facelib_parse, _frh, _realesr):
        _mod.load_file_from_url = _patched


def load_codeformer():
    global _codeformer_inference_fn
    if _codeformer_inference_fn is not None:
        return _codeformer_inference_fn
    cf_pth = WEIGHTS_DIR / "codeformer.pth"
    realesrgan2_pth = WEIGHTS_DIR / "RealESRGAN_x2plus.pth"
    if not cf_pth.exists() or not realesrgan2_pth.exists():
        raise FileNotFoundError(
            "CodeFormer weights not found. Run: make weights"
        )
    _setup_codeformer_symlinks()
    from codeformer.app import inference_app  # noqa: PLC0415
    _patch_codeformer_load_file_from_url()  # patch after all submodule imports settle
    _codeformer_inference_fn = inference_app
    return _codeformer_inference_fn


# ── Fast face detection (OpenCV Haar cascade, no extra weights needed) ─────────
_haar_cascade = None


def detect_faces_quick(img_bgr) -> bool:
    global _haar_cascade
    if _haar_cascade is None:
        _haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces) > 0


def enhance_image(
    input_path: str,
    output_path: str,
    fidelity_weight: float = 0.5,
) -> np.ndarray:
    """
    Enhance a single image file.

    Args:
        input_path:       Path to source image.
        output_path:      Where to save the enhanced image.
        fidelity_weight:  0.0 = maximum enhancement, 1.0 = preserve original identity.
    Returns:
        Enhanced image as BGR numpy array.
    """
    face_restorer, _ = load_models()

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    _, _, restored_img = face_restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=fidelity_weight,
    )
    if restored_img is None:
        # No face detected — fall back to background upscale only
        bg_up = make_bg_upsampler("RealESRGAN x4plus")
        restored_img, _ = bg_up.enhance(img, outscale=2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, restored_img)
    print(f"Saved enhanced image → {output_path}")
    return restored_img


# ── Internal helpers ──────────────────────────────────────────────────────────
def _check_weights():
    required = [
        "GFPGANv1.4.pth",
        "RealESRGAN_x4plus.pth",
        "detection_Resnet50_Final.pth",
        "parsing_parsenet.pth",
    ]
    missing = [f for f in required if not (WEIGHTS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model weights: {missing}\n"
            f"Run:  bash download_weights.sh   (or: make weights)"
        )
