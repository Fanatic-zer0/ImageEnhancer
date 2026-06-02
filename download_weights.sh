#!/bin/bash
# download_weights.sh — Downloads all required model weights

set -e

WEIGHTS_DIR="weights"
GFPGAN_DIR="gfpgan/weights"
mkdir -p "$WEIGHTS_DIR" "$GFPGAN_DIR"

# Helper: download if not already present
download_if_missing() {
    local path="$1"
    local url="$2"
    local name
    name="$(basename "$path")"
    if [ -f "$path" ]; then
        echo "[SKIP] $name already exists"
    else
        echo "[DOWNLOAD] $name ..."
        curl -L --progress-bar -o "$path" "$url"
        echo "[OK] $name"
    fi
}

echo "Downloading model weights..."
echo ""

# ── GFPGAN v1.4 ───────────────────────────────────────────────────────────────
download_if_missing \
    "$WEIGHTS_DIR/GFPGANv1.4.pth" \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

# ── Real-ESRGAN x4plus (photo, default) ───────────────────────────────────────
download_if_missing \
    "$WEIGHTS_DIR/RealESRGAN_x4plus.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# ── Real-ESRGAN x2plus (faster / smaller output) ─────────────────────────────
download_if_missing \
    "$WEIGHTS_DIR/RealESRGAN_x2plus.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

# ── Real-ESRGAN x4plus anime 6B (Anime / Illustration mode) ──────────────────
download_if_missing \
    "$WEIGHTS_DIR/RealESRGAN_x4plus_anime_6B.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

# ── realesr-general-x4v3 + WDN companion (denoising upscaler) ─────────────────
download_if_missing \
    "$WEIGHTS_DIR/realesr-general-x4v3.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"

download_if_missing \
    "$WEIGHTS_DIR/realesr-general-wdn-x4v3.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"

# ── 4x-NMKD-Superscale-SP (sharpest community ESRGAN model) ──────────────────
download_if_missing \
    "$WEIGHTS_DIR/4x_NMKD-Superscale-SP_178000_G.pth" \
    "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth"

# ── 4x-UltraSharp (High Quality mode) ────────────────────────────────────────
download_if_missing \
    "$WEIGHTS_DIR/4x-UltraSharp.pth" \
    "https://huggingface.co/philz1337x/upscaler/resolve/011deacac8270114eb7d2eeff4fe6fa9a837be70/4x-UltraSharp.pth"

# ── CodeFormer ────────────────────────────────────────────────────────────────
download_if_missing \
    "$WEIGHTS_DIR/codeformer.pth" \
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

# ── Detection + Parsing models (facexlib — GFPGAN reads from gfpgan/weights/) ───────
download_if_missing \
    "$GFPGAN_DIR/detection_Resnet50_Final.pth" \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"

download_if_missing \
    "$GFPGAN_DIR/parsing_parsenet.pth" \
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

# ── AnimeGANv3 ONNX models (Anime Style mode) ────────────────────────────────
download_if_missing \
    "$WEIGHTS_DIR/AnimeGANv3_Hayao_36.onnx" \
    "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Hayao_36.onnx"

download_if_missing \
    "$WEIGHTS_DIR/AnimeGANv3_Shinkai_37.onnx" \
    "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Shinkai_37.onnx"

echo ""
echo "All weights downloaded."
echo "  $WEIGHTS_DIR/  → $(ls $WEIGHTS_DIR | wc -l | tr -d ' ') files"
echo "  $GFPGAN_DIR/   → $(ls $GFPGAN_DIR | wc -l | tr -d ' ') files"
