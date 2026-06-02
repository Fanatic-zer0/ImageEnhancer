# ImageEnhancer

A fully local, on-device AI image enhancement tool. Restore faces, upscale backgrounds, and apply anime style transfer — no cloud, no API keys, no data leaves your machine.

Runs on **macOS (Apple Silicon or Intel)** and **Linux** via PyTorch. Apple Silicon users get hardware acceleration through Metal (MPS); all other hardware falls back to CPU automatically.

---

## Features

- **7 enhancement modes** — Auto, Photo, CodeFormer, Upscale Only, Anime / Illustration, Anime Style, High Quality
- **Face restoration** — GFPGAN v1.4 with adjustable identity-fidelity slider
- **Background upscaling** — 4 model options (RealESRGAN x4plus · x2plus · general-x4v3 · NMKD-Superscale-SP)
- **CodeFormer** — transformer-based restoration for heavily degraded faces
- **Anime style transfer** — AnimeGANv3 (Hayao / Shinkai styles) with optional upscale pass
- **High Quality mode** — 4x-UltraSharp ESRGAN upscaler
- **Before/After comparison** panel in the output view
- **Timestamped outputs** saved to `outputs/`
- Dark Gradio UI with pill-style mode selector and tabbed output panel

---

Screenshots:

<img width="1488" height="821" alt="image" src="https://github.com/user-attachments/assets/c3ecd5d4-b8dc-4c17-b306-b15c8d0ccc28" />


---

## Architecture

```
Input image (PIL)
       │
       ▼
 detect_faces_quick()          ← OpenCV Haar cascade (fast, no extra weights)
       │
  ┌────┴──────────────────────────────────────────────────┐
  │  Mode routing (app.py → enhance.py)                    │
  │                                                        │
  │  Auto          → GFPGAN (if faces) or bg-only         │
  │  Photo         → GFPGAN always                        │
  │  CodeFormer    → codeformer-pip inference              │
  │  Upscale Only  → background upscaler, no face model   │
  │  Anime         → RealESRGAN anime 6B                  │
  │  Anime Style   → AnimeGANv3 → optional upscale pass   │
  │  High Quality  → 4x-UltraSharp ESRGAN                 │
  └────────────────────────────────────────────────────────┘
       │
       ▼
 Output PNG  +  Before/After composite
```

**Model loading strategy:** GFPGAN and the default background upscaler are loaded once at startup. All other models (CodeFormer, AnimeGAN, NMKD, UltraSharp) are lazy-loaded on first use and cached in memory.

---

## Tech Stack

| Component | Library |
|---|---|
| Face restoration | GFPGAN v1.4 |
| Transformer restoration | CodeFormer (`codeformer-pip`) |
| Background upscaling | Real-ESRGAN (x4plus · x2plus · general-x4v3) |
| Community upscaler | NMKD-Superscale-SP (old-arch ESRGAN via `esrgan_model.py`) |
| High Quality upscaler | 4x-UltraSharp (old-arch ESRGAN) |
| Anime style transfer | AnimeGANv3 ONNX (Hayao · Shinkai) |
| Face detection | OpenCV Haar cascade |
| UI | Gradio 6.x (`gr.Blocks`) |
| Runtime — Apple Silicon | PyTorch MPS (Metal) |
| Runtime — Intel Mac / Linux | PyTorch CPU |

---

## Requirements

- Python 3.10 – 3.12
- macOS 12+ (Apple Silicon or Intel) **or** Linux (Ubuntu 20.04+)
- ~3 GB disk space for all model weights
- 8 GB RAM minimum (16 GB recommended for 4x upscaling)

> **CUDA (NVIDIA GPU):** PyTorch will automatically use CUDA if available on Linux. No code changes needed.

---

## Quick Start

```bash
# 1. Create virtualenv, install dependencies, and download all weights
make setup

# 2. Launch the web UI
make run
# → Open http://localhost:7860
```

`make setup` now runs `make weights` automatically — a single command gets you from zero to ready.

### Linux notes

```bash
# Install system dependencies first (if needed)
sudo apt install python3-venv python3-dev libgl1

make setup && make run
```

---

## CLI Usage

Enhance a single image without the UI:

```bash
make enhance IN=photo.jpg OUT=photo_enhanced.jpg
```

---

## Project Structure

```
Enhancer/
├── app.py                   # Gradio web UI + process() orchestration
├── enhance.py               # All model loading, inference, face detection
├── esrgan_model.py          # Old-arch ESRGAN loader (NMKD, UltraSharp)
├── pipeline_enhancer.py     # High Quality mode (4x-UltraSharp ESRGAN)
├── download_weights.sh      # Downloads all model weight files
├── requirements.txt         # Python dependencies
├── Makefile                 # setup / weights / run / enhance / clean
├── weights/                 # Model weights (created by make weights)
│   ├── GFPGANv1.4.pth                       # Face restoration
│   ├── codeformer.pth                        # CodeFormer face restoration
│   ├── RealESRGAN_x4plus.pth                 # Default background upscaler
│   ├── RealESRGAN_x2plus.pth                 # Faster / smaller output
│   ├── RealESRGAN_x4plus_anime_6B.pth        # Anime / illustration mode
│   ├── realesr-general-x4v3.pth              # General upscaler w/ denoising
│   ├── realesr-general-wdn-x4v3.pth          # WDN companion for above
│   ├── 4x_NMKD-Superscale-SP_178000_G.pth   # Sharpest community upscaler
│   ├── 4x-UltraSharp.pth                     # High Quality mode upscaler
│   ├── AnimeGANv3_Hayao_36.onnx              # Anime style — Ghibli
│   └── AnimeGANv3_Shinkai_37.onnx            # Anime style — Shinkai
├── gfpgan/weights/          # Facelib models (GFPGAN / CodeFormer)
│   ├── detection_Resnet50_Final.pth          # Face detection
│   └── parsing_parsenet.pth                  # Face parsing
└── outputs/                 # Enhanced images saved here (auto-created)
```

---

## Enhancement Modes

| Mode | Best for | Face model | Background model |
|---|---|---|---|
| **🔍 Auto** | Unknown content — detects faces automatically | GFPGAN (if faces found) | RealESRGAN x4plus |
| **🧑 Photo** | Portraits — always restores faces | GFPGAN | RealESRGAN x4plus |
| **⚡ CodeFormer** | Heavily degraded / old photos | CodeFormer | — |
| **🖼️ Upscale Only** | Pure resolution upscale, no face work | — | Selectable |
| **🎨 Anime / Illustration** | Illustrations, cartoons, manga | — | RealESRGAN anime 6B |
| **🎌 Anime Style** | Convert photo to anime art style | — | AnimeGANv3 → optional upscale |
| **🔬 High Quality** | Best quality, slower | — | 4x-UltraSharp ESRGAN |

---

## Background Models

| Model | Speed | Best for |
|---|---|---|
| **RealESRGAN x4plus** | Fast | General photos (default) |
| **RealESRGAN x2plus** | Fastest | 2x output, lower RAM |
| **realesr-general-x4v3** | Medium | Noisy / compressed images |
| **4x-NMKD-Superscale-SP** | Medium | Maximum sharpness |

---

## Controls

### Identity Fidelity (face modes only)

| Value | Effect |
|---|---|
| `0.0` | Maximum AI enhancement — reconstructs fine detail freely |
| `0.5` | Balanced — recommended default |
| `1.0` | Preserve original face identity — lighter correction only |

### Advanced Settings

| Control | Description |
|---|---|
| **Output Scale** | 1x / 2x / 4x output resolution multiplier |
| **Background Model** | Switch between the four upscaler variants |
| **Denoising Strength** | Active only for `realesr-general-x4v3`; higher = more noise removal |
| **Tile Size** | Lower values reduce peak RAM usage; `0` = process full image at once |
| **Anime Style** | Hayao (Ghibli look) or Shinkai — only in 🎌 Anime Style mode |
| **HQ Prompt** | Prompt text for 🔬 High Quality mode |

---

## Makefile Targets

| Target | Description |
|---|---|
| `make setup` | Create venv, install deps, download all weights |
| `make weights` | Download / update all model weights only |
| `make run` | Launch the Gradio UI at http://localhost:7860 |
| `make enhance IN=… OUT=…` | Enhance a single image from the CLI |
| `make pipeline` | Install SD pipeline Python deps (diffusers, accelerate, transformers) |
| `make clean` | Remove venv and pycache |
| `make clean-weights` | Remove the weights directory |

---

## Troubleshooting

**Apple Silicon — MPS errors:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py
# (already set automatically by make run)
```

**Out of memory on large images:**
Reduce the tile size in Advanced Settings (256 or lower).

**Missing weights:**
```bash
make weights   # re-runs download_weights.sh, skips files already present
```
