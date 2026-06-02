# Enhancer

A fully local, on-device AI image enhancement tool. Restore faces, upscale backgrounds, and sharpen details — no cloud, no API keys, no data leaves your machine.

Runs on **macOS (Apple Silicon or Intel)** and **Linux** via PyTorch. Apple Silicon users get hardware acceleration through Metal (MPS); all other hardware falls back to CPU automatically.

---

## Features

- **4 enhancement modes** — Auto, Photo, CodeFormer, Anime/Illustration
- **Face restoration** — GFPGAN v1.4 with adjustable identity-fidelity slider
- **Background upscaling** — 3 Real-ESRGAN variants (x4plus, x2plus, general-x4v3)
- **CodeFormer** — transformer-based restoration for heavily degraded faces
- **Anime mode** — Real-ESRGAN 6B tuned for illustrations and cartoons
- **Before/After comparison** panel in the output view
- **Timestamped outputs** saved to `outputs/`
- Dark, modern Gradio UI with tabbed output panel

---

## Architecture

```
Input image (PIL)
       │
       ▼
 detect_faces_quick()          ← OpenCV Haar cascade (fast, no extra weights)
       │
  ┌────┴─────────────────────────────────────────┐
  │  Mode routing (app.py → enhance.py)           │
  │                                               │
  │  Auto        → GFPGAN (if faces) or bg-only  │
  │  Photo       → GFPGAN always                 │
  │  CodeFormer  → codeformer-pip inference       │
  │  Anime       → RealESRGAN 6B (no face model) │
  └───────────────────────────────────────────────┘
       │
       ▼
 Background upscaling (Real-ESRGAN, lazy-cached per model)
       │
       ▼
 Output PNG  +  Before/After composite
```

**Model loading strategy:** GFPGAN and the default background upscaler are loaded once at startup. Anime, CodeFormer, and alternative upscaler models are lazy-loaded on first use and cached in memory for subsequent calls.

---

## Tech Stack

| Component | Library |
|---|---|
| Face restoration | GFPGAN v1.4 |
| Transformer restoration | CodeFormer (`codeformer-pip`) |
| Background upscaling | Real-ESRGAN (x4plus · x2plus · general-x4v3) |
| Face detection | OpenCV Haar cascade |
| UI | Gradio 4.x (`gr.Blocks`) |
| Runtime — Apple Silicon | PyTorch MPS (Metal) |
| Runtime — Intel Mac / Linux | PyTorch CPU |

---

## Requirements

- Python 3.10 – 3.12
- macOS 12+ (Apple Silicon or Intel) **or** Linux (Ubuntu 20.04+)
- ~2 GB disk space for all model weights
- 8 GB RAM minimum (16 GB recommended for 4x upscaling)

> **CUDA (NVIDIA GPU):** PyTorch will automatically use CUDA if available on Linux. No code changes needed.

---

## Quick Start

```bash
# 1. Create virtualenv and install all dependencies
make setup

# 2. Download all model weights (~1.5 GB total)
make weights

# 3. Launch the web UI
make run
# → Open http://localhost:7860
```

### Linux notes

```bash
# Install system dependencies first (if needed)
sudo apt install python3-venv python3-dev libgl1

# Then the same three commands above work identically
make setup && make weights && make run
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
├── app.py                  # Gradio web UI + process() orchestration
├── enhance.py              # All model loading, inference, face detection
├── download_weights.sh     # Downloads all 9 model weight files
├── requirements.txt        # Python dependencies
├── Makefile                # setup / weights / run / enhance / clean
├── weights/                # Model weights (created by make weights)
│   ├── GFPGANv1.4.pth                    # Face restoration
│   ├── codeformer.pth                    # CodeFormer face restoration
│   ├── RealESRGAN_x4plus.pth             # Default background upscaler
│   ├── RealESRGAN_x2plus.pth             # Faster / smaller output
│   ├── RealESRGAN_x4plus_anime_6B.pth    # Anime / illustration mode
│   ├── realesr-general-x4v3.pth          # General upscaler w/ denoising
│   ├── realesr-general-wdn-x4v3.pth      # WDN companion for above
│   ├── detection_Resnet50_Final.pth      # Face detection (facexlib)
│   └── parsing_parsenet.pth              # Face parsing (facexlib)
└── outputs/                # Enhanced images saved here (auto-created)
```

---

## Enhancement Modes

| Mode | Best for | Face model | Background model |
|---|---|---|---|
| **Auto** | Unknown content — detects faces automatically | GFPGAN (if faces found) | RealESRGAN x4plus |
| **Photo** | Portraits — always restores faces | GFPGAN | RealESRGAN x4plus |
| **CodeFormer** | Heavily degraded / old photos | CodeFormer | RealESRGAN x2plus |
| **Anime** | Illustrations, cartoons, manga | — | RealESRGAN anime 6B |

---

## Controls

### Identity Fidelity (face modes only)

| Value | Effect |
|---|---|
| `0.0` | Maximum AI enhancement — reconstructs fine detail freely |
| `0.5` | Balanced — recommended default |
| `1.0` | Preserve original face identity — lighter correction only |

### Advanced Controls

| Control | Description |
|---|---|
| **Output Scale** | 1x / 2x / 4x output resolution multiplier |
| **Background Model** | Switch between the three Real-ESRGAN variants |
| **Denoising Strength** | Active only for `realesr-general-x4v3`; higher = more noise removal |
| **Tile Size** | Lower values reduce peak RAM usage; `0` = process full image at once |

---

## Troubleshooting

**Apple Silicon — MPS errors:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py
# (already set automatically by make run)
```

**Out of memory on large images:**  
Reduce the tile size slider in the UI (Advanced → Tile Size → 256 or lower).

**Missing weights:**
```bash
make weights   # re-runs download_weights.sh, skips files already present
```

**`basicsr` / `torchvision` import error on Python 3.12:**
```bash
# Patch the installed basicsr package:
sed -i '' 's/from torchvision.transforms.functional_tensor/from torchvision.transforms.functional/' \
  venv/lib/python3.*/site-packages/basicsr/data/degradations.py
```

**CodeFormer downloads weights on first run:**  
This is prevented by symlinks created automatically in `enhance.py`. If you see download attempts, ensure `make weights` has been run first.
