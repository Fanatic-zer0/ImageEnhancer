import os
import warnings
import datetime
import tempfile

# Suppress torchvision deprecation warnings from GFPGAN/facexlib internals
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from enhance import (detect_faces_quick, get_bg_upsampler, load_anime_upsampler,
                     load_codeformer, load_models, apply_animegan,
                     BG_UPSAMPLER_CHOICES, ANIMEGAN_STYLES)

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading models... (this may take a few seconds on first run)")
face_restorer, bg_upsampler_photo = load_models()
print("Models ready.")

_anime_upsampler = None  # lazy-loaded on first anime request
_codeformer_fn = None   # lazy-loaded on first CodeFormer request


def _get_anime_upsampler():
    global _anime_upsampler
    if _anime_upsampler is None:
        _anime_upsampler = load_anime_upsampler()
    return _anime_upsampler


def _get_codeformer():
    global _codeformer_fn
    if _codeformer_fn is None:
        _codeformer_fn = load_codeformer()
    return _codeformer_fn

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _make_comparison(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    label_h = 24
    scale = enhanced.height / original.height
    orig_display = original.resize(
        (int(original.width * scale), enhanced.height), Image.LANCZOS
    )
    gap = 3
    canvas = Image.new(
        "RGB",
        (orig_display.width + gap + enhanced.width, enhanced.height + label_h),
        (30, 30, 30),
    )
    canvas.paste(orig_display, (0, label_h))
    canvas.paste(enhanced, (orig_display.width + gap, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((orig_display.width // 2 - 22, 5), "BEFORE", fill=(180, 180, 180))
    draw.text(
        (orig_display.width + gap + enhanced.width // 2 - 22, 5),
        "AFTER",
        fill=(100, 220, 100),
    )
    return canvas


def process(image: Image.Image, fidelity_weight: float, mode: str,
            bg_model: str, upscale_str: str, denoise_strength: float,
            tile_size: int, animegan_style: str = "Hayao (Ghibli)",
            hq_prompt: str = "masterpiece, best quality, highres"):
    if image is None:
        raise gr.Error("Please upload an image first.")

    # ── High Quality pipeline (ESRGAN + SD 1.5 ControlNet Tile) ─────────────────
    if mode == "High Quality (slow)":
        try:
            from pipeline_enhancer import pipeline_enhance
        except ImportError:
            raise gr.Error("Pipeline not installed. Run: make pipeline (installs ~4 GB of models)")
        print("[HQ] Starting ESRGAN + SD1.5 pipeline (5–15 min on Apple Silicon)...")
        result_pil = pipeline_enhance(
            image.convert("RGB"),
            prompt=hq_prompt or "masterpiece, best quality, highres",
        )
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"enhanced_{ts}.png")
        result_pil.save(out_path)
        comparison = _make_comparison(image, result_pil)
        return result_pil, comparison, out_path

    upscale = int(upscale_str[0])  # "2x" → 2
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if mode == "Upscale Only":
        try:
            bg_up = get_bg_upsampler(bg_model, denoise_strength)
        except FileNotFoundError as e:
            raise gr.Error(str(e))
        bg_up.tile = tile_size
        output_bgr, _ = bg_up.enhance(img_bgr, outscale=upscale)
    elif mode == "Anime Style":
        try:
            styled_bgr = apply_animegan(img_bgr, animegan_style)
        except FileNotFoundError:
            raise gr.Error("AnimeGAN models not downloaded. Run: make weights")
        # upscale the styled result if the user requested >1x
        if upscale > 1:
            try:
                upsampler = _get_anime_upsampler()
            except FileNotFoundError:
                raise gr.Error("Anime model not downloaded. Run: make weights-anime")
            upsampler.tile = tile_size
            output_bgr, _ = upsampler.enhance(styled_bgr, outscale=upscale)
        else:
            output_bgr = styled_bgr
    elif mode == "Anime / Illustration":
        try:
            upsampler = _get_anime_upsampler()
        except FileNotFoundError:
            raise gr.Error("Anime model not downloaded. Run: make weights-anime")
        upsampler.tile = tile_size
        output_bgr, _ = upsampler.enhance(img_bgr, outscale=upscale)
    elif mode == "CodeFormer (sharper)":
        try:
            cf_inference = _get_codeformer()
        except FileNotFoundError:
            raise gr.Error("CodeFormer weights not downloaded. Run: make weights-codeformer")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_in = tmp.name
        cv2.imwrite(tmp_in, img_bgr)
        result_path = cf_inference(
            image=tmp_in,
            background_enhance=True,
            face_upsample=True,
            upscale=upscale,
            codeformer_fidelity=fidelity_weight,
        )
        os.unlink(tmp_in)
        output_bgr = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
    else:
        try:
            bg_up = get_bg_upsampler(bg_model, denoise_strength)
        except FileNotFoundError as e:
            raise gr.Error(str(e))
        bg_up.tile = tile_size
        # Auto: run fast Haar-cascade check first; Photo: always restore
        has_faces = (
            mode == "Photo (always restore)"
            or detect_faces_quick(img_bgr)
        )
        if has_faces:
            face_restorer.upscale = upscale          # sync with user selection
            face_restorer.bg_upsampler = bg_up
            _, _, output_bgr = face_restorer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=fidelity_weight,
            )
        else:
            # No faces detected — skip GFPGAN, upscale background only
            output_bgr, _ = bg_up.enhance(img_bgr, outscale=upscale)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"enhanced_{ts}.png")
    cv2.imwrite(out_path, output_bgr)

    result_pil = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
    comparison = _make_comparison(image, result_pil)
    return result_pil, comparison, out_path


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
/* ── Reset & Base ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #0a0d14 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #cbd5e1 !important;
}
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; padding: 0 1.25rem 3rem !important; }

/* ── Topbar ───────────────────────────────────────────── */
#topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 0; margin-bottom: 1.5rem;
    border-bottom: 1px solid #1c2536;
}
#topbar-brand { display: flex; align-items: center; gap: 0.65rem; }
.logo-mark {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 900; color: #fff;
    flex-shrink: 0;
}
#topbar-brand h1 {
    font-size: 1.1rem !important; font-weight: 700 !important;
    color: #f8fafc !important; margin: 0 !important; letter-spacing: -0.2px;
}
#topbar-brand .sub {
    font-size: 0.7rem; color: #475569; font-weight: 500;
    letter-spacing: 0.06em; text-transform: uppercase; display: block;
}
#topbar-pills { display: flex; gap: 0.4rem; }
.pill {
    padding: 0.2rem 0.65rem;
    background: rgba(14,165,233,0.07); border: 1px solid rgba(14,165,233,0.18);
    border-radius: 999px; font-size: 0.7rem; font-weight: 600;
    color: #38bdf8; letter-spacing: 0.04em;
}

/* ── Image Workspace ──────────────────────────────────── */
#workspace > .gr-row,
#workspace { gap: 1rem !important; }

.img-card {
    background: #0d1117 !important;
    border: 1px solid #1c2536 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    min-height: 460px;
    position: relative;
}

/* suppress Gradio's own border/bg on inner image component */
.img-card > div,
.img-card .wrap,
.img-card [data-testid="image"],
.img-card .image-container {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
}
.img-card img { object-fit: contain !important; border-radius: 0 !important; }

.img-badge {
    position: absolute; top: 12px; left: 12px; z-index: 20;
    padding: 3px 9px;
    background: rgba(10,13,20,0.75); backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 5px;
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #64748b; pointer-events: none;
}
.img-badge.out { color: #34d399; border-color: rgba(52,211,153,0.2); background: rgba(52,211,153,0.07); }

/* ── Section wrapper ──────────────────────────────────── */
.section {
    background: #0d1117;
    border: 1px solid #1c2536;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
}
.section-title {
    font-size: 0.62rem !important; font-weight: 700 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    color: #334155 !important; margin: 0 0 0.75rem !important;
    display: block;
}

/* ── Mode selector — pill buttons, hide radio dots ───── */
#mode-row .wrap { display: flex !important; flex-wrap: wrap !important; gap: 0.45rem !important; }

/* hide the actual radio circle (input is INSIDE label in Gradio) */
#mode-row input[type="radio"] { display: none !important; }

#mode-row label {
    display: inline-flex !important; align-items: center !important;
    background: #111827 !important;
    border: 1px solid #1c2536 !important;
    border-radius: 8px !important;
    padding: 0.45rem 0.85rem !important;
    font-size: 0.83rem !important; font-weight: 500 !important;
    color: #64748b !important;
    cursor: pointer !important;
    transition: border-color 0.15s, background 0.15s, color 0.15s !important;
    user-select: none !important;
}
#mode-row label:hover {
    border-color: #0ea5e9 !important;
    background: #0c1e2c !important;
    color: #e2e8f0 !important;
}
/* :has() works when input is nested inside label */
#mode-row label:has(input[type="radio"]:checked) {
    border-color: #0ea5e9 !important;
    background: #0c2233 !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
}

/* ── Controls row ─────────────────────────────────────── */
#controls-row { display: flex; align-items: flex-end; gap: 1.25rem; flex-wrap: wrap; }
#controls-row > div { flex: 1; min-width: 180px; }
#controls-row .btn-col { flex: 0 0 auto; }

/* ── Enhance button ───────────────────────────────────── */
#enhance-btn {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    border: none !important; border-radius: 10px !important;
    font-size: 0.9rem !important; font-weight: 700 !important;
    letter-spacing: 0.05em !important; padding: 0.75rem 2rem !important;
    color: #fff !important; cursor: pointer !important;
    box-shadow: 0 2px 20px rgba(14,165,233,0.25) !important;
    transition: opacity 0.15s, transform 0.15s !important;
    white-space: nowrap !important; width: 100% !important;
}
#enhance-btn:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

/* ── Scale radio — also hide dots ────────────────────── */
#scale-row input[type="radio"] { display: none !important; }
#scale-row .wrap { display: flex !important; gap: 0.4rem !important; }
#scale-row label {
    display: inline-flex !important; align-items: center !important;
    background: #111827 !important; border: 1px solid #1c2536 !important;
    border-radius: 7px !important; padding: 0.35rem 0.9rem !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: #64748b !important; cursor: pointer !important;
    transition: border-color 0.15s, background 0.15s, color 0.15s !important;
}
#scale-row label:hover { border-color: #0ea5e9 !important; color: #e2e8f0 !important; }
#scale-row label:has(input[type="radio"]:checked) {
    border-color: #0ea5e9 !important; background: #0c2233 !important; color: #38bdf8 !important; font-weight: 600 !important;
}

/* ── Accordion ────────────────────────────────────────── */
.gr-accordion { background: transparent !important; border: none !important; }
.gr-accordion > .label-wrap,
.gr-accordion summary {
    font-size: 0.78rem !important; font-weight: 600 !important;
    color: #475569 !important; padding: 0.6rem 0 !important;
    letter-spacing: 0.03em !important;
}
.gr-accordion > .label-wrap:hover,
.gr-accordion summary:hover { color: #94a3b8 !important; }
.adv-row { gap: 1.5rem !important; margin-bottom: 0.75rem !important; }
.adv-row:last-child { margin-bottom: 0 !important; }

/* ── Sliders ──────────────────────────────────────────── */
input[type=range] { accent-color: #0ea5e9 !important; }
input[type=range]::-webkit-slider-runnable-track { background: #1e293b !important; height: 3px !important; }
input[type=range]::-webkit-slider-thumb { background: #0ea5e9 !important; }

/* ── Dropdowns / inputs ───────────────────────────────── */
select, input[type=text], textarea, .gr-box {
    background: #111827 !important; border: 1px solid #1c2536 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}
label > span, .gr-form label span { color: #475569 !important; font-size: 0.78rem !important; }

/* ── Tabs (output) ────────────────────────────────────── */
.tab-nav { border-bottom: 1px solid #1c2536 !important; margin-bottom: 0 !important; }
.tab-nav button {
    background: transparent !important; color: #334155 !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; font-size: 0.8rem !important;
    font-weight: 600 !important; padding: 0.5rem 1rem !important;
    transition: color 0.15s !important;
}
.tab-nav button.selected { color: #38bdf8 !important; border-bottom-color: #0ea5e9 !important; }

/* ── Download strip ───────────────────────────────────── */
#dl-strip {
    display: flex; align-items: center; gap: 1rem;
    background: #0d1117; border: 1px solid #1c2536;
    border-radius: 14px; padding: 0.75rem 1.25rem;
}
#dl-strip .dl-label { font-size: 0.72rem; color: #334155; font-weight: 600; white-space: nowrap; }

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0d14; }
::-webkit-scrollbar-thumb { background: #1c2536; border-radius: 3px; }
"""

with gr.Blocks(title="Enhancer · AI Image Upscaling") as demo:

    # ── Topbar ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="topbar">
        <div id="topbar-brand">
            <div class="logo-mark">✦</div>
            <div>
                <h1>Enhancer</h1>
                <span class="sub">AI Image Enhancement</span>
            </div>
        </div>
        <div id="topbar-pills">
            <span class="pill">On-device</span>
            <span class="pill">Apple Silicon</span>
            <span class="pill">MPS</span>
        </div>
    </div>
    """)

    # ── Image Workspace ───────────────────────────────────────────────────────
    with gr.Row(elem_id="workspace"):
        with gr.Column(elem_classes=["img-card"]):
            gr.HTML('<div class="img-badge">Input</div>')
            input_img = gr.Image(type="pil", show_label=False, height=460)
        with gr.Column(elem_classes=["img-card"]):
            gr.HTML('<div class="img-badge out">Output</div>')
            with gr.Tabs():
                with gr.TabItem("Enhanced"):
                    output_img = gr.Image(type="pil", show_label=False, height=420, interactive=False)
                with gr.TabItem("Before / After"):
                    comparison_img = gr.Image(type="pil", show_label=False, height=420, interactive=False)

    # ── Mode Selector ─────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(elem_classes=["section"]):
            gr.HTML('<span class="section-title">Processing Mode</span>')
            mode_radio = gr.Radio(
                choices=[
                    "🔍 Auto (detect faces)",
                    "🧑 Photo (always restore)",
                    "⚡ CodeFormer (sharper)",
                    "🖼️ Upscale Only",
                    "🎨 Anime / Illustration",
                    "🎌 Anime Style",
                    "🔬 High Quality (slow)",
                ],
                value="🔍 Auto (detect faces)",
                show_label=False,
                elem_id="mode-row",
            )

    # ── Controls ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(elem_classes=["section"]):
            gr.HTML('<div id="controls-row">')
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML('<span class="section-title">Face Fidelity</span>')
                    fidelity = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="Identity Fidelity",
                        info="0 = max AI enhancement  ·  1 = preserve face identity",
                    )
                with gr.Column(scale=2):
                    gr.HTML('<span class="section-title">Output Scale</span>')
                    upscale_radio = gr.Radio(
                        choices=["1x", "2x", "4x"], value="2x",
                        show_label=False, elem_id="scale-row",
                    )
                with gr.Column(scale=1, min_width=160, elem_classes=["btn-col"]):
                    gr.HTML('<span class="section-title">&nbsp;</span>')
                    btn = gr.Button("✦  Enhance", elem_id="enhance-btn", size="lg")
            gr.HTML('</div>')

    # ── Advanced ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(elem_classes=["section"]):
            with gr.Accordion("⚙️  Advanced Settings", open=False):
                with gr.Row(elem_classes=["adv-row"]):
                    bg_model_dd = gr.Dropdown(
                        choices=BG_UPSAMPLER_CHOICES, value="RealESRGAN x4plus",
                        label="Background Model",
                        info="x4plus = balanced · x2plus = faster · general-x4v3 = denoising · NMKD = sharpest",
                        scale=2,
                    )
                    denoise_sl = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="Denoising Strength", info="Only for realesr-general-x4v3",
                        scale=2,
                    )
                    tile_sl = gr.Slider(
                        minimum=0, maximum=512, value=400, step=128,
                        label="Tile Size", info="0 = full image · lower = less RAM",
                        scale=1,
                    )
                with gr.Row(elem_classes=["adv-row"]):
                    animegan_style_dd = gr.Dropdown(
                        choices=ANIMEGAN_STYLES, value="Hayao (Ghibli)",
                        label="Anime Style", info="Only for 🎌 Anime Style mode",
                        scale=1,
                    )
                    hq_prompt_tb = gr.Textbox(
                        value="masterpiece, best quality, highres",
                        label="HQ Prompt", info="Only for 🔬 High Quality mode", lines=1,
                        scale=3,
                    )

    # ── Download ──────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(elem_id="dl-strip"):
            gr.HTML('<span class="dl-label">📁 Save Result</span>')
            download_btn = gr.File(show_label=False)

    # ── Wiring ────────────────────────────────────────────────────────────────
    def _strip_emoji(mode: str) -> str:
        return mode.split(" ", 1)[1] if " " in mode else mode

    def process_ui(image, fidelity_weight, mode, bg_model,
                   upscale_str, denoise_strength, tile_size, animegan_style, hq_prompt):
        return process(image, fidelity_weight, _strip_emoji(mode),
                       bg_model, upscale_str, denoise_strength, tile_size,
                       animegan_style, hq_prompt)

    btn.click(
        fn=process_ui,
        inputs=[input_img, fidelity, mode_radio, bg_model_dd,
                upscale_radio, denoise_sl, tile_sl, animegan_style_dd, hq_prompt_tb],
        outputs=[output_img, comparison_img, download_btn],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
            theme=gr.themes.Base(), css=CSS, show_error=True)
