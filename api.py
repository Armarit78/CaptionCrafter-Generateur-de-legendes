from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os
from typing import Dict, Any

from image_utils import extract_text, describe_visual
from filters import apply_filter, add_text
from mistral_api import (
    generate_instagram_captions,
    detect_visual_entities_from_np,
    describe_landmark_from_np,
    agent_generate_image,
)

app = FastAPI(title="CaptionCrafter API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Utils ----------
def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

async def _load_np_from_upload(file: UploadFile) -> np.ndarray:
    """Charge une image UploadFile (async) -> np.ndarray BGR."""
    content = await file.read()
    try:
        pil = Image.open(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=422, detail="Fichier image invalide")
    return _pil_to_bgr(pil)

def _assert_mistral_key():
    """Vérifie que la clé Mistral est bien présente dans l'env du process."""
    key = os.getenv("MISTRAL_API_KEY")
    if not key or not key.strip():
        raise HTTPException(
            status_code=503,
            detail="MISTRAL_API_KEY manquante côté serveur."
        )

def _infer_orientation(img_bgr: np.ndarray) -> str:
    """Retourne 'landscape' | 'portrait' | 'square'."""
    h, w = img_bgr.shape[:2]
    if w > h:
        return "landscape"
    if h > w:
        return "portrait"
    return "square"

def _mean_std_luma(img_bgr: np.ndarray) -> Dict[str, float]:
    """Calcule une luminance moyenne (brightness) et un écart-type (contrast)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    # Normalisation simple 0..100 pour lecture humaine
    brightness = round(mean / 255.0 * 100.0, 2)
    contrast = round(std / 128.0 * 100.0, 2)  # 128 ~ std max approximative
    return {"brightness": brightness, "contrast": contrast}

def _dominant_colors(img_bgr: np.ndarray, k: int = 3) -> Any:
    """
    Renvoie une petite liste de couleurs dominantes en RGB (0..255).
    Utilise k-means OpenCV (léger, robuste).
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    # Critères KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(
        data=img,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=1,
        flags=cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.uint8)
    # Ordonne par fréquence
    hist = np.bincount(labels.flatten(), minlength=k).astype(float)
    order = np.argsort(hist)[::-1]
    colors = [centers[i].tolist() for i in order]
    return colors

# ---------- Routes basiques ----------
@app.get("/")
async def root():
    return {
        "message": "✅ CaptionCrafter API v3 is running",
        "docs": "/docs",
        "health": "/health",
        "env": "/env",
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/env")
async def env_check():
    """Diagnostic rapide (sans exposer la clé)."""
    return {
        "MISTRAL_API_KEY": "set" if bool(os.getenv("MISTRAL_API_KEY")) else "missing"
    }

# ---------- Analyse visuelle & OCR ----------
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Retourne un résumé visuel + OCR du texte détecté dans l'image.
    """
    np_img = await _load_np_from_upload(file)
    visual = describe_visual(np_img)      # idéalement: dict décrivant la scène
    ocr_text = extract_text(np_img)       # texte OCR
    return JSONResponse({"visual": visual, "ocr_text": ocr_text})

# ---------- Captions Instagram ----------
@app.post("/captions")
async def captions(
    file: UploadFile = File(...),
    style: str = Form("Inspiration"),
    overlay_text: str = Form(""),
):
    """
    Génère 3 légendes Instagram à partir de l'image + style + overlay_text éventuel.
    ⚠️ Requiert MISTRAL_API_KEY dans l'env serveur.
    Construit un contexte explicite et stable pour le modèle (dict complet).
    """
    _assert_mistral_key()
    np_img = await _load_np_from_upload(file)

    # 1) Base: description visuelle
    vis = describe_visual(np_img)  # peut être un dict ou un texte selon image_utils

    # 2) Normalisation + compléments attendus (orientation, brightness, contrast, colors)
    ctx: Dict[str, Any] = {}

    # Orientation / Brightness / Contrast / Colors (calcul local pour robustesse)
    ctx["orientation"] = _infer_orientation(np_img)
    bc = _mean_std_luma(np_img)
    ctx["brightness"] = bc["brightness"]
    ctx["contrast"] = bc["contrast"]
    ctx["colors"] = _dominant_colors(np_img, k=3)  # liste de 3 couleurs RGB

    # 3) Bloc visuel complet (on garde tout ce que décrit image_utils)
    #    - si `vis` est une string, on l'encapsule pour éviter l'ambiguïté
    if isinstance(vis, dict):
        ctx["visual"] = vis
    else:
        ctx["visual"] = {"scene": str(vis)}

    # 4) OCR + style + overlay
    ctx["ocr_text"] = extract_text(np_img)
    ctx["style"] = style
    ctx["overlay_text"] = overlay_text

    # 5) Appel modèle — l'API attend précisément un dict "ctx"
    captions = generate_instagram_captions(ctx)
    return JSONResponse({"captions": captions})

# ---------- Filtres image (PNG) ----------
@app.post("/filter")
async def filter_image(
    file: UploadFile = File(...),
    filter_type: str = Form(...),
    text: str = Form(""),
    pos_x: int = Form(50),
    pos_y: int = Form(50),
):
    """
    Applique un filtre simple et, optionnellement, un texte incrusté.
    Renvoie un PNG en binaire.
    """
    np_img = await _load_np_from_upload(file)
    out = apply_filter(np_img, filter_type=filter_type)
    if text:
        out = add_text(out, text, (pos_x, pos_y))

    # Encode en PNG
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(status_code=500, detail="Échec d'encodage PNG")
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/png")

# ---------- Agent image (génération) ----------
@app.post("/agent-image")
async def agent_image(
    prompt: str = Form(...),
    size: str = Form("1024x1024"),
):
    """
    Génère une image PNG via l'Agent Mistral (text-to-image).
    ⚠️ Requiert MISTRAL_API_KEY dans l'env serveur.
    """
    _assert_mistral_key()
    try:
        png_bytes = agent_generate_image(prompt=prompt, size=size)
        return StreamingResponse(BytesIO(png_bytes), media_type="image/png")
    except RuntimeError as e:
        # Surface un vrai 429 au client s'il s'agit d'un rate limit
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            import re
            headers = {}
            m = re.search(r"Retry-After=([0-9\.]+)", msg)
            if m:
                headers["Retry-After"] = m.group(1)
            raise HTTPException(status_code=429, detail="image_generation rate limit reached", headers=headers)
        # défaut: on remonte un 502 (upstream)
        raise HTTPException(status_code=502, detail=f"Upstream error: {msg}")

# ---------- Landmark (description du monument/élément principal) ----------
@app.post("/landmark")
async def describe_landmark(file: UploadFile = File(...)):
    """
    Retourne une description concise du monument/élément principal.
    ⚠️ Requiert MISTRAL_API_KEY dans l'env serveur.
    """
    _assert_mistral_key()
    np_img = await _load_np_from_upload(file)
    desc = describe_landmark_from_np(np_img)
    return JSONResponse({"landmark_description": desc})

# ---------- Scène & Entités ----------
@app.post("/entities")
async def analyze_entities(file: UploadFile = File(...)):
    """
    Retourne la scène, les entités et les actions détectées via Mistral.
    ⚠️ Requiert MISTRAL_API_KEY dans l'env serveur.
    """
    _assert_mistral_key()
    np_img = await _load_np_from_upload(file)
    entities = detect_visual_entities_from_np(np_img)
    # Normalise la forme de réponse
    return JSONResponse({
        "scene": entities.get("scene", ""),
        "entities": entities.get("entities", []),
        "actions": entities.get("actions", []),
        "raw": entities
    })
