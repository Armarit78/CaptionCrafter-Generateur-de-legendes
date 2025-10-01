import io
import cv2
import hashlib
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import urllib.parse

# Utils projet
from image_utils import describe_visual, extract_text
from filters import apply_filter, add_text
from mistral_api import (
    generate_instagram_captions,
    detect_visual_entities_from_np,
    describe_landmark_from_np,
    agent_generate_image
)

# --------------------------- Config Streamlit -----------------------------------
load_dotenv()
st.set_page_config(page_title="CaptionCrafter", page_icon="🖼️", layout="wide")

# --------------------------- Helpers internes -----------------------------------
def _init_once(k, v):
    """Initialise une variable de session si absente."""
    if k not in st.session_state:
        st.session_state[k] = v

def _bgr_to_pil(np_bgr: np.ndarray) -> Image.Image:
    """Convertit un tableau BGR en Image PIL (RGB)."""
    return Image.fromarray(cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB))

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convertit une image PIL en numpy BGR."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _ensure_3ch(np_img: np.ndarray) -> np.ndarray:
    """Force 3 canaux pour l’image."""
    if np_img is None:
        return np_img
    if len(np_img.shape) == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    return np_img

def _load_np_from_bytes(b: bytes) -> np.ndarray:
    """Charge une image (numpy BGR) depuis des bytes (png/jpg)."""
    arr = np.frombuffer(b, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        im = _pil_to_bgr(Image.open(io.BytesIO(b)))
    return _ensure_3ch(im)

def _hex_to_bgr(hx: str):
    """Convertit couleur hex en tuple BGR."""
    hx = hx.lstrip("#")
    r = int(hx[0:2], 16)
    g = int(hx[2:4], 16)
    b = int(hx[4:6], 16)
    return (b, g, r)

def _get_np_for_ai() -> np.ndarray:
    """Retourne l’image traitée ou brute prête pour IA."""
    arr = st.session_state.get("processed_np", None)
    if isinstance(arr, np.ndarray) and arr.size > 0:
        return arr
    b = st.session_state.get("image_bytes", None)
    if isinstance(b, (bytes, bytearray)):
        return _load_np_from_bytes(b)
    return None

def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def _data_url_txt(content: str, fname: str) -> str:
    """Lien HTML de téléchargement direct d’un texte (pas de rerun)."""
    quoted = urllib.parse.quote(content or "")
    return f'<a download="{fname}" href="data:text/plain;charset=utf-8,{quoted}" target="_blank" rel="noopener"><button style="padding:8px 12px;border:1px solid #e5e7eb;border-radius:8px;background:#fff;cursor:pointer;">📄 {fname}</button></a>'

def _intent_anchor(caption: str, label: str) -> str:
    """Bouton-lien qui ouvre 𝕏 dans un nouvel onglet avec la légende."""
    url = "https://twitter.com/intent/tweet?text=" + urllib.parse.quote(caption or "")
    return (
        f'<a href="{url}" target="_blank" rel="noopener">'
        f'<button style="padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#fff;cursor:pointer;">🕊️ {label}</button></a>'
    )

# --------------------------- Initialisation session -----------------------------
_init_once("image_bytes", None)
_init_once("image_name", None)
_init_once("image_hash", None)
_init_once("processed_np", None)

_init_once("custom_text", "")
_init_once("color_hex", "#ffffff")
_init_once("font_scale", 1.0)
_init_once("thickness", 2)
_init_once("font_face", "Sans (Simplex)")
_init_once("pos_x", 50)
_init_once("pos_y", 50)

_init_once("vision_cache", None)
_init_once("vision_detect", None)
_init_once("landmark_desc", None)

# Légendes
_init_once("cap_0", "")
_init_once("cap_1", "")
_init_once("cap_2", "")
_init_once("style_select", "Inspiration")

# --------------------------- Sidebar -------------------------------------------
with st.sidebar:
    st.markdown("## 🖼️ Source")
    file = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="src_file")

    # Reset strict si image change
    if file is not None:
        new_bytes = file.getvalue()
        new_hash = _hash_bytes(new_bytes)
        if new_hash != st.session_state.get("image_hash"):
            st.session_state.image_bytes = new_bytes
            st.session_state.image_name = file.name
            st.session_state.image_hash = new_hash
            st.session_state.vision_cache = None
            st.session_state.landmark_desc = None
            st.session_state.vision_detect = None
            st.session_state.processed_np = None
            st.session_state.cap_0 = ""
            st.session_state.cap_1 = ""
            st.session_state.cap_2 = ""

    st.markdown("---")
    st.markdown("## 🎛️ Filtres & texte")
    filter_options = [
        "Aucun", "Grayscale", "Blur", "Canny Edge", "Sepia", "Cartoon",
        "Anime", "Watercolor", "Oil Paint", "Cubism",
        "Pencil Sketch", "Posterize", "Pixelate",
    ]
    st.selectbox("Filtre", filter_options, index=0, key="filter_type")

    st.markdown("### 🔤 Texte")
    st.session_state.custom_text = st.text_input(
        "Texte à afficher (overlay)",
        value=st.session_state.custom_text,
        placeholder="Votre titre / slogan…"
    )
    st.session_state.color_hex = st.color_picker("Couleur texte", st.session_state.color_hex)
    st.session_state.font_scale = st.slider("Taille (font scale)", 0.4, 5.0, st.session_state.font_scale, 0.1)
    st.session_state.thickness = st.slider("Épaisseur", 1, 8, st.session_state.thickness, 1)
    st.session_state.font_face = st.selectbox(
        "Police (Hershey)",
        [
            "Sans (Simplex)", "Sans (Plain)", "Sans (Duplex)",
            "Sans (Complex)", "Sans (Triplex)",
            "Serif (Complex)", "Serif (Triplex)", "Serif (Complex Small)",
            "Script (Simplex)", "Script (Complex)"
        ],
        index=0
    )
    st.markdown("### 📍 Position du texte")
    st.session_state.pos_x = st.number_input("X", min_value=0, value=st.session_state.pos_x, step=5)
    st.session_state.pos_y = st.number_input("Y", min_value=0, value=st.session_state.pos_y, step=5)

# --------------------------- Corps principal ------------------------------------
st.title("CaptionCrafter · Instagram Captions & Vision")

# --------------------------- Preview image --------------------------------------
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("### 🎨 Prévisualisation & export image")
    if st.session_state.image_bytes:
        np_img = _load_np_from_bytes(st.session_state.image_bytes)
        render = apply_filter(np_img, st.session_state.filter_type)
        if (st.session_state.custom_text or "").strip():
            render = add_text(
                render,
                st.session_state.custom_text,
                (int(st.session_state.pos_x), int(st.session_state.pos_y)),
                color=_hex_to_bgr(st.session_state.color_hex),
                font_face_name=st.session_state.font_face,
                font_scale=st.session_state.font_scale,
                thickness=st.session_state.thickness
            )
        render_image = _ensure_3ch(render)
        st.session_state.processed_np = render_image.copy()
        st.image(_bgr_to_pil(render_image), use_column_width=True, caption="Aperçu")
        buf = BytesIO(); _bgr_to_pil(render_image).save(buf, format="PNG")
        st.download_button("💾 Télécharger le PNG", data=buf.getvalue(), file_name="captioncrafter.png", mime="image/png", key="dl_png")
    else:
        st.info("Aperçu indisponible : importe une image.")

st.markdown("---")

# --------------------------- Analyses -------------------------------------------
if st.session_state.image_bytes:
    np_for_ai = _get_np_for_ai()
    colA, colB, colC = st.columns(3, gap="large")

    with colA:
        st.markdown("### 👁️ Résumé visuel")
        btn_vis = st.button("Analyser l'image (vision rapide)")
        with st.container():
            if btn_vis:
                try:
                    with st.spinner("Analyse en cours…"):
                        st.session_state.vision_cache = describe_visual(np_for_ai)
                except Exception as e:
                    st.session_state.vision_cache = {"error": str(e)}
            vis = st.session_state.get("vision_cache", None)
            if isinstance(vis, dict) and vis and not vis.get("error"):
                st.json(vis)
            elif isinstance(vis, dict) and vis.get("error"):
                st.error(f"Erreur: {vis.get('error')}")
            else:
                st.caption("— Aucune analyse encore —")

    with colB:
        st.markdown("### 🧭 Monument / objet principal")
        btn_landmark = st.button("🔎 Analyser le monument / l'objet principal", key="btn_landmark")
        box_landmark = st.container()
        if btn_landmark:
            try:
                with st.spinner("Pixtral analyse…"):
                    st.session_state.landmark_desc = describe_landmark_from_np(np_for_ai)
            except Exception as e:
                if not st.session_state.get("landmark_desc"):
                    st.session_state.landmark_desc = f"(Erreur) {e}"
        with box_landmark:
            txt = st.session_state.get("landmark_desc", None)
            if isinstance(txt, str) and txt.strip():
                st.markdown(
                    f"""<div style="background:#e8f7ec;border:1px solid #bfe7c9;border-radius:8px;padding:12px;">
                    <b>Description générée</b><br>{txt}</div>""",
                    unsafe_allow_html=True
                )
            else:
                st.caption("— Aucune description encore —")

    with colC:
        st.markdown("### 🧠 Scène & entités")
        btn_detect = st.button("Détecter (personnes/objets/actions)", key="btn_detect")
        box_detect = st.container()
        if btn_detect:
            try:
                with st.spinner("Détection en cours…"):
                    st.session_state.vision_detect = detect_visual_entities_from_np(np_for_ai)
            except Exception as e:
                st.session_state.vision_detect = {"error": str(e)}
        with box_detect:
            det = st.session_state.get("vision_detect", None)
            if isinstance(det, dict) and det and not det.get("error"):
                st.json(det)
            elif isinstance(det, dict) and det.get("error"):
                st.error(f"Erreur: {det.get('error')}")
            else:
                st.caption("— Aucune détection encore —")

# --------------------------- Génération de légendes -----------------------------
st.markdown("---")
st.markdown("## ✍️ Légendes Instagram")

if not st.session_state.image_bytes:
    st.warning("⚠️ Aucune image importée : impossible de générer des légendes Instagram.")
else:
    st.session_state.style_select = st.selectbox(
        "Style de légendes",
        ["Humour", "Émotion", "Inspiration", "Promotion", "Citation"],
        index=(["Humour","Émotion","Inspiration","Promotion","Citation"].index(st.session_state.get("style_select","Inspiration"))
               if st.session_state.get("style_select") in ["Humour","Émotion","Inspiration","Promotion","Citation"] else 2)
    )

    if st.button("Générer 3 légendes", key="btn_generate_caps"):
        try:
            vis = st.session_state.vision_cache or {}
            detect = st.session_state.vision_detect or {}
            vis_colors = (vis.get("colors") or vis.get("dominant_colors") or [])
            ocr_txt = extract_text(np_for_ai) if np_for_ai is not None else ""
            ctx = {
                "style": st.session_state.get("style_select", "Inspiration"),
                "overlay_text": st.session_state.custom_text,
                "ocr_text": ocr_txt,
                "filter_type": st.session_state.filter_type,
                "colors": vis_colors,
                "orientation": (vis or {}).get("orientation", ""),
                "brightness": (vis or {}).get("brightness", ""),
                "contrast": (vis or {}).get("contrast", ""),
                "scene": (detect or {}).get("scene", ""),
                "entities": (detect or {}).get("entities", []),
                "actions": (detect or {}).get("actions", []),
                "landmark_description": st.session_state.get("landmark_desc", ""),
            }
            with st.spinner("Génération des légendes…"):
                caps = generate_instagram_captions(ctx) or []
            st.session_state.cap_0 = caps[0] if len(caps) > 0 else ""
            st.session_state.cap_1 = caps[1] if len(caps) > 1 else ""
            st.session_state.cap_2 = caps[2] if len(caps) > 2 else ""
            st.success("3 légendes générées.")
        except Exception as e:
            st.error(f"Erreur: {e}")

    has_caps = any(bool(st.session_state[k].strip()) for k in ("cap_0","cap_1","cap_2"))

    if has_caps:
        st.markdown("### Propositions (éditables)")
        cols = st.columns(3)
        with cols[0]:
            st.text_area("Légende 1", key="cap_0", height=180)
        with cols[1]:
            st.text_area("Légende 2", key="cap_1", height=180)
        with cols[2]:
            st.text_area("Légende 3", key="cap_2", height=180)

        st.markdown("#### Poster sur 𝕏")
        buttons_html = f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          {_intent_anchor(st.session_state.cap_0, "Poster Légende 1")}
          {_intent_anchor(st.session_state.cap_1, "Poster Légende 2")}
          {_intent_anchor(st.session_state.cap_2, "Poster Légende 3")}
        </div>
        """
        st.markdown(buttons_html, unsafe_allow_html=True)

st.markdown("---")

# --------------------------- Génération d’images (Agent) ------------------------
st.markdown("## 🖼️ Générer une nouvelle image (Agent Mistral)")
gen_cols = st.columns([2, 1, 1])
with gen_cols[0]:
    gen_prompt = st.text_input("Prompt de génération",
                               placeholder="Ex: “Vue stylisée aquarelle d’un monument au coucher du soleil…”")
with gen_cols[1]:
    size = st.selectbox("Taille", ["512x512", "768x768", "1024x1024"], index=2)
with gen_cols[2]:
    do_gen = st.button("🎨 Générer")

if do_gen and gen_prompt:
    try:
        with st.spinner("Agent en cours…"):
            png_bytes = agent_generate_image(gen_prompt, size=size)
        st.success("Image générée")
        st.image(png_bytes, caption="Résultat agent (PNG)", use_column_width=True)
        st.download_button(
            "💾 Télécharger cette image",
            data=png_bytes,
            file_name="agent_generated.png",
            mime="image/png",
            key="dl_agent_png",
        )
    except Exception as e:
        st.error(f"Échec de génération : {e}")

# --------------------------- Footer ---------------------------------------------
st.markdown("---")
st.caption("🚀 Fait avec ❤️ par Armarit78 · CaptionCrafter × Mistral Vision & Agents — intent/tweet only")
