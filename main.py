import streamlit as st
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import urllib.parse

from image_utils import analyze_image, describe_visual
from mistral_api import generate_instagram_captions, detect_visual_entities_from_np
from filters import apply_filter, add_text
from dotenv import load_dotenv
from streamlit_drawable_canvas import st_canvas

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="CaptionCrafter", page_icon="📷")

# ---------- State (tout persistant) ----------
defaults = {
    "final_image": None,          # image BGR finale pour le mode Aperçu
    "selected": None,             # légende choisie (texte)
    "captions": None,             # propositions de légendes (liste)
    "image_bytes": None,          # image uploadée (bytes)
    "image_name": None,           # nom du fichier

    # Options créatives
    "style": "Inspiration",       # style de génération
    "style_backup": None,         # 🔒 sauvegarde style avant passage en mode final

    # Filtre & sauvegarde
    "filter_type": "Aucun",
    "filter_type_backup": None,   # 🔒 sauvegarde filtre

    "image_title": "",

    # Texte & sauvegardes
    "custom_text": "",
    "custom_text_backup": None,   # 🔒 sauvegarde texte

    # Apparence du texte & sauvegardes
    "font_scale": 1.2,
    "font_scale_backup": None,    # 🔒 sauvegarde taille
    "color": "#00ff00",
    "color_backup": None,         # 🔒 sauvegarde couleur
    "thickness": 2,
    "thickness_backup": None,     # 🔒 sauvegarde épaisseur

    # Typo (police)
    "font_face": "Sans (Simplex)",     # valeur lisible mappée Hershey
    "font_face_backup": None,          # 🔒 sauvegarde typo

    "pos_x": 30,
    "pos_y": 30,

    "vision_cache": None,         # cache du mini-détecteur vision
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------- Utilitaires ----------
hex_to_bgr = lambda hx: tuple(int(hx.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))


# =============================== MODE APERÇU FINAL ===============================
if st.session_state.final_image is not None:
    st.title("✅ Aperçu du visuel final")

    image = st.session_state.final_image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="📷 Image finale", use_column_width=True)

    caption = st.session_state.get("selected")
    if caption:
        st.markdown("### 📝 Texte sélectionné :")
        st.write(caption)

        # Téléchargements
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_data = buf.getvalue()

        colA, colB = st.columns(2)
        with colA:
            st.download_button("📷 Télécharger l'image (PNG)", data=img_data,
                               file_name="caption_image.png", mime="image/png")
        with colB:
            st.download_button("📥 Télécharger la légende (txt)", caption,
                               file_name="caption.txt", mime="text/plain")

        st.markdown("---")
        st.markdown("## 📤 Poster sur Twitter")
        st.text_area("✍️ Message à poster sur Twitter :", value=caption, height=100, key="tweet_area")
        twitter_url = "https://twitter.com/intent/tweet?text=" + urllib.parse.quote(caption)
        st.markdown(f"[📲 Ouvrir X/Twitter pour publier]({twitter_url})")

    st.markdown("---")

    # 🔙 Retour : on restaure tout ce qui a été sauvegardé, puis on revient en création
    if st.button("🔙 Retour à la création"):
        # Texte
        if (st.session_state.get("custom_text") in (None, "")) and st.session_state.get("custom_text_backup"):
            st.session_state.custom_text = st.session_state.custom_text_backup
        # Couleur / taille / épaisseur / style / typo
        if st.session_state.get("color_backup"):
            st.session_state.color = st.session_state.color_backup
        if st.session_state.get("font_scale_backup") is not None:
            st.session_state.font_scale = st.session_state.font_scale_backup
        if st.session_state.get("thickness_backup") is not None:
            st.session_state.thickness = st.session_state.thickness_backup
        if st.session_state.get("style_backup"):
            st.session_state.style = st.session_state.style_backup
        if st.session_state.get("font_face_backup"):
            st.session_state.font_face = st.session_state.font_face_backup
        # Filtre
        if st.session_state.get("filter_type_backup"):
            st.session_state.filter_type = st.session_state.filter_type_backup

        st.session_state.final_image = None
        st.experimental_rerun()

    st.stop()


# =============================== MODE CRÉATION ===============================
st.title("📷 CaptionCrafter")
st.subheader("Créateur de visuels Instagram ancrés sur l’image (Vision + OCR + filtres)")

# Réhydratation AUTOMATIQUE si l'input/valeur serait perdue après retour
if st.session_state.final_image is None:
    if (st.session_state.get("custom_text") in (None, "")) and st.session_state.get("custom_text_backup"):
        st.session_state.custom_text = st.session_state.custom_text_backup
    if st.session_state.get("color_backup") and not st.session_state.get("color"):
        st.session_state.color = st.session_state.color_backup
    if st.session_state.get("font_scale_backup") is not None and st.session_state.get("font_scale") is None:
        st.session_state.font_scale = st.session_state.font_scale_backup
    if st.session_state.get("thickness_backup") is not None and st.session_state.get("thickness") is None:
        st.session_state.thickness = st.session_state.thickness_backup
    if st.session_state.get("style_backup") and not st.session_state.get("style"):
        st.session_state.style = st.session_state.style_backup
    if st.session_state.get("filter_type_backup") and not st.session_state.get("filter_type"):
        st.session_state.filter_type = st.session_state.filter_type_backup
    if st.session_state.get("font_face_backup") and not st.session_state.get("font_face"):
        st.session_state.font_face = st.session_state.font_face_backup

# --- Entrées & options (avec keys pour persistance) ---
st.text_input("🖼️ Titre (facultatif) de l'image :", key="image_title")
uploaded = st.file_uploader("📤 Upload une image", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded is not None:
    st.session_state.image_bytes = uploaded.getvalue()
    st.session_state.image_name = uploaded.name
    st.session_state.vision_cache = None  # invalide la vision si nouvelle image

# Style
st.selectbox(
    "🎭 Style de texte généré :",
    ["Inspiration", "Humour", "Émotion", "Promotion", "Citation"],
    key="style"
)

# Filtre
st.selectbox(
    "🎨 Filtre à appliquer :",
    [
        "Aucun", "Grayscale", "Blur", "Canny Edge", "Sepia", "Cartoon",
        "Anime", "Watercolor", "Oil Paint", "Cubism", "Pencil Sketch",
        "Posterize", "Pixelate"
    ],
    key="filter_type"
)
st.caption(f"🎛️ Filtre actuel : **{st.session_state.filter_type}**")

st.markdown("---")

# Typo (police)
font_options = [
    "Sans (Simplex)",
    "Sans (Plain)",
    "Sans (Duplex)",
    "Serif (Times/Complex)",
    "Serif (Triplex)",
    "Serif (Complex Small)",
    "Script (Simplex)",
    "Script (Complex)",
]
st.selectbox("🅰️ Typo (police)", font_options, key="font_face")

# Apparence texte
st.text_input("💬 Texte à ajouter sur l'image", key="custom_text")
st.slider("🔠 Taille du texte", 0.5, 3.0, key="font_scale")
st.color_picker("🎨 Couleur du texte", key="color")
st.slider("🖋️ Épaisseur", 1, 8, key="thickness")

# --- Traitement si on a une image en mémoire ---
if st.session_state.image_bytes:
    # Lecture image + filtre (réappliqué à CHAQUE rendu selon filter_type)
    image = Image.open(BytesIO(st.session_state.image_bytes)).convert('RGB')
    np_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    np_image = apply_filter(np_image, st.session_state.filter_type)

    # Analyses locales
    ocr_text = analyze_image(image)         # "—" si rien
    vis = describe_visual(np_image)         # {orientation, brightness, contrast, colors}

    # Mini-détecteur Vision (API) — cache pour éviter des appels multiples
    if st.session_state.vision_cache is None:
        with st.spinner("Analyse visuelle (détection entités/actions)…"):
            try:
                vision_js = detect_visual_entities_from_np(np_image)
            except Exception as e:
                st.warning(f"Vision indisponible : {e}")
                vision_js = {"entities": [], "actions": [], "scene": ""}
        st.session_state.vision_cache = vision_js
    else:
        vision_js = st.session_state.vision_cache

    # Aperçu visuel (une seule image, pas de toolbar)
    st.markdown("## 🖼️ Aperçu (une seule image)")
    st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=0,
        background_image=Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)),
        update_streamlit=False,
        height=np_image.shape[0],
        width=np_image.shape[1],
        drawing_mode="transform",
        key="canvas",
        display_toolbar=False
    )

    # Position curseurs — persistants
    max_x, max_y = int(np_image.shape[1] - 1), int(np_image.shape[0] - 1)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.pos_x = st.slider("📍 Position X", 0, max_x, st.session_state.pos_x, key="pos_x_slider")
    with col2:
        st.session_state.pos_y = st.slider("📍 Position Y", 0, max_y, st.session_state.pos_y, key="pos_y_slider")

    # Rendu avec texte — utilise tjs les valeurs persistées (couleur/taille/épaisseur/typo)
    render_image = np_image.copy()
    if st.session_state.custom_text:
        render_image = add_text(
            render_image,
            st.session_state.custom_text,
            position=(st.session_state.pos_x, st.session_state.pos_y),
            font_scale=float(st.session_state.font_scale),
            color=hex_to_bgr(st.session_state.color),
            thickness=int(st.session_state.thickness),
            font_face_name=st.session_state.font_face
        )

    st.image(cv2.cvtColor(render_image, cv2.COLOR_BGR2RGB), caption="📸 Image modifiée", use_column_width=True)

    # Génération des légendes (contexte riche : OCR + vision + visuel + overlay text)
    if st.button("✍️ Générer texte de post Instagram"):
        try:
            with st.spinner("Génération via Mistral…"):
                old_selected = st.session_state.get("selected")
                description = (st.session_state.image_title + ". " if st.session_state.image_title else "") + (ocr_text or "")
                ctx = {
                    "style":       st.session_state.style,
                    "ocr_text":    description.strip() if description.strip() else "—",
                    "overlay_text": st.session_state.custom_text or "—",
                    "filter_type": st.session_state.filter_type,
                    "colors":      vis.get("colors", []),
                    "orientation": vis.get("orientation"),
                    "brightness":  vis.get("brightness"),
                    "contrast":    vis.get("contrast"),
                    "scene":       vision_js.get("scene", ""),
                    "entities":    vision_js.get("entities", []),
                    "actions":     vision_js.get("actions", []),
                }
                new_caps = generate_instagram_captions(ctx)
                st.session_state.captions = new_caps

                if old_selected in (new_caps or []):
                    st.session_state.selected = old_selected
                else:
                    st.session_state.selected = new_caps[0] if new_caps else None

        except Exception as e:
            st.error(f"Erreur pendant la génération : {e}")
            st.session_state.captions = None

    # Choix de légende — index basé sur la sélection existante
    if st.session_state.captions:
        st.markdown("### 📣 Propositions de texte Instagram :")
        options = st.session_state.captions
        idx = 0
        if st.session_state.selected in options:
            idx = options.index(st.session_state.selected)

        st.session_state.selected = st.radio(
            "✅ Choisis la légende que tu veux utiliser :",
            options,
            index=idx,
            key="caption_choice"
        )

    # Passage en Aperçu final — on SAUVEGARDE tout, puis on prévisualise
    if st.session_state.get("selected"):
        st.success("Cette légende sera utilisée pour ton post :")
        st.write(st.session_state.selected)

        if st.button("➡️ Prévisualiser le rendu final"):
            # Sauvegardes avant passage en mode final
            st.session_state.custom_text_backup = st.session_state.custom_text
            st.session_state.color_backup = st.session_state.color
            st.session_state.font_scale_backup = st.session_state.font_scale
            st.session_state.thickness_backup = st.session_state.thickness
            st.session_state.style_backup = st.session_state.style
            st.session_state.filter_type_backup = st.session_state.filter_type
            st.session_state.font_face_backup = st.session_state.font_face

            st.session_state.final_image = render_image
            st.experimental_rerun()

st.markdown("---")
st.caption("🚀 Fait avec ❤️ par Armarit78 · CaptionCrafter pour Mistral AI")
