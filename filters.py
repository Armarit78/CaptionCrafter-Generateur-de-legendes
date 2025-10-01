import cv2
import numpy as np

def _to_bgr(img):
    if img is None:
        return img
    if len(img.shape) == 2:  # gray -> BGR
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _posterize(img, levels=6):
    # Quantification des couleurs pour un rendu "toon"
    div = max(int(256 / max(2, levels)), 1)
    return _to_bgr((img // div) * div)

def _pixelate(img, scale=0.15):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return _to_bgr(cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST))

def _edge_mask(img, k=5, t1=100, t2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, k)
    edges = cv2.Canny(gray, t1, t2)
    return edges

# --- Typos (Hershey) ------------------------------------------------------------
def _cv2_font(face: str):
    """
    Mappe un nom lisible vers une police Hershey OpenCV.
    Les libellés sont des alias conviviaux (pas de TTF natif).
    """
    face = (face or "Sans (Simplex)").strip().lower()
    aliases = {
        "sans (complex)": "serif (times/complex)",   # pas de 'sans complex' dans Hershey
        "sans (triplex)": "serif (triplex)",         # idem pour triplex
        "serif (complex)": "serif (times/complex)",  # uniformiser le label
    }
    face = aliases.get(face, face)
    mapping = {
        "sans (simplex)": cv2.FONT_HERSHEY_SIMPLEX,
        "sans (plain)": cv2.FONT_HERSHEY_PLAIN,
        "sans (duplex)": cv2.FONT_HERSHEY_DUPLEX,
        "serif (times/complex)": cv2.FONT_HERSHEY_COMPLEX,
        "serif (triplex)": cv2.FONT_HERSHEY_TRIPLEX,
        "serif (complex small)": cv2.FONT_HERSHEY_COMPLEX_SMALL,
        "script (simplex)": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        "script (complex)": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    }
    return mapping.get(face, cv2.FONT_HERSHEY_SIMPLEX)

# --- Filtres --------------------------------------------------------------------
def apply_filter(image, filter_type):
    """
    Retourne toujours une image BGR 3 canaux (pour un affichage/texte sûr).
    Filtres disponibles:
      - Aucun, Grayscale, Blur, Canny Edge, Sepia, Cartoon
      - Anime, Watercolor, Oil Paint, Cubism, Pencil Sketch, Posterize, Pixelate
    """
    img = image.copy()

    if filter_type in (None, "Aucun"):
        return _to_bgr(img)

    if filter_type == "Grayscale":
        return _to_bgr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    if filter_type == "Blur":
        return _to_bgr(cv2.GaussianBlur(img, (9, 9), 2))

    if filter_type == "Canny Edge":
        edges = _edge_mask(img, k=5, t1=80, t2=160)
        return _to_bgr(edges)

    if filter_type == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sep = cv2.transform(img, kernel)
        return _to_bgr(np.clip(sep, 0, 255).astype(np.uint8))

    if filter_type == "Cartoon":
        color = cv2.bilateralFilter(img, 9, 250, 250)
        edges = _edge_mask(img, k=7, t1=50, t2=150)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return _to_bgr(cv2.bitwise_and(color, edges))

    if filter_type == "Anime":
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        edges = _edge_mask(blur, k=7, t1=80, t2=160)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return _to_bgr(cv2.bitwise_and(blur, edges))

    if filter_type == "Watercolor":
        dst = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
        return _to_bgr(dst)

    if filter_type == "Oil Paint":
        try:
            return _to_bgr(cv2.xphoto.oilPainting(img, 7, 1))
        except Exception:
            return _to_bgr(img)

    if filter_type == "Cubism":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return _to_bgr(cv2.addWeighted(img, 0.7, edges, 0.3, 0))

    if filter_type == "Pencil Sketch":
        gray, sketch = cv2.pencilSketch(img, sigma_s=50, sigma_r=0.07, shade_factor=0.05)
        return _to_bgr(sketch)

    if filter_type == "Posterize":
        return _posterize(img, levels=6)

    if filter_type == "Pixelate":
        return _pixelate(img, scale=0.15)

    return _to_bgr(img)  # fallback

# --- Overlay texte -----------------------------------------------------
def add_text(image, text, position, color=(255,255,255),
             font_face_name="Sans (Simplex)", font_scale=1.0, thickness=2,
             max_chars=40):
    """
    Ajoute du texte avec retour à la ligne forcé tous les `max_chars`.
    - Contour noir automatique
    - Centrage si position=(-1,-1)
    """
    img = _to_bgr(image).copy()
    font = _cv2_font(font_face_name)
    h, w = img.shape[:2]

    if not text:
        return img

    # ----------- découpage par nombre de caractères ----------
    lines = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    # ----------- calcul hauteur ligne ----------
    (_, text_height), _ = cv2.getTextSize("Ay", font, font_scale, thickness)
    line_height = int(text_height * 1.5)
    total_h = line_height * len(lines)

    # ----------- position verticale ----------
    if position == (-1, -1):
        start_y = (h - total_h) // 2
    else:
        start_y = position[1]

    # ----------- dessin ----------
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        if position == (-1, -1):
            x = (w - tw) // 2
        else:
            x = position[0]
        y = start_y + (i+1) * line_height

        # contour noir
        cv2.putText(img, line, (x, y), font, font_scale,
                    (0,0,0), thickness+2, cv2.LINE_AA)
        # texte principal
        cv2.putText(img, line, (x, y), font, font_scale,
                    color, thickness, cv2.LINE_AA)

    return img




