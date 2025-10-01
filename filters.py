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
        return _to_bgr(cv2.GaussianBlur(img, (15, 15), 0))

    if filter_type == "Canny Edge":
        return _to_bgr(_edge_mask(img))

    if filter_type == "Sepia":
        sepia = cv2.transform(
            img,
            np.array([[0.272, 0.534, 0.131],
                      [0.349, 0.686, 0.168],
                      [0.393, 0.769, 0.189]])
        )
        return _to_bgr(np.clip(sepia, 0, 255).astype(np.uint8))

    if filter_type == "Cartoon":
        # edges + lissage couleur
        edges = _edge_mask(img, k=5, t1=80, t2=160)
        color = cv2.bilateralFilter(img, d=9, sigmaColor=150, sigmaSpace=150)
        cartoon = cv2.bitwise_and(color, color, mask=cv2.bitwise_not(edges))
        return _to_bgr(cartoon)

    if filter_type == "Posterize":
        # quantification + edges légers
        post = _posterize(img, levels=6)
        edges = _edge_mask(img, k=5, t1=60, t2=140)
        edges_inv = cv2.bitwise_not(edges)
        edges_inv_col = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
        return _to_bgr(cv2.bitwise_and(post, edges_inv_col))

    if filter_type == "Pixelate":
        return _pixelate(img, scale=0.12)

    if filter_type == "Pencil Sketch":
        # sketch crayon via dodge blend
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (25, 25), 0)
        blend = cv2.divide(gray, 255 - blur, scale=256)
        return _to_bgr(blend)

    if filter_type == "Watercolor":
        # cv2.stylization donne souvent un rendu aquarelle
        try:
            water = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
            return _to_bgr(water)
        except Exception:
            # fallback: lissage + posterize
            soft = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
            return _to_bgr(_posterize(soft, levels=8))

    if filter_type == "Oil Paint":
        # xphoto.oilPainting si dispo, sinon fallback
        try:
            oil = cv2.xphoto.oilPainting(img, size=7, dynRatio=1)
            return _to_bgr(oil)
        except Exception:
            # approximation: forte bilateral + posterize
            smooth = cv2.bilateralFilter(img, 11, 250, 250)
            return _to_bgr(_posterize(smooth, levels=8))

    if filter_type == "Anime":
        # rendu type 'anime': lissage + contours marques + posterize
        smooth = cv2.bilateralFilter(img, 9, 200, 200)
        post = _posterize(smooth, levels=8)
        edges = _edge_mask(post, k=3, t1=80, t2=160)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges_inv = cv2.bitwise_not(edges)
        edges_inv_col = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
        return _to_bgr(cv2.bitwise_and(post, edges_inv_col))

    if filter_type == "Cubism":
        # effet "bloc/cubique": pixelate leger + quantif + contours
        base = _pixelate(img, scale=0.2)
        post = _posterize(base, levels=5)
        edges = _edge_mask(img, k=5, t1=70, t2=150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges_inv = cv2.bitwise_not(edges)
        edges_inv_col = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
        return _to_bgr(cv2.bitwise_and(post, edges_inv_col))

    # par defaut, renvoie l'image originale
    return _to_bgr(img)

# --- Ajout de texte avec choix de typo ------------------------------------------
def add_text(
    image,
    text,
    position=(30, 30),
    font_scale=1.2,
    color=(0, 255, 0),
    thickness=2,
    font_face_name="Sans (Simplex)"
):
    """
    Dessine du texte avec choix de typo (polices Hershey).
    font_face_name options suggerees :
      - "Sans (Simplex)", "Sans (Plain)", "Sans (Duplex)"
      - "Serif (Times/Complex)", "Serif (Triplex)", "Serif (Complex Small)"
      - "Script (Simplex)", "Script (Complex)"
    """
    img = _to_bgr(image).copy()
    font = _cv2_font(font_face_name)
    return cv2.putText(
        img,
        text or "",
        position,
        font,
        float(font_scale),
        color,
        int(thickness),
        cv2.LINE_AA
    )
