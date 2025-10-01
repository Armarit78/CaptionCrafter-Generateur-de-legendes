import cv2
import numpy as np
import os
import platform

# OCR optionnel (inchange dans l'esprit)
_USE_OCR = True
_TESS_PATH_SET = False

try:
    import pytesseract
    from PIL import Image
except Exception:
    _USE_OCR = False
    from PIL import Image  # pour les types uniquement


def _configure_tesseract_path():
    global _TESS_PATH_SET
    if not _USE_OCR or _TESS_PATH_SET:
        return
    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd and os.path.exists(tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
        _TESS_PATH_SET = True
        return
    if platform.system().lower().startswith("win"):
        for c in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
            if os.path.exists(c):
                pytesseract.pytesseract.tesseract_cmd = c
                _TESS_PATH_SET = True
                return

def _tesseract_available() -> bool:
    if not _USE_OCR:
        return False
    try:
        _configure_tesseract_path()
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def analyze_image(pil_image: "Image.Image") -> str:
    """OCR simple. Ne plante jamais si Tesseract absent."""
    image = np.array(pil_image.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    scale = 800 / max(h, w)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))

    if not _tesseract_available():
        return "—"  # neutre pour le prompt, pas de roman inutile

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
    except Exception:
        return "—"
    text = (text or "").strip().replace("\n", " ")
    return text if text else "—"

# >>> NOUVEAU : resume visuel factuel (couleurs, orientation, luminosite/contraste)
def describe_visual(np_bgr: np.ndarray) -> dict:
    h, w = np_bgr.shape[:2]
    orientation = "paysage" if w > h else ("portrait" if h > w else "carré")

    hsv = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[:, :, 2]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    brightness = "sombre" if v_mean < 90 else ("moyen" if v_mean < 170 else "lumineux")
    # contraste approxime via ecart-type de V
    v_std = float(np.std(hsv[:, :, 2]))
    contrast = "faible" if v_std < 35 else ("moyen" if v_std < 70 else "élevé")

    # Couleurs dominantes rapides (kmeans sur echantillon reduit)
    small = cv2.resize(np_bgr, (128, 128)).reshape(-1, 3).astype(np.float32)
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(small, K, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(int)  # BGR
    # Mapper BGR -> nom simple
    def name_color(bgr):
        b, g, r = bgr
        mx = max(b, g, r)
        if mx < 50: return "noir"
        if min(b, g, r) > 200: return "blanc"
        if r > g and r > b: return "rouge" if r - max(g, b) > 40 else "rosé"
        if g > r and g > b: return "vert" if g - max(r, b) > 40 else "jaune-vert"
        if b > r and b > g: return "bleu" if b - max(r, g) > 40 else "cyan"
        if r > 180 and g > 130 and b < 80: return "jaune"
        return "gris"
    color_names = []
    for c in centers:
        n = name_color(c.tolist())
        if n not in color_names:
            color_names.append(n)
    if len(color_names) == 0:
        color_names = ["gris"]

    return {
        "orientation": orientation,
        "brightness": brightness,
        "contrast": contrast,
        "colors": color_names[:3]
    }
