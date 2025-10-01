import os, time, random, base64, json, requests

# ----------- INFRA COMMUNE: appel robuste avec retry/backoff + fallback -----------
_DEFAULT_MODELS_TEXT = ["mistral-small-latest", "mistral-medium-latest"]
_DEFAULT_MODELS_VISION = ["pixtral-12b-latest", "mistral-medium-latest", "mistral-small-latest"]

def _chat_complete(payload, headers, models, max_retries=5):
    last_err = None
    for model in models:
        data = dict(payload, model=model)
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json=data, headers=headers, timeout=60
                )
                if resp.status_code == 200:
                    return resp.json()
                # 429 / 5xx â†’ backoff + retry
                if resp.status_code in (429, 500, 502, 503, 504):
                    sleep_s = (2 ** attempt) + random.uniform(0.0, 0.6)
                    time.sleep(min(sleep_s, 8))
                    continue
                resp.raise_for_status()
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt, 8))
                continue
        # on tente le modele suivant si tous les retries ont echoue
        last_err = RuntimeError(f"Échec sur modèle {model}")
    raise RuntimeError(f"Mistral API indisponible: {last_err}")

# ----------- GENERATION DE LEGENDES -----------
def generate_instagram_captions(context: dict):
    """
    context attendu (tous facultatifs sauf style):
    {
      "style": "Humour" | "Émotion" | "Inspiration" | "Promotion" | "Citation",
      "ocr_text": str,
      "overlay_text": str,
      "filter_type": str,
      "colors": ["bleu","blanc","noir"],
      "orientation": "portrait|paysage|carré",
      "brightness": "sombre|moyen|lumineux",
      "contrast": "faible|moyen|élevé",
      # Données vision (mini-détecteur):
      "scene": "street" | ...,
      "entities": [{"label":"person","count":1}, ...],
      "actions": ["running", "smiling"]
    }
    """
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        raise ValueError("Clé API Mistral non trouvée (MISTRAL_API_KEY)")

    # Prompt SYSTEM: ancrage strict Ã l'image
    system = (
        "Tu es un créateur de légendes Instagram. "
        "Ancre CHAQUE proposition UNIQUEMENT sur les éléments visuels fournis (scène, entités, actions, couleurs, texte). "
        "N’invente rien qui ne soit pas mentionné. "
        "Écris en français naturel, fluide et engageant, avec une structure de phrases variée. "
        "Produis exactement 3 propositions, chacune â‰¤ 300 caractères (espaces inclus). "
        "Elles doivent pouvoir être publiées aussi sur Twitter : pas de contenu offensant, inapproprié, politique, sexuel, haineux ou discriminant. "
        "Respecte le style demandé. Hashtags autorisés mais discrets (0 Ã  2). "
        "Pas d’explications ni d’emoji sauf si le style le suggère clairement."
    )

    # Contexte compact + vision
    # On synthetise les entites/action/scene pour guider le modele
    ents = ", ".join([f"{e.get('label','?')} x{e.get('count',1)}" for e in context.get("entities", [])]) or "—"
    acts = ", ".join(context.get("actions", [])) or "—"
    cols = ", ".join(context.get("colors", [])) or "—"

    user = (
        "Éléments visuels:\n"
        f"- Orientation: {context.get('orientation')}\n"
        f"- Scène: {context.get('scene', '—')}\n"
        f"- Entités: {ents}\n"
        f"- Actions dominantes: {acts}\n"
        f"- Couleurs dominantes: {cols}\n"
        f"- Luminosité: {context.get('brightness')} | Contraste: {context.get('contrast')}\n"
        f"- Filtre appliqué: {context.get('filter_type')}\n"
        f"- Texte détecté (OCR): {context.get('ocr_text') or '—'}\n"
        f"- Texte superposé par l’utilisateur: {context.get('overlay_text') or '—'}\n\n"
        f"Style demandé: {context.get('style')}\n\n"
        "Consignes de sortie:\n"
        "- Donne exactement 3 propositions numérotées 1), 2), 3)\n"
        "- Chaque ligne = 1 proposition courte, ancrée aux éléments ci-dessus.\n"
        "- Pas d’explications, pas de code.\n"
    )

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 280
    }

    data = _chat_complete(payload, headers, models=_DEFAULT_MODELS_TEXT)
    content = data["choices"][0]["message"]["content"].strip()

    # Extraction robuste des 3 lignes
    lines = []
    for line in content.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s[0] in "123" and ")" in s[:3]:
            s = s.split(")", 1)[1].strip()
        lines.append(s)
        if len(lines) == 3:
            break
    while len(lines) < 3:
        lines.append("Légende indisponible.")
    return lines[:3]

# ----------- MINI DETECTEUR VISION (Pixtral/Small/Medium) -----------
def detect_visual_entities_from_np(np_bgr) -> dict:
    """
    Envoie l'image (en mémoire) au modèle vision et récupère un JSON:
    { "entities": [{"label":"person","count":1}, {"label":"dog","count":1}],
      "actions": ["running"], "scene":"street" }
    """
    import cv2  # import local pour eviter dependance inutile si non utilise
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        raise ValueError("MISTRAL_API_KEY absent")

    # Encode PNG -> base64 (data URI)
    _, buf = cv2.imencode(".png", np_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    system = (
        "Tu es un assistant de vision par ordinateur. "
        "Analyse l'image et RENDS UNIQUEMENT un JSON compact avec les clés: "
        "\"entities\" (liste d'objets {label,count} avec labels génériques type COCO), "
        "\"actions\" (liste de verbes simples), "
        "\"scene\" (1-2 mots max). Pas d'explications."
    )
    user_content = [
        {"type": "text", "text":
            "Détecte: personnes/animaux/véhicules/objets saillants et actions dominantes. "
            "Libellés en anglais si possible (person, dog, car...)."
        },
        {"type": "image_url", "image_url": data_uri}
    ]
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 300,
        "temperature": 0.2
    }

    data = _chat_complete(payload, headers, models=_DEFAULT_MODELS_VISION)
    text = data["choices"][0]["message"]["content"]

    # Essaye d'extraire un JSON meme si le modele a entoure de texte
    try:
        start = text.find("{")
        end = text.rfind("}")
        js = json.loads(text[start:end+1])
        # garde une forme standardisee
        return {
            "entities": js.get("entities", []),
            "actions": js.get("actions", []),
            "scene": js.get("scene", "")
        }
    except Exception:
        return {"entities": [], "actions": [], "scene": ""}
