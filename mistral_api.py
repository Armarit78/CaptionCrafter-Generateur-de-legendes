import os
import time
import random
import base64
import json
import requests
from threading import Lock

# -------------------- Modèles par défaut --------------------
_DEFAULT_MODELS_TEXT = ["mistral-small-latest", "mistral-medium-latest"]
_DEFAULT_MODELS_VISION = ["pixtral-12b-latest", "mistral-medium-latest", "mistral-small-latest"]

# -------------------- Cache agent (réutilisation) --------------------
_AGENT_ID = None
_AGENT_EXPIRES_AT = 0
_AGENT_LOCK = Lock()
_AGENT_TTL_SECONDS = 3600  # 1h

# -------------------- Appel chat robuste (retry) --------------------
def _chat_complete(payload, headers, models, max_retries=5):
    """
    Appelle l'API Mistral avec retry exponentiel et fallback sur plusieurs modèles.
    Retourne directement le JSON de sortie si succès, sinon [].
    """
    last_err = None
    for model in models:
        data = dict(payload, model=model)
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json=data,
                    headers=headers,
                    timeout=60
                )
                if resp.status_code == 200:
                    out = resp.json()
                    return out
                else:
                    last_err = resp.text
            except Exception as e:
                last_err = str(e)
            # backoff exponentiel + jitter
            time.sleep((2 ** attempt) + random.uniform(0, 0.25))
        # tentative avec modèle suivant
        continue
    # échec complet : renvoie liste vide
    return []

# -------------------- Helpers de normalisation --------------------
def _norm_entities(ents_in):
    """
    Accepte: list[dict(label,count)] OU list[str] OU mixte.
    Retourne: list[str] (ex: ["person x2", "dog"])
    """
    out = []
    if not isinstance(ents_in, (list, tuple)):
        return out
    for e in ents_in:
        if isinstance(e, dict):
            lbl = e.get("label", "?")
            cnt = e.get("count", 1)
            try:
                cnt = int(cnt)
            except Exception:
                cnt = 1
            lbl = f"{lbl}".strip() or "?"
            out.append(f"{lbl} x{cnt}")
        elif isinstance(e, str):
            s = e.strip()
            if s:
                out.append(s)
        else:
            try:
                s = str(e).strip()
                if s:
                    out.append(s)
            except Exception:
                pass
    return out

def _norm_list_str(lst):
    """Nettoie toute liste/tuple en liste[str] sans None/vides."""
    if not isinstance(lst, (list, tuple)):
        return []
    out = []
    for x in lst:
        if isinstance(x, str):
            s = x.strip()
        else:
            try:
                s = str(x).strip()
            except Exception:
                s = ""
        if s:
            out.append(s)
    return out

def _pick_from_dict_list(lst, keys):
    """
    Récupère la première valeur non vide correspondant à un des keys donnés.
    Exemple: [{"label":"dog"}], keys=["label"] -> ["dog"]
    """
    out = []
    for e in lst:
        if isinstance(e, dict):
            for k in keys:
                if k in e and str(e[k]).strip():
                    out.append(str(e[k]).strip())
                    break
        else:
            try:
                s = str(e).strip()
                if s:
                    out.append(s)
            except Exception:
                pass
    return out

# -------------------- Génération de légendes IG --------------------
def generate_instagram_captions(context: dict):
    """
    Génère 3 légendes Instagram contextualisées.
    Le contexte inclut style, OCR, overlay, vision, couleurs...
    """
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        raise ValueError("Clé API Mistral non trouvée (MISTRAL_API_KEY)")

    # Normalisations robustes
    ents = ", ".join(_norm_entities(context.get("entities", []))) or "—"

    actions_in = context.get("actions", [])
    if isinstance(actions_in, list) and actions_in and isinstance(actions_in[0], dict):
        acts = _pick_from_dict_list(actions_in, ["label", "name", "text", "value", "title"])
    else:
        acts = _norm_list_str(actions_in)
    acts = ", ".join(acts) or "—"

    colors_in = context.get("colors", [])
    if isinstance(colors_in, list) and colors_in and isinstance(colors_in[0], dict):
        cols = _pick_from_dict_list(colors_in, ["name", "label", "value", "text", "title"])
    else:
        cols = _norm_list_str(colors_in)
    cols = ", ".join(cols) or "—"

    landmark = (context.get("landmark_description") or "").strip() or "—"

    # Prompt système : consignes claires
    system = (
        "Tu es un créateur de légendes Instagram. "
        "Ancre CHAQUE proposition UNIQUEMENT sur les éléments visuels fournis. "
        "Si un monument célèbre est mentionné explicitement dans les données, tu peux le nommer, "
        "mais n'invente JAMAIS. "
        "Écris en français naturel. "
        "Produis exactement 3 propositions de 500 caractères chacune. "
        "Hashtags: maximum 2. Pas d’explications."
    )

    user = (
        "Éléments visuels:\n"
        f"- Orientation: {context.get('orientation')}\n"
        f"- Scène: {context.get('scene', '—')}\n"
        f"- Entités: {ents}\n"
        f"- Actions dominantes: {acts}\n"
        f"- Couleurs dominantes: {cols}\n"
        f"- Luminosité: {context.get('brightness')} | Contraste: {context.get('contrast')}\n"
        f"- Filtre appliqué: {context.get('filter_type')}\n"
        f"- Texte dans l'image (OCR): {(context.get('ocr_text') or '—')}\n"
        f"- Texte overlay: {(context.get('overlay_text') or '—')}\n"
        f"- Monument/ouvrage (si présent): {landmark}\n"
        f"- Style demandé: {context.get('style')}"
    )

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 650
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

# -------------------- Mini-détecteur vision (VLM) --------------------
def detect_visual_entities_from_np(np_bgr) -> dict:
    """
    Envoie l'image au modèle vision et récupère un JSON:
    {
      "entities": [{"label":"person","count":1}, {"label":"dog","count":1}],
      "actions": ["running"],
      "scene": "street"
    }
    """
    import cv2
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        raise ValueError("Clé API Mistral non trouvée (MISTRAL_API_KEY)")

    # Encode image en base64
    _, buf = cv2.imencode(".jpg", np_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_uri = {"url": f"data:image/jpeg;base64,{b64}"}

    system = (
        "Tu es un assistant vision. Réponds UNIQUEMENT avec un JSON compact: "
        "{\"entities\":[], \"actions\":[], \"scene\":\"\"}."
    )
    user = (
        "Retourne un JSON avec trois champs: "
        "\"entities\" (chaque entrée: {\"label\":str, \"count\":int}), "
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

    # Essaye d'extraire un JSON même si le modèle rajoute des fences ```json
    try:
        import re
        txt = text.strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", txt)
        mjson = re.search(r"\{[\s\S]*\}", txt)
        js = json.loads(mjson.group(0)) if mjson else {}
        entities = js.get("entities", [])
        actions = js.get("actions", [])
        scene = js.get("scene", "")
        if not isinstance(entities, list): entities = []
        if not isinstance(actions, list): actions = []
        if not isinstance(scene, str): scene = ""
        return {"entities": entities, "actions": actions, "scene": scene}
    except Exception:
        return {"entities": [], "actions": [], "scene": ""}

# -------------------- Agents: génération d'image --------------------
def _create_agent(headers_json: dict) -> str:
    """Crée un agent d'image sur Mistral et renvoie son id."""
    agent_payload = {
        "model": "mistral-medium-latest",
        "name": "Image Generation Agent",
        "description": "Agent used to generate images.",
        "instructions": "Use the image generation tool when you have to create images.",
        "tools": [{"type": "image_generation"}],
        "completion_args": {"temperature": 0.3, "top_p": 0.95},
    }
    r = requests.post(
        "https://api.mistral.ai/v1/agents",
        json=agent_payload,
        headers=headers_json,
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Agent creation failed: {r.status_code} {r.text}")
    agent_id = (r.json() or {}).get("id")
    if not agent_id:
        raise RuntimeError(f"Agent creation returned no id: {r.text}")
    return agent_id

def _get_or_create_agent(headers_json: dict) -> str:
    """
    Renvoie un agent réutilisable (cache mémoire + TTL).
    Recrée l'agent si expiré ou invalide.
    """
    global _AGENT_ID, _AGENT_EXPIRES_AT
    now = time.time()
    if _AGENT_ID and now < _AGENT_EXPIRES_AT:
        return _AGENT_ID
    with _AGENT_LOCK:
        now = time.time()
        if _AGENT_ID and now < _AGENT_EXPIRES_AT:
            return _AGENT_ID
        agent_id = _create_agent(headers_json)
        _AGENT_ID = agent_id
        _AGENT_EXPIRES_AT = now + _AGENT_TTL_SECONDS
        return _AGENT_ID

def _start_conversation_with_retry(agent_id: str, prompt_full: str, headers_json: dict, max_retries: int = 5):
    """
    Lance la conversation avec gestion 429/5xx, Retry-After et backoff exponentiel.
    Retourne le JSON si succès, sinon lève RuntimeError.
    """
    delay = 1.0
    last_err = None
    for attempt in range(max_retries):
        rr = requests.post(
            "https://api.mistral.ai/v1/conversations",
            json={"agent_id": agent_id, "inputs": prompt_full, "stream": False},
            headers=headers_json,
            timeout=120,
        )
        if rr.status_code == 200:
            return rr.json()
        if rr.status_code == 429:
            retry_after = rr.headers.get("Retry-After")
            try:
                wait = min(float(retry_after), 15.0) if retry_after else delay
            except Exception:
                wait = delay
            time.sleep(wait)
            delay = min(delay * 2, 8.0)
            last_err = f"429 {rr.text}"
            continue
        if 500 <= rr.status_code < 600:
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
            last_err = f"{rr.status_code} {rr.text}"
            continue
        if rr.status_code in (400, 404, 410):
            global _AGENT_ID, _AGENT_EXPIRES_AT
            with _AGENT_LOCK:
                _AGENT_ID = None
                _AGENT_EXPIRES_AT = 0
            time.sleep(0.5)
            last_err = f"{rr.status_code} {rr.text}"
            continue
        raise RuntimeError(f"Conversation start failed: {rr.status_code} {rr.text}")
    raise RuntimeError(f"Conversation start failed after retries: {last_err or 'unknown error'}")

def agent_generate_image(prompt: str, size="1024x1024") -> bytes:
    """
    Utilise l'API Agents + connecteur image_generation et renvoie les bytes PNG.
    Gère le rate-limit (429) et réutilise l'agent.
    """
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        raise ValueError("Clé API Mistral non trouvée (MISTRAL_API_KEY)")

    headers_json = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # 1) Agent réutilisé
    agent_id = _get_or_create_agent(headers_json)

    # 2) Lancer la conversation (avec respect de Retry-After sur 429)
    prompt_full = f"{prompt}\n\n[image_size={size}]"
    conv = _start_conversation_with_retry(agent_id, prompt_full, headers_json, max_retries=5)

    outputs = conv.get("outputs") or []
    if not outputs:
        raise RuntimeError("Agent n'a renvoyé aucun output.")

    # 3) Récupérer le file_id
    entry = outputs[-1]
    content = entry.get("content") or []
    file_id = None
    for c in content:
        if isinstance(c, dict):
            file_id = c.get("file_id") or c.get("fileId")
            if file_id:
                break
    if not file_id:
        raise RuntimeError(f"Aucun fichier image généré. Sortie: {entry}")

    # 4) Télécharger l'image
    file_headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/octet-stream",
    }
    file_resp = requests.get(
        f"https://api.mistral.ai/v1/files/{file_id}/content",
        headers=file_headers,
        timeout=120,
    )
    if file_resp.status_code == 429:
        ra = file_resp.headers.get("Retry-After", "1")
        # on encode le Retry-After dans l'exception pour que la route le propage
        raise RuntimeError(f"Image download rate-limited (429). Retry-After={ra}")
    if file_resp.status_code >= 400:
        raise RuntimeError(f"Image download failed: {file_resp.status_code} {file_resp.text}")

    return file_resp.content

# -------------------- Vision: description de monument --------------------
def describe_landmark_from_np(np_bgr) -> str:
    """
    Analyse l'image (Pixtral) et retourne un court paragraphe FR (3–5 phrases max) :
    - Si un monument/bâtiment/ouvrage est visible ET reconnu avec certitude (~80%+),
      donner le NOM OFFICIEL (Tour Eiffel, Taj Mahal, etc.), la ville/pays,
      le style/période, les dates de constructions et 1–2 éléments distinctifs.
    - Si incertain : NE PAS inventer de nom, décrire seulement le sujet principal
      et mentionner explicitement l’incertitude.
    Sortie : texte concis, naturel, sans listes ni code.
    """
    import cv2
    API_KEY = os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        raise ValueError("Clé API Mistral non trouvée (MISTRAL_API_KEY)")

    _, buf = cv2.imencode(".jpg", np_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_uri = {"url": f"data:image/jpeg;base64,{b64}"}

    system = (
        "Décris en français le sujet principal ou, si c’est un bâtiment/ouvrage, "
        "donne son nom officiel probable si il est celebre (tour Eiffel, Taj Mahal, etc.), la ville/pays, le style/période, les dates de construction et des éléments distinctifs. "
        "Reste concis (3–5 phrases), indique l’incertitude si besoin."
    )
    user_content = [
        {"type": "text", "text": "Analyse l’image et réponds uniquement en texte continu."},
        {"type": "image_url", "image_url": data_uri}
    ]

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 320,
        "model": "pixtral-12b-latest"
    }
    resp = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"Vision/subject error: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"].strip()