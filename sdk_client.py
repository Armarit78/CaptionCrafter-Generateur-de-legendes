"""
Mini SDK client pour l'API CaptionCrafter (FastAPI).

Endpoints couverts :
- POST /analyze        -> analyse visuelle + OCR
- POST /captions       -> 3 légendes Instagram
- POST /filter         -> image filtrée (PNG)
- POST /agent-image    -> image générée par l'Agent Mistral (PNG)
- POST /entities       -> scène + entités + actions
- POST /landmark       -> description du monument/élément principal

Usage CLI (exemples) :
    python sdk_client.py analyze --file path/to/img.jpg
    python sdk_client.py captions --file img.jpg --style "Inspiration" --overlay "Hello"
    python sdk_client.py filter --file img.jpg --filter "Cartoon" --out out.png
    python sdk_client.py agent-image --prompt "une fusée" --size 1024x1024 --out rocket.png
    python sdk_client.py entities --file img.jpg
    python sdk_client.py landmark --file img.jpg
"""

from __future__ import annotations
import os
import sys
import argparse
import json
from typing import Dict, Any, Optional

import requests


class APIError(RuntimeError):
    def __init__(self, status_code: int, text: str):
        super().__init__(f"HTTP {status_code}: {text}")
        self.status_code = status_code
        self.text = text


class CaptionCrafterClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Optionnel: en-têtes communs
        self.headers = {
            "Accept": "application/json",
        }

    # ---------------------------
    def _check(self, resp: requests.Response) -> requests.Response:
        if not resp.ok:
            # Tente d'afficher l'erreur lisible si JSON; expose Retry-After si 429
            try:
                data = resp.json()
                retry_after = resp.headers.get("Retry-After")
                if resp.status_code == 429 and retry_after:
                    payload = {"detail": data.get("detail", "rate limited"), "retry_after": retry_after}
                    raise APIError(resp.status_code, json.dumps(payload, ensure_ascii=False))
                raise APIError(resp.status_code, json.dumps(data, ensure_ascii=False))
            except Exception:
                raise APIError(resp.status_code, resp.text)
        return resp

    # ---------------------------
    # Endpoints
    # ---------------------------
    def analyze_image(self, file_path: str) -> Dict[str, Any]:
        """
        POST /analyze -> { visual: {.}, ocr_text: "." }
        """
        url = f"{self.base_url}/analyze"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/*")}
            resp = requests.post(url, files=files, headers=self.headers, timeout=self.timeout)
        self._check(resp)
        return resp.json()

    def generate_captions(
        self,
        file_path: str,
        style: str = "Inspiration",
        overlay_text: str = ""
    ) -> Dict[str, Any]:
        """
        POST /captions -> {captions: [str, str, str]}
        """
        url = f"{self.base_url}/captions"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/*")}
            data = {"style": style, "overlay_text": overlay_text}
            resp = requests.post(url, files=files, data=data, headers=self.headers, timeout=self.timeout)
        self._check(resp)
        return resp.json()

    def apply_filter(
        self,
        file_path: str,
        filter_type: str,
        text: str = "",
        pos_x: int = 50,
        pos_y: int = 50,
        out_path: Optional[str] = None
    ) -> bytes:
        """
        POST /filter -> renvoie des bytes PNG.
        Si out_path est renseigné, écrit le fichier et renvoie aussi les bytes.
        """
        url = f"{self.base_url}/filter"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/*")}
            data = {
                "filter_type": filter_type,
                "text": text,
                "pos_x": str(pos_x),
                "pos_y": str(pos_y),
            }
            resp = requests.post(url, files=files, data=data, headers=self.headers, timeout=self.timeout)
        self._check(resp)
        content = resp.content
        if out_path:
            with open(out_path, "wb") as w:
                w.write(content)
        return content

    def agent_image(self, prompt: str, size: str = "1024x1024", out_path: Optional[str] = None) -> bytes:
        """
        POST /agent-image -> renvoie des bytes PNG.
        Si out_path est renseigné, écrit le fichier et renvoie aussi les bytes.

        ⚠️ Nécessite la variable d'env MISTRAL_API_KEY côté serveur (et clé valide).
        """
        url = f"{self.base_url}/agent-image"
        data = {"prompt": prompt, "size": size}
        resp = requests.post(url, data=data, headers=self.headers, timeout=self.timeout)
        self._check(resp)
        content = resp.content
        if out_path:
            with open(out_path, "wb") as w:
                w.write(content)
        return content

    def entities(self, file_path: str) -> Dict[str, Any]:
        """
        POST /entities -> {scene, entities[], actions[], raw}
        """
        url = f"{self.base_url}/entities"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/*")}
            resp = requests.post(url, files=files, headers=self.headers, timeout=self.timeout)
        self._check(resp)
        return resp.json()

    def landmark(self, file_path: str) -> Dict[str, Any]:
        """
        POST /landmark -> {landmark_description: "."}
        """
        url = f"{self.base_url}/landmark"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "image/*")}
            resp = requests.post(url, files=files, headers=self.headers, timeout=self.timeout)
        self._check(resp)
        return resp.json()

# ---------------------------
# CLI
# ---------------------------
def _cli():
    parser = argparse.ArgumentParser(description="CLI client pour CaptionCrafter API")
    parser.add_argument("--base-url", default=os.getenv("CAPTIONCRAFTER_BASE_URL", "http://127.0.0.1:8000"),
                        help="URL de base de l'API (défaut: http://127.0.0.1:8000)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout en secondes (défaut: 60)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # analyze
    p_an = sub.add_parser("analyze", help="Analyser une image")
    p_an.add_argument("--file", required=True, help="Chemin de l'image")

    # captions
    p_cap = sub.add_parser("captions", help="Générer 3 légendes Instagram")
    p_cap.add_argument("--file", required=True, help="Chemin de l'image")
    p_cap.add_argument("--style", default="Inspiration")
    p_cap.add_argument("--overlay", default="")

    # filter
    p_f = sub.add_parser("filter", help="Appliquer un filtre, renvoie un PNG")
    p_f.add_argument("--file", required=True)
    p_f.add_argument(
        "--filter",
        required=True,
        choices=[
            "Aucun",
            "Grayscale",
            "Blur",
            "Canny Edge",
            "Sepia",
            "Cartoon",
            "Anime",
            "Watercolor",
            "Oil Paint",
            "Cubism",
            "Pencil Sketch",
            "Posterize",
            "Pixelate",
        ],
        help="Type de filtre à appliquer"
    )
    p_f.add_argument("--text", default="", help="Texte à incruster")
    p_f.add_argument("--pos-x", type=int, default=50)
    p_f.add_argument("--pos-y", type=int, default=50)
    p_f.add_argument("--out", default="output.png", help="Fichier de sortie PNG")

    # agent-image
    p_ai = sub.add_parser("agent-image", help="Générer une image avec l'Agent Mistral")
    p_ai.add_argument("--prompt", required=True)
    p_ai.add_argument("--size", default="1024x1024")
    p_ai.add_argument("--out", default="agent.png")

    # entities
    p_en = sub.add_parser("entities", help="Scène & entités")
    p_en.add_argument("--file", required=True)

    # landmark
    p_lm = sub.add_parser("landmark", help="Description monument/élément principal")
    p_lm.add_argument("--file", required=True)

    args = parser.parse_args()
    client = CaptionCrafterClient(base_url=args.base_url, timeout=args.timeout)

    try:
        if args.cmd == "analyze":
            res = client.analyze_image(args.file)
            print(json.dumps(res, ensure_ascii=False, indent=2))

        elif args.cmd == "captions":
            res = client.generate_captions(args.file, style=args.style, overlay_text=args.overlay)
            print(json.dumps(res, ensure_ascii=False, indent=2))

        elif args.cmd == "filter":
            client.apply_filter(
                file_path=args.file,
                filter_type=args.filter,
                text=args.text,
                pos_x=args.pos_x,
                pos_y=args.pos_y,
                out_path=args.out,
            )
            print(f"✅ PNG écrit dans {args.out}")

        elif args.cmd == "agent-image":
            client.agent_image(prompt=args.prompt, size=args.size, out_path=args.out)
            print(f"✅ Image générée écrite dans {args.out}")

        elif args.cmd == "entities":
            res = client.entities(args.file)
            print(json.dumps(res, ensure_ascii=False, indent=2))

        elif args.cmd == "landmark":
            res = client.landmark(args.file)
            print(json.dumps(res, ensure_ascii=False, indent=2))

    except APIError as e:
        print(f"❌ APIError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    _cli()
