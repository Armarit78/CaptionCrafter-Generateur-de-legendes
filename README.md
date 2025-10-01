# CaptionCrafter — Générateur de légendes (Vision + OCR + Filtres + Agent Mistral)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green?logo=opencv)
![Pillow](https://img.shields.io/badge/Pillow-10.4-yellow)
![Mistral API](https://img.shields.io/badge/API-Mistral-orange)
![OCR](https://img.shields.io/badge/OCR-Tesseract-lightgrey)

**V2 — robuste, enrichie et interactive**  
Application **Python + Streamlit** qui génère **exactement 3 légendes Instagram contextualisées** à partir du contenu réel de l’image.  
Cette version ajoute :  
- **OCR (Tesseract, FR+EN)** intégré au contexte,  
- **Overlay texte amélioré** (contour noir, retour à la ligne auto, centrage),  
- **Prompt sécurisé pour monuments** (pas de nom si incertitude < 80%),  
- **Génération d’images** via l’Agent Mistral (`agent_generate_image`).  

---

## ✨ Fonctionnalités

- **Analyse d’image locale**
  - **OCR (Tesseract)** : extraction de texte intégré dans les légendes (tolérant, fallback = chaîne vide).
  - **Descripteurs visuels** : orientation, luminosité, contraste, couleurs dominantes (k-means).
  - **Pixtral** : compréhension d’image (scène, entités, actions).
- **Contrôles créatifs**
  - **Filtres** artistiques : Cartoon, Anime, Watercolor, Oil Paint, Pixelate, Pencil Sketch, Posterize, Cubism, etc.
  - **Texte overlay** : 
    - polices Hershey OpenCV,
    - taille, couleur, épaisseur configurables,
    - **anti-alias + contour noir automatique**,
    - **retour à la ligne** (par largeur ou nombre de caractères),
    - **centrage optionnel**.
- **Génération de légendes**
  - Prompt SYSTEM strict :  
    - interdit d’inventer des monuments,  
    - oblige 3 propositions de **~500 caractères chacune**,  
    - hashtags limités à 2.
  - Style paramétrable : *Inspiration, Humour, Émotion, Promotion, Citation*.
- **Description de monuments**
  - Si certitude ≥ 80% : nom officiel + contexte historique.  
  - Sinon : description visuelle, incertitude explicitée.
- **Agent Mistral (V2)**
  - Génération d’images à partir d’un prompt texte.
  - Téléchargement direct en PNG.

---

## 🧱 Pile technique

- UI : **Streamlit**
- Vision : **OpenCV**, **Pillow**, **Tesseract OCR** (optionnel)
- LLM/Vision : API **Mistral** (`requests`)
- Structure :
  - `main.py` — interface Streamlit (UI + flux complet)
  - `mistral_api.py` — appels API Mistral (texte, vision, agent image)
  - `image_utils.py` — OCR + analyse visuelle
  - `filters.py` — filtres + overlay texte

---

## 🚀 Installation rapide

### 1) Dépendances
```bash
git clone https://github.com/Armarit78/CaptionCrafter-Generateur-de-legendes.git
cd CaptionCrafter-Generateur-de-legendes
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) OCR (optionnel mais recommandé)
- **Windows**  
  ```powershell
  choco install tesseract
  ```
  Puis, si besoin, `.env` :  
  ```
  TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
  ```
- **macOS** : `brew install tesseract`  
- **Linux (Debian/Ubuntu)** : `sudo apt-get install -y tesseract-ocr`

### 3) API Mistral
Créer `.env` :  
```
MISTRAL_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 4) Lancer l’app
```bash
streamlit run main.py
```

---

## 📸 Aperçu visuel


<p align="center">
  <img src="images/Tour%20en%20flammes%20-%20image%20générée.png" alt="Tour en flammes - image générée" width="80%"><br>
  <em>1) Tour en flammes — image générée par l’Agent Mistral.</em>
</p>

<p align="center">
  <img src="images/Image%20modifiée%20avec%20texte%20ajouté.png" alt="Image modifiée avec texte ajouté" width="80%"><br>
  <em>2) Image modifiée avec texte ajouté.</em>
</p>

<p align="center">
  <img src="images/Capture%20d’écran%20du%20prompt%20dans%20CaptionCrafter.png" alt="Capture d’écran du prompt dans CaptionCrafter" width="80%"><br>
  <em>3) Capture d’écran du prompt dans CaptionCrafter.</em>
</p>

---

## 📦 Dépendances

Voir `requirements.txt` :

- `streamlit==1.36.0`
- `streamlit-drawable-canvas==0.9.3`
- `opencv-python-headless==4.10.0.84`
- `pillow==10.4.0`
- `requests==2.32.3`
- `python-dotenv==1.0.1`
- `pytesseract==0.3.10`
- `numpy==1.26.4`

---

## 👤 Auteur

Armarit78 — 2025
