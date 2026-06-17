
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ExifTags, ImageStat, ImageFilter
from pathlib import Path
from urllib.parse import quote_plus
import hashlib, io, os, json, datetime, requests, re, base64

app = Flask(__name__)
CORS(app)

TAG_MAP = ExifTags.TAGS
GPS_TAG_MAP = ExifTags.GPSTAGS

# Optional AI modules. The backend stays alive even if Render Free cannot install heavy AI libs.
AI = {
    "ocr_space": True,
    "opencv": False,
    "mediapipe": False,
    "yolo": False,
    "clip": False
}

try:
    import cv2
    import numpy as np
    AI["opencv"] = True
except Exception:
    cv2 = None
    np = None

try:
    import mediapipe as mp
    AI["mediapipe"] = True
except Exception:
    mp = None

try:
    from ultralytics import YOLO
    AI["yolo"] = True
except Exception:
    YOLO = None

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    AI["clip"] = True
except Exception:
    torch = None
    CLIPProcessor = None
    CLIPModel = None

YOLO_MODEL = None
CLIP_MODEL = None
CLIP_PROCESSOR = None


def google_url(q): 
    return "https://www.google.com/search?q=" + quote_plus(q)

def sha256_bytes(data): 
    return hashlib.sha256(data).hexdigest()

def md5_bytes(data): 
    return hashlib.md5(data).hexdigest()

def unique(seq):
    out, seen = [], set()
    for x in seq:
        x = str(x).strip()
        if x and x.lower() not in seen:
            out.append(x)
            seen.add(x.lower())
    return out

def safe_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)

def rational_to_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return x[0] / x[1]
        except Exception:
            return None

def gps_to_decimal(values, ref):
    try:
        d, m, s = rational_to_float(values[0]), rational_to_float(values[1]), rational_to_float(values[2])
        if d is None or m is None or s is None:
            return None
        dec = d + m / 60 + s / 3600
        if ref in ["S", "W"]:
            dec = -dec
        return round(dec, 7)
    except Exception:
        return None

def average_hash(image, hash_size=8):
    img = image.convert("L").resize((hash_size, hash_size))
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hex(int(bits, 2))[2:].rjust(hash_size * hash_size // 4, "0")

def difference_hash(image, hash_size=8):
    img = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(img.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append("1" if left > right else "0")
    return hex(int("".join(bits), 2))[2:].rjust(hash_size * hash_size // 4, "0")

def extract_exif(image):
    exif_out, gps_raw = {}, {}
    gps_out = {"latitude": None, "longitude": None, "google_maps": None, "raw": {}}
    try:
        raw = image.getexif()
    except Exception:
        raw = None
    if not raw:
        return exif_out, gps_out

    for tag_id, value in raw.items():
        tag = TAG_MAP.get(tag_id, f"TAG_{tag_id}")
        if tag == "GPSInfo":
            try:
                for gps_id, gps_value in value.items():
                    gps_name = GPS_TAG_MAP.get(gps_id, f"GPS_{gps_id}")
                    gps_raw[gps_name] = safe_value(gps_value)
            except Exception:
                pass
        else:
            exif_out[tag] = safe_value(value)

    gps_out["raw"] = gps_raw
    if "GPSLatitude" in gps_raw and "GPSLatitudeRef" in gps_raw:
        gps_out["latitude"] = gps_to_decimal(gps_raw["GPSLatitude"], gps_raw["GPSLatitudeRef"])
    if "GPSLongitude" in gps_raw and "GPSLongitudeRef" in gps_raw:
        gps_out["longitude"] = gps_to_decimal(gps_raw["GPSLongitude"], gps_raw["GPSLongitudeRef"])

    if gps_out["latitude"] is not None and gps_out["longitude"] is not None:
        gps_out["google_maps"] = f"https://www.google.com/maps?q={gps_out['latitude']},{gps_out['longitude']}"
    return exif_out, gps_out

def image_quality(image):
    rgb = image.convert("RGB")
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = round(stat.mean[0], 2)
    contrast = round(stat.stddev[0], 2)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = round(ImageStat.Stat(edges).mean[0], 2)
    dominant = rgb.resize((1,1)).getpixel((0,0))
    notes = []
    if brightness < 45:
        notes.append("immagine molto scura")
    elif brightness > 215:
        notes.append("immagine molto chiara/sovraesposta")
    if contrast < 25:
        notes.append("contrasto basso")
    if sharpness < 8:
        notes.append("possibile immagine poco nitida o sfocata")
    return {
        "brightness_0_255": brightness,
        "contrast_estimate": contrast,
        "sharpness_estimate": sharpness,
        "dominant_color_rgb": dominant,
        "dominant_color_hex": "#{:02x}{:02x}{:02x}".format(*dominant),
        "quality_note": ", ".join(notes) if notes else "qualità tecnica apparentemente utilizzabile"
    }

def ocr_space(data, filename):
    api_key = os.environ.get("OCR_SPACE_API_KEY", "").strip()
    if not api_key:
        return {"enabled": False, "provider": "OCR.space", "text": "", "note": "OCR non attivo: manca OCR_SPACE_API_KEY su Render."}
    try:
        r = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (filename, data)},
            data={"apikey": api_key, "language": "ita", "isOverlayRequired": "false", "OCREngine": "2"},
            timeout=45
        )
        js = r.json()
        text = ""
        if js.get("ParsedResults"):
            text = "\n".join([x.get("ParsedText", "") for x in js["ParsedResults"]])
        return {"enabled": True, "provider": "OCR.space", "text": text.strip(), "raw_status": js.get("OCRExitCode"), "error": js.get("ErrorMessage")}
    except Exception as e:
        return {"enabled": True, "provider": "OCR.space", "text": "", "error": str(e)}

def extract_entities(text, filename="", exif=None, manual=None):
    exif = exif or {}
    manual = manual or {}
    corpus = "\n".join([
        text or "",
        filename or "",
        " ".join([str(v) for v in exif.values() if isinstance(v, (str, int, float))]),
        " ".join([str(v) for v in manual.values()])
    ])
    emails = re.findall(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', corpus)
    urls = re.findall(r'\b(?:https?://|www\.)[^\s<>"\']+', corpus)
    hashtags = re.findall(r'(?<!\w)#[A-Za-z0-9_]{2,50}\b', corpus)
    usernames = re.findall(r'(?<![\w@])@[A-Za-z0-9._]{3,40}\b', corpus)
    bare_users = re.findall(r'\b(?:username|user|ig|instagram|telegram|tiktok|x|twitter)\s*[:=@]?\s*([A-Za-z0-9._]{3,40})\b', corpus, flags=re.I)
    usernames += ["@" + u for u in bare_users if "@" not in u]
    raw_phones = re.findall(r'(?:(?:\+|00)\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?){2,5}\d{2,5}', corpus)
    phones = []
    for p in raw_phones:
        cleaned = re.sub(r'[^\d+]', '', p)
        digits = re.sub(r'\D', '', cleaned)
        if 8 <= len(digits) <= 15:
            phones.append(cleaned)
    plates = re.findall(r'\b[A-Z]{2}\s?\d{3}\s?[A-Z]{2}\b', corpus.upper())
    fiscal_codes = re.findall(r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b', corpus.upper())
    coordinates = re.findall(r'[-+]?\d{1,2}\.\d{4,}\s*,\s*[-+]?\d{1,3}\.\d{4,}', corpus)
    domains = re.findall(r'\b(?:[a-zA-Z0-9-]+\.)+(?:it|com|net|org|eu|gov|edu|info|io)\b', corpus)
    return {
        "phones": unique(phones),
        "emails": unique(emails),
        "urls": unique(urls),
        "domains": unique(domains),
        "usernames": unique(usernames),
        "hashtags": unique(hashtags),
        "possible_plates": unique(plates),
        "possible_fiscal_codes": unique(fiscal_codes),
        "coordinates_in_text": unique(coordinates),
        "device": unique([str(exif.get("Model", "")).strip()] if exif.get("Model") else []),
        "software": unique([str(exif.get("Software", "")).strip()] if exif.get("Software") else []),
        "datetime": unique([str(exif.get("DateTime", "")).strip()] if exif.get("DateTime") else [])
    }

def phone_enrichment(phone):
    digits = re.sub(r'\D', '', phone)
    out = {"input": phone, "digits": digits, "type_guess": None, "country_guess": None, "country_code": None, "international_format_guess": None, "notes": []}
    if phone.startswith("+39") or (len(digits) == 12 and digits.startswith("39")):
        national = digits[2:] if digits.startswith("39") else digits
        out.update({"country_guess":"Italia", "country_code":"+39", "international_format_guess":"+39"+national})
    elif len(digits) == 10 and digits.startswith("3"):
        out.update({"country_guess":"Italia", "country_code":"+39", "international_format_guess":"+39"+digits})
    elif digits.startswith("39") and len(digits) >= 11:
        out.update({"country_guess":"Italia", "country_code":"+39", "international_format_guess":"+"+digits})
    elif phone.startswith("+"):
        out.update({"country_code":"+" + digits[:2], "international_format_guess":"+"+digits, "country_guess":"da verificare"})
    else:
        out["international_format_guess"] = phone
        out["notes"].append("Paese non determinabile senza prefisso internazionale.")
    if out["country_guess"] == "Italia" and digits[-10:].startswith("3"):
        out["type_guess"] = "mobile italiano probabile"
    elif out["country_guess"] == "Italia":
        out["type_guess"] = "numero italiano probabile"
    else:
        out["type_guess"] = "numero da verificare"
    out["notes"].append("Arricchimento locale euristico: non conferma intestatario, operatore o validità reale.")
    return out

def enrich_entities(entities):
    return {
        "phones": [phone_enrichment(p) for p in entities.get("phones", [])],
        "emails": [{"input": e, "domain": e.split("@",1)[1].lower() if "@" in e else "", "provider_guess": "Google/Gmail" if e.endswith("@gmail.com") else "da verificare"} for e in entities.get("emails", [])],
        "usernames": [{"input": u, "normalized": u.lstrip("@"), "candidate_urls": {
            "instagram": f"https://www.instagram.com/{u.lstrip('@')}/",
            "tiktok": f"https://www.tiktok.com/@{u.lstrip('@')}",
            "x": f"https://x.com/{u.lstrip('@')}",
            "reddit": f"https://www.reddit.com/user/{u.lstrip('@')}",
            "github": f"https://github.com/{u.lstrip('@')}",
            "telegram": f"https://t.me/{u.lstrip('@')}"
        }} for u in entities.get("usernames", [])],
        "domains": [{"input": d, "google_dork": f"site:{d}", "wayback": f"https://web.archive.org/web/*/{d}"} for d in entities.get("domains", [])],
        "possible_plates": [{"input": p, "normalized": p.replace(" ",""), "note":"Possibile targa: verificare manualmente."} for p in entities.get("possible_plates", [])],
        "coordinates_in_text": [{"input": c, "google_maps": "https://www.google.com/maps?q=" + quote_plus(c)} for c in entities.get("coordinates_in_text", [])]
    }

def epieos_like_links(entities):
    links = []
    for phone in entities.get("phones", []):
        clean = re.sub(r'\D', '', phone)
        links += [
            {"category":"telefono", "value":phone, "name":"Google", "url":google_url(f'"{phone}"')},
            {"category":"telefono", "value":phone, "name":"Facebook", "url":google_url(f'site:facebook.com "{phone}"')},
            {"category":"telefono", "value":phone, "name":"Instagram", "url":google_url(f'site:instagram.com "{phone}"')},
            {"category":"telefono", "value":phone, "name":"Telegram", "url":f"https://t.me/+{clean}"},
            {"category":"telefono", "value":phone, "name":"WhatsApp", "url":f"https://wa.me/{clean}"},
            {"category":"telefono", "value":phone, "name":"Truecaller", "url":f"https://www.truecaller.com/search/it/{clean}"},
            {"category":"telefono", "value":phone, "name":"Sync.me", "url":f"https://sync.me/search/?number={clean}"}
        ]
    for email in entities.get("emails", []):
        links += [
            {"category":"email", "value":email, "name":"Google", "url":google_url(f'"{email}"')},
            {"category":"email", "value":email, "name":"LinkedIn", "url":google_url(f'site:linkedin.com "{email}"')},
            {"category":"email", "value":email, "name":"GitHub", "url":google_url(f'site:github.com "{email}"')},
            {"category":"email", "value":email, "name":"Gravatar", "url":f"https://en.gravatar.com/site/check/{email}"},
            {"category":"email", "value":email, "name":"HaveIBeenPwned", "url":"https://haveibeenpwned.com/"}
        ]
    for user in entities.get("usernames", []):
        clean = user.lstrip("@")
        links += [
            {"category":"username", "value":user, "name":"Google", "url":google_url(f'"{clean}"')},
            {"category":"username", "value":user, "name":"Instagram", "url":f"https://www.instagram.com/{clean}/"},
            {"category":"username", "value":user, "name":"TikTok", "url":f"https://www.tiktok.com/@{clean}"},
            {"category":"username", "value":user, "name":"X", "url":f"https://x.com/{clean}"},
            {"category":"username", "value":user, "name":"Reddit", "url":f"https://www.reddit.com/user/{clean}"},
            {"category":"username", "value":user, "name":"GitHub", "url":f"https://github.com/{clean}"},
            {"category":"username", "value":user, "name":"Telegram", "url":f"https://t.me/{clean}"}
        ]
    return links

def entity_queries(entities):
    queries = []
    for phone in entities.get("phones", []):
        queries += [
            {"type":"telefono", "value":phone, "query":f'"{phone}"'},
            {"type":"telefono-facebook", "value":phone, "query":f'site:facebook.com "{phone}"'},
            {"type":"telefono-instagram", "value":phone, "query":f'site:instagram.com "{phone}"'}
        ]
    for email in entities.get("emails", []):
        queries += [
            {"type":"email", "value":email, "query":f'"{email}"'},
            {"type":"email-linkedin", "value":email, "query":f'site:linkedin.com "{email}"'},
            {"type":"email-github", "value":email, "query":f'site:github.com "{email}"'}
        ]
    for user in entities.get("usernames", []):
        clean = user.lstrip("@")
        queries += [
            {"type":"username", "value":user, "query":f'"{clean}"'},
            {"type":"username-instagram", "value":user, "query":f'site:instagram.com "{clean}"'},
            {"type":"username-tiktok", "value":user, "query":f'site:tiktok.com "{clean}"'}
        ]
    return queries

def investigative_profile(entities, gps=None, ocr=None, exif=None):
    gps, ocr, exif = gps or {}, ocr or {}, exif or {}
    counts = {k: len(v) for k, v in entities.items() if isinstance(v, list)}
    score = 0
    score += counts.get("phones", 0) * 20
    score += counts.get("emails", 0) * 20
    score += counts.get("usernames", 0) * 15
    score += counts.get("urls", 0) * 10
    score += counts.get("domains", 0) * 10
    score += counts.get("possible_plates", 0) * 25
    score += counts.get("possible_fiscal_codes", 0) * 30
    if gps.get("latitude") is not None:
        score += 15
    if ocr.get("text"):
        score += 10
    if exif.get("Software"):
        score += 5
    score = min(score, 100)
    priority = "HIGH" if score >= 70 else "MEDIUM" if score >= 35 else "LOW"
    actions = []
    if counts.get("phones", 0): actions += ["Verifica telefono su fonti aperte", "Controlla eventuali profili collegati al numero"]
    if counts.get("emails", 0): actions += ["Verifica email su motori di ricerca", "Controlla eventuali data breach pubblici"]
    if counts.get("usernames", 0): actions += ["Verifica username sui social principali"]
    if counts.get("domains", 0): actions += ["Verifica dominio con WHOIS/DNS/Wayback"]
    if counts.get("possible_plates", 0): actions += ["Verifica manualmente possibile targa", "Controlla errori OCR su caratteri simili"]
    actions += ["Documenta fonte, data, hash e passaggi eseguiti"]
    return {"score": score, "priority": priority, "entity_counts": counts, "recommended_actions": unique(actions)}

def reverse_sources():
    return [
        {"name":"Google Lens","url":"https://lens.google.com/"},
        {"name":"Google Immagini","url":"https://images.google.com/"},
        {"name":"TinEye","url":"https://tineye.com/"},
        {"name":"Yandex Images","url":"https://yandex.com/images/"},
        {"name":"Bing Visual Search","url":"https://www.bing.com/images/search"},
    ]

def build_queries(filename, exif, gps, ocr_text, manual, entities):
    stem = Path(filename).stem if filename else ""
    queries = []
    if stem:
        queries += [f'"{stem}"', f'site:facebook.com "{stem}"', f'site:instagram.com "{stem}"']
    make, model = str(exif.get("Make","")).strip(), str(exif.get("Model","")).strip()
    if make or model:
        queries.append(f'"{(make+" "+model).strip()}"')
    if ocr_text:
        words = " ".join(ocr_text.split()[:8])
        if words:
            queries.append(f'"{words}"')
    for q in entity_queries(entities):
        queries.append(q["query"])
    return unique(queries)

# ---------------- AI NON IDENTIFICATIVE ----------------

def opencv_analysis(image):
    if not AI["opencv"]:
        return {"enabled": False, "note": "OpenCV non installato."}
    try:
        arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        face_count = 0
        boxes = []
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face_count += 1
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return {
            "enabled": True,
            "sharpness_laplacian": round(lap_var, 2),
            "face_count_non_identifying": face_count,
            "face_boxes": boxes,
            "note": "Rilevamento volto non identificativo: non riconosce identità."
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}

def blur_faces_base64(image, face_boxes):
    try:
        img = image.convert("RGB")
        for b in face_boxes or []:
            x, y, w, h = b["x"], b["y"], b["w"], b["h"]
            crop = img.crop((x, y, x+w, y+h)).filter(ImageFilter.GaussianBlur(radius=18))
            img.paste(crop, (x, y))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None

def mediapipe_analysis(image):
    if not AI.get["mediapipe"]:
        return {"enabled": False, "note": "MediaPipe non installato."}
    try:
        arr = np.array(image.convert("RGB"))
        face_detection_module = None
            
        try:
            import mediapipe as mp_local
            if hasattr(mp_local, "solutions") and hasattr(mp_local.solutions, "face_detection"):
                face_detection_module = mp_local.solutions.face_detection
        except Exception:
            pass
        if face_detection_module is None:
            try:
                from mediapipe.phyton.solutions import face_detection as face_detection_module
            except Exception:
                face_detection_module = None

        if face_detection_module is None:
            return {
                "enabled": True,
                "error": "MediaPipe installato ma face_detection non disponibile.",
                "note": "Backend operativo; usare Opencv come fallback."
            }
        with face_detection_module.FaceDetection(
            model_selection=1,
            min_detection_confidence=0,5
        ) as fd:
            results = fd.process(arr)
            detections = results.detections or []

           return {
               "enabled": True,
               "face_detections": len(detections),
               "note": "MediaPipe usato solo per rilevamento volto non identificativo."
           }

except Exception as e:
    return {"enabled": True, "error": str(e)}

def yolo_analysis(image):
    global YOLO_MODEL
    if not AI["yolo"]:
        return {"enabled": False, "note": "YOLO non installato."}
    try:
        if YOLO_MODEL is None:
            YOLO_MODEL = YOLO("yolov8n.pt")
        arr = np.array(image.convert("RGB"))
        results = YOLO_MODEL(arr, verbose=False)
        items = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [float(x) for x in box.xyxy[0]]
                items.append({"label": names.get(cls_id, str(cls_id)), "confidence": round(conf, 3), "box": xyxy})
        return {"enabled": True, "objects": items[:30], "note": "YOLO rileva oggetti/veicoli/persone come classi, non identifica individui."}
    except Exception as e:
        return {"enabled": True, "error": str(e)}

def clip_analysis(image):
    global CLIP_MODEL, CLIP_PROCESSOR
    if not AI["clip"]:
        return {"enabled": False, "note": "CLIP non installato."}
    try:
        if CLIP_MODEL is None:
            CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        labels = [
            "a person", "a car", "a motorcycle", "a bicycle", "a backpack",
            "a helmet", "a phone", "a street sign", "a building entrance",
            "a document with text", "a social media screenshot", "a CCTV frame"
        ]
        inputs = CLIP_PROCESSOR(text=labels, images=image.convert("RGB"), return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = CLIP_MODEL(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
        ranked = sorted([{"label": l, "score": round(p, 4)} for l, p in zip(labels, probs)], key=lambda x: x["score"], reverse=True)
        return {"enabled": True, "semantic_labels": ranked[:8], "note": "CLIP usato per descrivere scena/oggetti, non per identità facciale."}
    except Exception as e:
        return {"enabled": True, "error": str(e)}

def ai_suite(image):
    cv = opencv_analysis(image)
    return {
        "ai_status": AI,
        "opencv": cv,
        "mediapipe": mediapipe_analysis(image),
        "yolo": yolo_analysis(image),
        "clip": clip_analysis(image),
        "privacy_blur_preview": blur_faces_base64(image, cv.get("face_boxes", [])) if cv.get("face_boxes") else None,
        "safety_note": "AI non identificativa: nessuna ricerca facciale online, nessun riconoscimento identità."
    }

def text_osint_report(text, manual=None):
    manual = manual or {}
    entities = extract_entities(text, "", {}, manual)
    return {
        "software":"IVAN-OSINT Investigativo v9",
        "mode":"text_osint",
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "input_text": text,
        "case_manual_fields": manual,
        "investigative_profile": investigative_profile(entities),
        "entities": entities,
        "entity_enrichment": enrich_entities(entities),
        "entity_queries": entity_queries(entities),
        "epieos_like_links": epieos_like_links(entities),
        "queries": build_queries("", {}, {}, text, manual, entities),
        "legal_note": "Uso previsto: OSINT su dati autorizzati e fonti aperte. Le verifiche automatiche sono euristiche."
    }

@app.get("/")
def home():
    return jsonify({"software":"IVAN-OSINT Investigativo v9 Backend","status":"online","ai_status":AI,"features":["OCR","OpenCV optional","MediaPipe optional","YOLO optional","CLIP optional","text_osint","image_osint"]})

@app.get("/health")
def health():
    return jsonify({"ok": True, "time": datetime.datetime.utcnow().isoformat()+"Z", "ai_status": AI})

@app.post("/analyze-text")
def analyze_text():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    manual = payload.get("manual") or {}
    if not text:
        return jsonify({"error":"Inserisci un testo, numero, email, username, dominio o targa."}), 400
    return jsonify(text_osint_report(text, manual))

@app.post("/analyze-image")
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error":"Nessuna immagine ricevuta."}), 400
    f = request.files["image"]
    data = f.read()
    filename = f.filename or "image"
    if len(data) > 25 * 1024 * 1024:
        return jsonify({"error":"File troppo grande. Max 25 MB."}), 400
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as e:
        return jsonify({"error":f"Immagine non valida: {e}"}), 400

    manual = {}
    raw_manual = request.form.get("manual","")
    if raw_manual:
        try:
            manual = json.loads(raw_manual)
        except Exception:
            manual = {}

    exif, gps = extract_exif(image)
    quality = image_quality(image)
    ocr = ocr_space(data, filename)
    entities = extract_entities(ocr.get("text",""), filename, exif, manual)

    technical = {
        "filename": filename,
        "content_type": f.content_type,
        "size_kb": round(len(data)/1024, 2),
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
        "sha256": sha256_bytes(data),
        "md5": md5_bytes(data),
        "average_hash": average_hash(image),
        "difference_hash": difference_hash(image)
    }

    warnings = []
    if not exif:
        warnings.append("Nessun EXIF trovato. Normale per social, WhatsApp, screenshot, CCTV o editor.")
    if gps.get("latitude") is None:
        warnings.append("Nessun GPS trovato nei metadati.")

    return jsonify({
        "software":"IVAN-OSINT Investigativo v9",
        "mode":"image_osint",
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "case_manual_fields": manual,
        "investigative_profile": investigative_profile(entities, gps, ocr, exif),
        "technical": technical,
        "quality": quality,
        "exif": exif,
        "gps": gps,
        "ocr": ocr,
        "entities": entities,
        "entity_enrichment": enrich_entities(entities),
        "entity_queries": entity_queries(entities),
        "epieos_like_links": epieos_like_links(entities),
        "ai_suite": ai_suite(image),
        "api_status": {"ocr_space":{"enabled": bool(os.environ.get("OCR_SPACE_API_KEY","").strip())}, "serpapi":{"enabled": bool(os.environ.get("SERPAPI_KEY","").strip())}},
        "queries": build_queries(filename, exif, gps, ocr.get("text",""), manual, entities),
        "reverse_image_sources": reverse_sources(),
        "warnings": warnings,
        "legal_note": "Uso previsto: analisi tecnica e OSINT su dati autorizzati. Nessun riconoscimento facciale identificativo."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","5000")))
