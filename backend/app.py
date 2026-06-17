
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ExifTags, ImageStat, ImageFilter
from pathlib import Path
from urllib.parse import quote_plus
import hashlib, io, os, json, datetime, requests, re

app = Flask(__name__)
CORS(app)

TAG_MAP = ExifTags.TAGS
GPS_TAG_MAP = ExifTags.GPSTAGS

def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def md5_bytes(data): return hashlib.md5(data).hexdigest()

def safe_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)

def rational_to_float(x):
    try: return float(x)
    except Exception:
        try: return x[0] / x[1]
        except Exception: return None

def gps_to_decimal(values, ref):
    try:
        d, m, s = rational_to_float(values[0]), rational_to_float(values[1]), rational_to_float(values[2])
        if d is None or m is None or s is None: return None
        dec = d + m/60 + s/3600
        if ref in ["S","W"]: dec = -dec
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
    try: raw = image.getexif()
    except Exception: raw = None
    if not raw: return exif_out, gps_out

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
    if brightness < 45: notes.append("immagine molto scura")
    elif brightness > 215: notes.append("immagine molto chiara/sovraesposta")
    if contrast < 25: notes.append("contrasto basso")
    if sharpness < 8: notes.append("possibile immagine poco nitida o sfocata")
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

def unique(seq):
    out, seen = [], set()
    for x in seq:
        x = str(x).strip()
        if x and x.lower() not in seen:
            out.append(x)
            seen.add(x.lower())
    return out

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

def google_url(q):
    return "https://www.google.com/search?q=" + quote_plus(q)

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
            {"category":"username", "value":user, "name":"GitHub", "url":f"https://github.com/{clean}"}
        ]
    for plate in entities.get("possible_plates", []):
        links += [
            {"category":"possibile_targa", "value":plate, "name":"Google", "url":google_url(f'"{plate}"')},
            {"category":"possibile_targa", "value":plate, "name":"Subito", "url":google_url(f'site:subito.it "{plate}"')},
            {"category":"possibile_targa", "value":plate, "name":"Autoscout", "url":google_url(f'site:autoscout24.it "{plate}"')}
        ]
    return links

def entity_queries(entities):
    queries = []
    for phone in entities.get("phones", []):
        queries += [
            {"type":"telefono", "value":phone, "query":f'"{phone}"'},
            {"type":"telefono-facebook", "value":phone, "query":f'site:facebook.com "{phone}"'},
            {"type":"telefono-instagram", "value":phone, "query":f'site:instagram.com "{phone}"'},
            {"type":"telefono-subito", "value":phone, "query":f'site:subito.it "{phone}"'}
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
            {"type":"username-tiktok", "value":user, "query":f'site:tiktok.com "{clean}"'},
            {"type":"username-x", "value":user, "query":f'site:x.com "{clean}" OR site:twitter.com "{clean}"'}
        ]
    for plate in entities.get("possible_plates", []):
        queries += [
            {"type":"possibile-targa", "value":plate, "query":f'"{plate}"'},
            {"type":"possibile-targa-web", "value":plate, "query":f'"{plate}" auto OR veicolo OR targa'}
        ]
    for url in entities.get("urls", []):
        queries.append({"type":"url", "value":url, "query":f'"{url}"'})
    for domain in entities.get("domains", []):
        queries.append({"type":"dominio", "value":domain, "query":f'site:{domain}'})
    return queries

def investigative_profile(entities, gps, ocr, exif):
    counts = {k: len(v) for k, v in entities.items() if isinstance(v, list)}
    score = 0
    score += counts.get("phones", 0) * 20
    score += counts.get("emails", 0) * 20
    score += counts.get("usernames", 0) * 15
    score += counts.get("urls", 0) * 10
    score += counts.get("domains", 0) * 10
    score += counts.get("possible_plates", 0) * 25
    score += counts.get("possible_fiscal_codes", 0) * 30
    if gps.get("latitude") is not None: score += 15
    if ocr.get("text"): score += 10
    if exif.get("Software"): score += 5
    score = min(score, 100)
    priority = "HIGH" if score >= 70 else "MEDIUM" if score >= 35 else "LOW"
    actions = []
    if counts.get("phones", 0): actions += ["Verifica telefono su fonti aperte", "Controlla eventuali profili collegati al numero"]
    if counts.get("emails", 0): actions += ["Verifica email su motori di ricerca", "Controlla eventuali data breach pubblici"]
    if counts.get("usernames", 0): actions += ["Verifica username sui social principali"]
    if counts.get("possible_plates", 0): actions += ["Verifica manualmente possibile targa", "Controlla errori OCR su caratteri simili"]
    if gps.get("latitude") is not None: actions += ["Verifica coordinate su mappa", "Confronta luogo con contesto immagine"]
    if ocr.get("text"): actions += ["Rileggi manualmente il testo OCR", "Confronta numeri/nomi estratti con l'immagine originale"]
    actions += ["Esegui reverse image search", "Documenta fonte, data, hash e passaggi eseguiti"]
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
    stem = Path(filename).stem
    queries = []
    if stem:
        queries += [f'"{stem}"', f'site:facebook.com "{stem}"', f'site:instagram.com "{stem}"', f'site:x.com "{stem}" OR site:twitter.com "{stem}"']
    make, model = str(exif.get("Make","")).strip(), str(exif.get("Model","")).strip()
    if make or model: queries.append(f'"{(make+" "+model).strip()}"')
    dt = exif.get("DateTimeOriginal") or exif.get("DateTime")
    if dt: queries.append(f'"{dt}"')
    if gps.get("latitude") is not None and gps.get("longitude") is not None:
        queries.append(f'"{gps["latitude"]}" "{gps["longitude"]}"')
    if ocr_text:
        words = " ".join(ocr_text.split()[:8])
        if words: queries.append(f'"{words}"')
    for q in entity_queries(entities):
        queries.append(q["query"])
    for key in ["luogo","telecamera","veicolo","targa","abbigliamento","oggetti","note"]:
        val = str(manual.get(key,"")).strip()
        if val: queries.append(f'"{val}"')
    queries.append(f'"{filename}"')
    return unique(queries)

def checklist(exif, gps, ocr, manual, entities):
    items = [
        "Conservare copia originale e hash SHA256 per catena di custodia digitale.",
        "Verificare data/ora di acquisizione e fonte del file originale.",
        "Usare reverse image search manuale per eventuali copie online.",
        "Non usare il sistema per identificazione automatica o riconoscimento facciale."
    ]
    if gps.get("latitude") is not None:
        items.append("GPS presente: verificare coerenza del luogo su mappa.")
    else:
        items.append("GPS assente: usare luogo/telecamera dichiarati o ricostruzione manuale.")
    if ocr.get("text"):
        items.append("OCR utile: verificare targhe, insegne, numeri, orari o parole estratte.")
    if entities.get("phones"):
        items.append("Telefono individuato: effettuare ricerche OSINT solo su fonti pubbliche/autorizzate.")
    if entities.get("emails") or entities.get("usernames"):
        items.append("Identificativi online individuati: verificare corrispondenze con più fonti.")
    if entities.get("possible_plates"):
        items.append("Possibile targa individuata: verificare manualmente, OCR può commettere errori.")
    return items

@app.get("/")
def home():
    return jsonify({"software":"IVAN-OSINT Investigativo v5 Backend","status":"online","features":["epieos_like_links","investigative_profile","entities","ocr","exif","gps"]})

@app.get("/health")
def health():
    return jsonify({"ok": True, "time": datetime.datetime.utcnow().isoformat()+"Z"})

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
        try: manual = json.loads(raw_manual)
        except Exception: manual = {}

    exif, gps = extract_exif(image)
    quality = image_quality(image)
    ocr = ocr_space(data, filename)
    entities = extract_entities(ocr.get("text",""), filename, exif, manual)
    serpapi_on = bool(os.environ.get("SERPAPI_KEY","").strip())
    profile = investigative_profile(entities, gps, ocr, exif)

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
    if not exif: warnings.append("Nessun EXIF trovato. Normale per social, WhatsApp, screenshot, CCTV o editor.")
    if gps.get("latitude") is None: warnings.append("Nessun GPS trovato nei metadati.")
    if not ocr.get("enabled"): warnings.append("OCR non attivo: aggiungere OCR_SPACE_API_KEY su Render se serve.")
    if not serpapi_on: warnings.append("SERPAPI_KEY non presente: reverse image automatico non attivo.")

    return jsonify({
        "software":"IVAN-OSINT Investigativo v5",
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "case_manual_fields": manual,
        "investigative_profile": profile,
        "technical": technical,
        "quality": quality,
        "exif": exif,
        "gps": gps,
        "ocr": ocr,
        "entities": entities,
        "entity_queries": entity_queries(entities),
        "epieos_like_links": epieos_like_links(entities),
        "api_status": {"ocr_space":{"enabled": bool(os.environ.get("OCR_SPACE_API_KEY","").strip())}, "serpapi":{"enabled": serpapi_on}},
        "queries": build_queries(filename, exif, gps, ocr.get("text",""), manual, entities),
        "reverse_image_sources": reverse_sources(),
        "investigative_checklist": checklist(exif, gps, ocr, manual, entities),
        "warnings": warnings,
        "legal_note": "Uso previsto: analisi tecnica e OSINT su dati autorizzati. Nessun riconoscimento facciale."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","5000")))
