
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ExifTags, ImageStat, ImageFilter
from pathlib import Path
import hashlib, io, os, json, datetime, requests

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

def reverse_sources():
    return [
        {"name":"Google Lens","url":"https://lens.google.com/"},
        {"name":"Google Immagini","url":"https://images.google.com/"},
        {"name":"TinEye","url":"https://tineye.com/"},
        {"name":"Yandex Images","url":"https://yandex.com/images/"},
        {"name":"Bing Visual Search","url":"https://www.bing.com/images/search"},
    ]

def build_queries(filename, exif, gps, ocr_text, manual):
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
    for key in ["luogo","telecamera","veicolo","targa","abbigliamento","oggetti","note"]:
        val = str(manual.get(key,"")).strip()
        if val: queries.append(f'"{val}"')
    queries.append(f'"{filename}"')
    return [q for q in queries if q and q != '""']

def checklist(exif, gps, ocr, manual):
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
    if manual.get("abbigliamento"):
        items.append("Abbigliamento indicato: può aiutare a collegare fotogrammi diversi senza biometria.")
    if manual.get("oggetti"):
        items.append("Oggetti distintivi indicati: verificare ricorrenza in altre riprese.")
    return items

@app.get("/")
def home():
    return jsonify({"software":"IVAN-OSINT Investigativo v3 Backend","status":"online","endpoints":["/health","/analyze-image"],"optional_env":["OCR_SPACE_API_KEY","SERPAPI_KEY"]})

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
    serpapi_on = bool(os.environ.get("SERPAPI_KEY","").strip())

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

    report = {
        "software":"IVAN-OSINT Investigativo v3",
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "case_manual_fields": manual,
        "technical": technical,
        "quality": quality,
        "exif": exif,
        "gps": gps,
        "ocr": ocr,
        "api_status": {"ocr_space":{"enabled": bool(os.environ.get("OCR_SPACE_API_KEY","").strip())}, "serpapi":{"enabled": serpapi_on}},
        "queries": build_queries(filename, exif, gps, ocr.get("text",""), manual),
        "reverse_image_sources": reverse_sources(),
        "investigative_checklist": checklist(exif, gps, ocr, manual),
        "warnings": warnings,
        "legal_note": "Uso previsto: analisi tecnica e OSINT su dati autorizzati. Nessun riconoscimento facciale."
    }
    return jsonify(report)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","5000")))
