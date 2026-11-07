#!/usr/bin/env python3
"""
Web UI for the Vulnerability & Code Risk Analyzer.

Put this file and the `static/` folder next to Analyzer.py, then run:
    pip install flask scikit-learn joblib pandas
    python app.py
and open http://127.0.0.1:5000

All detection logic is reused from Analyzer.py (rules + trained models in ./models).
"""
import importlib.util
import io
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parent


def load_core():
    for name in ("Analyzer.py", "analyzer.py"):
        p = BASE / name
        if p.exists():
            spec = importlib.util.spec_from_file_location("analyzer_core", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("Analyzer.py not found next to app.py")


A = load_core()
A.MODEL_DIR = BASE / "models"

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB per request

MAX_FILES = 2000
MAX_ZIP_TOTAL = 80 * 1024 * 1024
SOURCE_PREVIEW_CHARS = 200_000
LANG_EXT = {"cpp": ".cpp", "python": ".py", "sql": ".sql"}

_model_cache = {"stamp": None, "models": {}}


# ------------------------------------------------------------------ models
def get_models():
    """Load trained models from ./models (reloads automatically when they change)."""
    mdir = A.MODEL_DIR
    files = sorted(mdir.glob("*.joblib")) if mdir.exists() else []
    stamp = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    if stamp != _model_cache["stamp"]:
        models = {}
        if files:
            import joblib
            for f in files:
                try:
                    obj = joblib.load(f)
                    for lang in obj["langs"]:
                        models[lang] = obj["model"]
                except Exception as e:  # damaged / incompatible model file
                    print(f"[!] Could not load {f.name}: {e}")
        _model_cache.update(stamp=stamp, models=models)
    return _model_cache["models"]


# ---------------------------------------------------------------- helpers
def decode(data: bytes):
    if b"\x00" in data[:2048]:
        return None
    return data.decode("utf-8", errors="ignore")


def skipped(name: str):
    return any(part in A.SKIP_DIRS for part in name.split("/"))


def analyze(name, text, models, min_sev, threshold, use_ml):
    lang = A.detect_lang(Path(name))
    lines = text.splitlines()
    findings = [x for x in A.rule_scan(lang, lines) if A.SEV_ORDER[x["severity"]] >= min_sev]
    findings.sort(key=lambda x: (-A.SEV_ORDER[x["severity"]], x["line"]))
    r = A.rule_score(findings)

    ml, regions = None, []
    if use_ml and lang in models:
        ml, hot = A.ml_scan(models[lang], lines)
        for p, start in hot:
            if p >= threshold:
                regions.append({
                    "start_line": start, "end_line": min(start + A.WINDOW - 1, len(lines)),
                    "probability": round(p, 2),
                    "explanation": "The AI model finds this block similar to known vulnerable code.",
                    "remediation": "Manually review input validation, bounds checks and memory handling here."})
    final = r if ml is None else 0.5 * ml + 0.5 * r
    return {
        "file": name, "language": lang,
        "risk_score": round(final * 100, 1), "risk_level": A.level(final * 100),
        "ai_score": None if ml is None else round(ml * 100, 1),
        "rule_score": round(r * 100, 1),
        "findings": findings, "ai_regions": regions,
        "line_count": len(lines),
        "source": text[:SOURCE_PREVIEW_CHARS],
    }


def collect_inputs():
    """Yield (name, text) from pasted code, uploaded files, and uploaded .zip archives."""
    code = request.form.get("code", "")
    if code.strip():
        lang = request.form.get("lang", "python")
        yield f"pasted_code{LANG_EXT.get(lang, '.py')}", code
        return

    count, total = 0, 0
    for f in request.files.getlist("files"):
        name = (f.filename or "").replace("\\", "/").lstrip("/")
        if name.lower().endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(f.read()))
            except zipfile.BadZipFile:
                continue
            for info in zf.infolist():  # read members in memory; nothing is extracted to disk
                n = info.filename.replace("\\", "/").lstrip("/")
                if info.is_dir() or skipped(n) or not A.detect_lang(Path(n)):
                    continue
                if info.file_size > A.MAX_FILE_BYTES:
                    continue
                total += info.file_size
                count += 1
                if count > MAX_FILES or total > MAX_ZIP_TOTAL:
                    return
                text = decode(zf.read(info))
                if text and text.strip():
                    yield f"{Path(name).stem}.zip/{n}", text
            continue
        if skipped(name) or not A.detect_lang(Path(name)):
            continue
        data = f.read()
        if len(data) > A.MAX_FILE_BYTES:
            continue
        count += 1
        if count > MAX_FILES:
            return
        text = decode(data)
        if text and text.strip():
            yield name, text


# ----------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    models = get_models()
    return jsonify({"models": sorted(models), "rules": len(A.RULES),
                    "languages": ["cpp", "python", "sql"]})


@app.post("/api/scan")
def scan():
    min_sev = A.SEV_ORDER.get(request.form.get("min_severity", "low"), 1)
    try:
        threshold = min(max(float(request.form.get("threshold", "0.5")), 0.0), 1.0)
    except ValueError:
        threshold = 0.5
    use_ml = request.form.get("use_ml", "1") == "1"
    models = get_models()

    results = [analyze(n, t, models, min_sev, threshold, use_ml) for n, t in collect_inputs()]
    if not results:
        return jsonify({"error": "No C/C++, Python or SQL code found. "
                                 "Paste code or upload .c/.cpp/.h/.py/.sql files or a .zip."}), 400

    results.sort(key=lambda x: x["risk_score"], reverse=True)
    avg = sum(r["risk_score"] for r in results) / len(results)
    project = round(0.6 * results[0]["risk_score"] + 0.4 * avg, 1)
    counts, kinds = {}, {}
    for r in results:
        for fd in r["findings"]:
            counts[fd["severity"]] = counts.get(fd["severity"], 0) + 1
            kinds[fd["kind"]] = kinds.get(fd["kind"], 0) + 1
    return jsonify({
        "project_risk": project, "risk_level": A.level(project),
        "files_scanned": len(results), "counts": counts, "kinds": kinds,
        "models_used": sorted(models) if use_ml else [], "files": results,
    })


@app.errorhandler(413)
def too_big(_):
    return jsonify({"error": "Upload too large (limit 60 MB)."}), 413


if __name__ == "__main__":
    print("Analyzer UI -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
