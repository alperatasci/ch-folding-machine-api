import os
import io
import logging
from typing import Optional, Tuple, Dict, Any, List

import requests
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

import psycopg2
from psycopg2.pool import SimpleConnectionPool

# ----------------------
# Config
# ----------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
PGSSL = os.getenv("PGSSL", "require").strip()
FALLBACK_PDF_BASE = os.getenv("FALLBACK_PDF_BASE", "").strip()  # e.g. https://<app>/label
INCLUDE_PNG_URL = os.getenv("INCLUDE_PNG_URL", "0").strip() == "1"
PNG_TO_PDF_MODE = os.getenv("PNG_TO_PDF_MODE", "auto").strip()  # auto|force|off
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ch-folding-api")

app = FastAPI(title="CH Folding Machine API", version="1.0.0")

# Connection pool (optional; only if DATABASE_URL is set)
POOL: Optional[SimpleConnectionPool] = None

ORDER_QUERY = """
WITH qty AS (
  SELECT oi."OrderId", SUM(oi."Quantity") AS qty
  FROM "OrderItems" oi
  GROUP BY oi."OrderId"
),
vars AS (
  SELECT oi."OrderId",
         jsonb_agg(DISTINCT jsonb_build_object('name', oiv."FormattedName",
                                               'value', oiv."FormattedValue")) AS vars
  FROM "OrderItems" oi
  LEFT JOIN "OrderItemVariations" oiv ON oiv."OrderItemId" = oi."Id"
  GROUP BY oi."OrderId"
)
SELECT o."OrderNumber" AS barcode,
       COALESCE(os."LabelUrl", os."TrackingUrl", '') AS label_url,
       COALESCE(q.qty, 1) AS quantity,
       COALESCE(v.vars, '[]'::jsonb) AS vars
FROM "Orders" o
LEFT JOIN "OrderShipments" os ON os."OrderId" = o."Id"
LEFT JOIN qty q ON q."OrderId" = o."Id"
LEFT JOIN vars v ON v."OrderId" = o."Id"
WHERE o."OrderNumber" = %s
ORDER BY os."CreatedAt" DESC NULLS LAST
LIMIT 1;

"""

# ----------------------
# Models
# ----------------------
class ScanBody(BaseModel):
    barcode: str
    mac: Optional[str] = None

# ----------------------
# App lifecycle
# ----------------------
@app.on_event("startup")
def on_startup():
    global POOL
    if DATABASE_URL:
        # add sslmode if missing
        dsn = DATABASE_URL
        if "sslmode=" not in dsn and PGSSL:
            sep = "&" if "?" in dsn else "?"
            dsn = f"{dsn}{sep}sslmode={PGSSL}"
        POOL = SimpleConnectionPool(minconn=1, maxconn=5, dsn=dsn)
        log.info("DB pool created.")
    else:
        log.warning("DATABASE_URL not set. API will run in FALLBACK mode only.")

@app.on_event("shutdown")
def on_shutdown():
    global POOL
    if POOL:
        POOL.closeall()
        log.info("DB pool closed.")

# ----------------------
# Helpers
# ----------------------
def db_get_order(barcode: str) -> Optional[Dict[str, Any]]:
    """Fetch order info from DB; returns None if not found or DB not configured."""
    if not POOL:
        return None
    conn = None
    try:
        conn = POOL.getconn()
        with conn.cursor() as cur:
            cur.execute(ORDER_QUERY, (barcode,))
            row = cur.fetchone()
            if not row:
                return None
            # row: (barcode, label_url, quantity, vars_json)
            return {
                "barcode": row[0],
                "label_url": row[1] or "",
                "quantity": int(row[2] or 1),
                "vars": row[3] or [],
            }
    finally:
        if conn:
            POOL.putconn(conn)

def extract_var(vars_list: List[Dict[str, Any]], names: List[str]) -> str:
    """Find first matching variable name (case-insensitive)."""
    lname = [n.lower() for n in names]
    for item in vars_list or []:
        n = str(item.get("name", "")).lower()
        if n in lname:
            return str(item.get("value", "") or "")
    return ""

def build_urls(barcode: str, label_url: str) -> Tuple[str, Optional[str]]:
    """
    Decide the PDFUrl (required) and optional PNGUrl.
    - If label_url is a PDF and PNG_TO_PDF_MODE!=force: use it directly at /label/{barcode}.pdf (proxied).
    - If label_url is an image (png/jpg), we expose /label/{barcode}.pdf that will convert on demand.
    - If no label_url, fall back to FALLBACK_PDF_BASE/{barcode}.pdf (served by /label/... making a simple PDF).
    """
    # Always point machine to *our* /label/{barcode}.pdf endpoint
    base = FALLBACK_PDF_BASE.rstrip("/") if FALLBACK_PDF_BASE else ""
    if not base:
        # try to auto-derive from current app root (best effort for dev)
        base = "/label"
    pdf_url = f"{base}/{barcode}.pdf"

    png_url = None
    if INCLUDE_PNG_URL:
        # If the source is actually a PNG/JPG, and it's public, we can also surface it
        if label_url and label_url.lower().endswith((".png", ".jpg", ".jpeg")):
            png_url = label_url
        else:
            # Provide a local alias for consistency even if src is PDF
            # (Factory said PNG not needed; this is optional.)
            png_url = None
    return pdf_url, png_url

def http_get(url: str) -> bytes:
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.content

def image_to_pdf_bytes(img_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(img_bytes)) as im:
        # Convert to RGB and save as single-page PDF
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PDF")
        return out.getvalue()

def make_fallback_pdf(barcode: str) -> bytes:
    # Very simple one-page PDF with the barcode text (for dev/testing only)
    img = Image.new("RGB", (600, 900), "white")
    draw = ImageDraw.Draw(img)
    msg = f"NO LABEL FOUND\nBARCODE: {barcode}"
    try:
        # default PIL font
        draw.multiline_text((40, 100), msg, fill="black", spacing=10)
    except Exception:
        draw.text((40, 100), msg, fill="black")
    out = io.BytesIO()
    img.save(out, format="PDF")
    return out.getvalue()

def resolve_label(barcode: str) -> Dict[str, Any]:
    """
    Returns a dict with fields needed for response + serving /label/{barcode}.pdf.
    """
    row = db_get_order(barcode)
    label_url = (row or {}).get("label_url", "") if row else ""
    quantity = (row or {}).get("quantity", 1)
    vars_list = (row or {}).get("vars", [])

    # Try to map vars to Size/Color (best effort; safe if empty)
    size = extract_var(vars_list, ["Size", "SIZE", "size"])
    color = extract_var(vars_list, ["Color", "COLOR", "color", "Colorway"])

    pdf_url, png_url = build_urls(barcode, label_url)

    return {
        "barcode": barcode,
        "label_url": label_url,
        "quantity": int(quantity or 1),
        "order": "",   # not present in your schema; leave empty
        "color": color or "",
        "size": size or "",
        "pdf_url": pdf_url,
        "png_url": png_url,
    }

# ----------------------
# API Routes
# ----------------------
@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.get("/")
def root():
    return {"ok": True, "service": "ch-folding-machine-api"}

def _factory_payload(barcode: str) -> Dict[str, Any]:
    data = resolve_label(barcode)
    label_url = data["label_url"]
    payload = {
        "code": 1,
        "msg": "OK",
        "Quantity": data["quantity"],
        "Order": data["order"],
        "Color": data["color"],
        "Size": data["size"],
        "Barcode": barcode,
        "PDFUrl": data["pdf_url"],
    }
    if INCLUDE_PNG_URL and data.get("png_url"):
        payload["PNGUrl"] = data["png_url"]

    # If nothing in DB and no FALLBACK_PDF_BASE set, treat as not found
    if not label_url and not FALLBACK_PDF_BASE:
        payload.update({"code": 0, "msg": "Not found"})
    return payload

# Preferred by factory (GET)
@app.get("/api/scanbarcode/getpdfurl")
def get_pdfurl(barcode: str = Query(..., description="order number / barcode"),
               mac: Optional[str] = Query(None)):
    return JSONResponse(_factory_payload(barcode))

# Alias to the same behavior (GET)
@app.get("/api/printdata")
def get_pdfurl_alias(barcode: str = Query(...), mac: Optional[str] = Query(None)):
    return JSONResponse(_factory_payload(barcode))

# POST variants (just in case they use POST)
@app.post("/api/scanbarcode/getpdfurl")
def post_pdfurl(body: ScanBody = Body(...)):
    return JSONResponse(_factory_payload(body.barcode))

@app.post("/api/printdata")
def post_pdfurl_alias(body: ScanBody = Body(...)):
    return JSONResponse(_factory_payload(body.barcode))

# Serve the actual PDF the machine will download
@app.get("/label/{barcode}.pdf")
def serve_pdf(barcode: str):
    data = resolve_label(barcode)
    src = data["label_url"]

    # Decide how to produce PDF bytes
    # - If src is PDF AND not forcing conversion → proxy it.
    # - If src is image OR forcing conversion → convert to PDF.
    # - If no src → generate fallback PDF (dev only).
    try:
        if src:
            lower = src.lower()
            if lower.endswith(".pdf") and PNG_TO_PDF_MODE != "force":
                pdf_bytes = http_get(src)
            elif lower.endswith((".png", ".jpg", ".jpeg")) or PNG_TO_PDF_MODE in ("auto", "force"):
                img_bytes = http_get(src)
                pdf_bytes = image_to_pdf_bytes(img_bytes)
            else:
                # Unknown extension, try as PDF first
                try:
                    pdf_bytes = http_get(src)
                except Exception:
                    # last resort: try reading as image then convert
                    img_bytes = http_get(src)
                    pdf_bytes = image_to_pdf_bytes(img_bytes)
        else:
            # No DB label_url; fallback generated PDF
            pdf_bytes = make_fallback_pdf(barcode)
    except requests.HTTPError as e:
        log.warning(f"Upstream fetch failed ({src}): {e}")
        # give a clear machine-friendly 404:
        raise HTTPException(status_code=404, detail="PDF source not reachable")
    except Exception as e:
        log.exception("Error producing PDF")
        raise HTTPException(status_code=500, detail="PDF conversion error")

    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf")
