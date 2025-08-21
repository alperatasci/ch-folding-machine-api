import io
import os
import re
import json
import time
import math
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests
from PIL import Image, ImageOps

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

from fastapi import FastAPI, Query, Body, Response
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# App / config
# -----------------------------------------------------------------------------
app = FastAPI()
log = logging.getLogger("uvicorn.error")

DATABASE_URL = os.getenv("DATABASE_URL", "")
FORCE_PG_SSL = os.getenv("PGSSL", "").strip() not in ("", "0", "false", "False", "no", "NO")
FALLBACK_PDF_BASE = os.getenv("FALLBACK_PDF_BASE", "").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "12"))
PNG_TO_PDF_DPI = int(os.getenv("PNG_TO_PDF_DPI", "300"))
CACHE_DIR = "/tmp/label-cache"
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

DB_POOL: Optional[SimpleConnectionPool] = None

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _dsn_with_ssl(db_url: str) -> str:
    """
    Ensure sslmode=require if requested or missing. Works with postgres:// and postgresql:// URLs.
    """
    try:
        u = urlparse(db_url)
        if u.scheme.startswith("postgres"):
            qs = dict(parse_qsl(u.query, keep_blank_values=True))
            if FORCE_PG_SSL and qs.get("sslmode") != "require":
                qs["sslmode"] = "require"
                u = u._replace(query=urlencode(qs))
                return urlunparse(u)
    except Exception:
        pass
    return db_url

def _open_db_conn():
    global DB_POOL
    if not DATABASE_URL:
        return None
    if DB_POOL is None:
        dsn = _dsn_with_ssl(DATABASE_URL)
        # Dict cursor for column access by name
        DB_POOL = SimpleConnectionPool(
            minconn=1,
            maxconn=8,
            dsn=dsn,
            cursor_factory=psycopg2.extras.DictCursor,
        )
        log.info("DB pool initialized.")
    return DB_POOL

def _get_conn():
    pool = _open_db_conn()
    if not pool:
        return None
    return pool.getconn()

def _put_conn(conn):
    if DB_POOL and conn:
        DB_POOL.putconn(conn)

def _infer_is_pdf(url: str) -> bool:
    u = url.lower()
    if ".pdf" in u:
        return True
    return False

def _vars_to_lookup(vars_arr: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in vars_arr or []:
        name = str(it.get("name", "")).strip()
        val = it.get("value", "")
        if not name:
            continue
        out[name.lower()] = val
    return out

def _compose_label_pdf_from_image(img_bytes: bytes) -> bytes:
    """
    Convert PNG/JPG bytes to a single-page PDF sized to 4x6 inches at PNG_TO_PDF_DPI.
    """
    with Image.open(io.BytesIO(img_bytes)) as im:
        # Ensure RGB (remove alpha on white background)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")

        # Target canvas 4x6 in at DPI
        target_w = int(4 * PNG_TO_PDF_DPI)
        target_h = int(6 * PNG_TO_PDF_DPI)

        # Letterbox/pad to fit while preserving aspect ratio
        im = ImageOps.contain(im, (target_w, target_h))
        canvas = Image.new("RGB", (target_w, target_h), "white")
        x = (target_w - im.width) // 2
        y = (target_h - im.height) // 2
        canvas.paste(im, (x, y))

        out = io.BytesIO()
        canvas.save(out, format="PDF", resolution=float(PNG_TO_PDF_DPI))
        return out.getvalue()

def _cache_path(barcode: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", barcode)
    return os.path.join(CACHE_DIR, f"{safe}.pdf")

def _read_cache(path: str) -> Optional[bytes]:
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    except Exception as e:
        log.warning(f"cache read failed: {e}")
    return None

def _write_cache(path: str, data: bytes) -> None:
    try:
        with open(path, "wb") as f:
            f.write(data)
    except Exception as e:
        log.warning(f"cache write failed: {e}")

# -----------------------------------------------------------------------------
# DB query
# -----------------------------------------------------------------------------
SQL_LABEL = """
with item_vars as (
  select oi."OrderId",
         json_agg(json_build_object('name', oiv."FormattedName", 'value', oiv."FormattedValue")) as vars,
         sum(oi."Quantity") as qty
  from "OrderItems" oi
  left join "OrderItemVariations" oiv on oiv."OrderItemId" = oi."Id"
  group by oi."OrderId"
)
select o."OrderNumber" as barcode,
       coalesce(os."LabelUrl", os."TrackingUrl", '') as label_url,
       coalesce(iv.qty, 1) as quantity,
       coalesce(iv.vars, '[]'::json) as vars
from "Orders" o
left join "OrderShipments" os on os."OrderId" = o."Id"
left join item_vars iv on iv."OrderId" = o."Id"
where o."OrderNumber" = %s
order by os."CreatedAt" desc nulls last
limit 1;
"""

def db_fetch_order(barcode: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    if conn is None:
        log.error("DATABASE_URL not configured; cannot fetch from DB.")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(SQL_LABEL, (barcode,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "barcode": row["barcode"],
                "label_url": row["label_url"] or "",
                "quantity": int(row["quantity"] or 1),
                "vars": row["vars"] or [],
            }
    finally:
        _put_conn(conn)

# -----------------------------------------------------------------------------
# API routes
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

def _make_payload(
    barcode: str, prefer: Optional[str] = None
) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (http_status, payload_dict)
    payload follows factory schema.
    """
    if not barcode:
        return 400, {"code": 0, "msg": "barcode required"}

    rec = db_fetch_order(barcode)
    if not rec:
        return 200, {"code": 0, "msg": "Not found"}

    label_url = rec.get("label_url") or ""
    qty = int(rec.get("quantity") or 1)
    vars_arr = rec.get("vars") or []
    var_map = _vars_to_lookup(vars_arr)

    color = var_map.get("color", "") or var_map.get("colour", "")
    size = var_map.get("size", "")
    order_code = var_map.get("order", "")

    # Decide final PDF URL
    if label_url and _infer_is_pdf(label_url):
        pdf_url = label_url
        png_url = ""
    elif label_url:
        # Image => expose local PDF proxy for machines
        if FALLBACK_PDF_BASE:
            pdf_url = f"{FALLBACK_PDF_BASE}/{barcode}.pdf"
        else:
            # No base configured; fall back to image (not recommended by factory)
            pdf_url = f"/label/{barcode}.pdf"
        png_url = label_url
    else:
        # No url in DB at all
        return 200, {"code": 0, "msg": "Not found"}

    payload: Dict[str, Any] = {
        "code": 1,
        "msg": "OK",
        "Quantity": qty,
        "Order": order_code,
        "Color": color,
        "Size": size,
        "Barcode": barcode,
        "PDFUrl": pdf_url,
    }

    # For your own testing: include PNGUrl when available or explicitly requested
    if png_url and (prefer == "png" or os.getenv("INCLUDE_PNG_URL", "0") == "1"):
        payload["PNGUrl"] = png_url

    return 200, payload

def _extract_barcode_and_prefer(
    barcode: Optional[str], mac: Optional[str], prefer: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    # Factory only cares about barcode. We accept mac but ignore it.
    bc = (barcode or "").strip()
    pf = (prefer or "").strip().lower()
    if pf not in ("", "png", "pdf"):
        pf = ""
    return bc, pf

@app.get("/api/scanbarcode/getpdfurl")
def scanbarcode_getpdfurl(
    barcode: Optional[str] = Query(default=None),
    mac: Optional[str] = Query(default=None),
    prefer: Optional[str] = Query(default=None, description="png|pdf (debug)"),
):
    bc, pf = _extract_barcode_and_prefer(barcode, mac, prefer)
    status, payload = _make_payload(bc, pf)
    return JSONResponse(status_code=status, content=payload)

@app.post("/api/scanbarcode/getpdfurl")
def scanbarcode_getpdfurl_post(body: Dict[str, Any] = Body(default={})):
    bc = str(body.get("barcode", "") or "").strip()
    prefer = str(body.get("prefer", "") or "").strip().lower()
    status, payload = _make_payload(bc, prefer)
    return JSONResponse(status_code=status, content=payload)

# Backward-compat alias that behaves the same
@app.get("/api/printdata")
@app.post("/api/printdata")
def printdata_alias(
    barcode: Optional[str] = Query(default=None),
    mac: Optional[str] = Query(default=None),
    prefer: Optional[str] = Query(default=None),
    body: Optional[Dict[str, Any]] = Body(default=None),
):
    if body and not barcode:
        barcode = str(body.get("barcode", "") or "")
        prefer = str((body.get("prefer") or "")).lower()
    bc, pf = _extract_barcode_and_prefer(barcode, mac, prefer)
    status, payload = _make_payload(bc, pf)
    return JSONResponse(status_code=status, content=payload)

# -----------------------------------------------------------------------------
# PDF proxy/converter
# -----------------------------------------------------------------------------
@app.get("/label/{barcode}.pdf")
def label_pdf(barcode: str):
    """
    For a given barcode:
    - If DB URL is already a PDF, proxy it back (no redirect).
    - If DB URL is image, convert to 4x6in PDF and cache.
    """
    if not barcode:
        return Response(status_code=404, content=b"Not Found", media_type="text/plain")

    cache_file = _cache_path(barcode)
    cached = _read_cache(cache_file)
    if cached:
        return Response(content=cached, media_type="application/pdf")

    rec = db_fetch_order(barcode)
    if not rec:
        return Response(status_code=404, content=b"Not Found", media_type="text/plain")

    label_url = rec.get("label_url") or ""
    if not label_url:
        return Response(status_code=404, content=b"Not Found", media_type="text/plain")

    try:
        r = requests.get(label_url, timeout=REQUEST_TIMEOUT, stream=True)
        r.raise_for_status()
        content_type = (r.headers.get("content-type") or "").lower()

        # If it's a PDF already, stream it back
        if _infer_is_pdf(label_url) or "application/pdf" in content_type:
            blob = r.content if hasattr(r, "content") else r.raw.read()
            _write_cache(cache_file, blob)
            return Response(content=blob, media_type="application/pdf")

        # Otherwise treat as an image and convert
        blob = r.content if hasattr(r, "content") else r.raw.read()
        pdf_bytes = _compose_label_pdf_from_image(blob)
        _write_cache(cache_file, pdf_bytes)
        return Response(content=pdf_bytes, media_type="application/pdf")
    except Exception as e:
        log.exception(f"label proxy/convert failed for {barcode}: {e}")
        return Response(status_code=502, content=b"Upstream fetch/convert failed", media_type="text/plain")

# -----------------------------------------------------------------------------
# Startup / shutdown
# -----------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    try:
        if DATABASE_URL:
            _open_db_conn()
    except Exception as e:
        # Don't crash the pod if DB is temporarily unavailable; API will return Not found
        log.error(f"DB pool init failed: {e}")

@app.on_event("shutdown")
def on_shutdown():
    global DB_POOL
    try:
        if DB_POOL:
            DB_POOL.closeall()
            DB_POOL = None
            log.info("DB pool closed.")
    except Exception as e:
        log.warning(f"DB pool close failed: {e}")
