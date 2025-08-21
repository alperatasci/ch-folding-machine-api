import io, os, time, json, logging, pathlib, re
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse

app = FastAPI()
log = logging.getLogger("uvicorn.error")

# ---------- env / config ----------
DATABASE_URL = os.getenv("DATABASE_URL", "")
FALLBACK_PDF_BASE = os.getenv("FALLBACK_PDF_BASE", "").rstrip("/")  # e.g. https://<app>.ondigitalocean.app/label
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "12"))
CACHE_DIR = "/tmp/label-cache"
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# ---------- DB pool ----------
pool: Optional[psycopg2.pool.SimpleConnectionPool] = None

def _dsn_with_ssl(dsn: str) -> str:
    # If no explicit sslmode is provided, add 'sslmode=require'
    if "sslmode=" not in dsn:
        sep = "" if dsn.endswith(" ") else " "
        dsn = f"{dsn}{sep}sslmode=require"
    return dsn

@app.on_event("startup")
def startup():
    global pool
    if not DATABASE_URL:
        log.warning("DATABASE_URL not set. API will return errors for DB-backed routes.")
        return
    dsn = _dsn_with_ssl(DATABASE_URL)
    pool = psycopg2.pool.SimpleConnectionPool(
        1, 8, dsn=dsn, cursor_factory=psycopg2.extras.DictCursor
    )
    log.info("DB connection pool initialized.")

@app.on_event("shutdown")
def shutdown():
    global pool
    if pool:
        pool.closeall()
        log.info("DB pool closed.")

# ---------- helpers ----------
SQL_LOOKUP = """
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

def db_fetch(barcode: str) -> Optional[Dict[str, Any]]:
    if not pool:
        return None
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(SQL_LOOKUP, (barcode,))
            row = cur.fetchone()
            if not row:
                return None
            # row keys: barcode, label_url, quantity, vars
            vars_val = row["vars"]
            if isinstance(vars_val, str):
                try:
                    vars_val = json.loads(vars_val)
                except Exception:
                    vars_val = []
            return {
                "barcode": row["barcode"],
                "label_url": row["label_url"] or "",
                "quantity": int(row["quantity"] or 1),
                "vars": vars_val or [],
            }
    finally:
        pool.putconn(conn)

def extract_var(vars_list: List[Dict[str, Any]], name: str) -> Optional[str]:
    name_l = name.lower()
    for v in vars_list:
        n = str(v.get("name", "")).lower()
        if n == name_l:
            return str(v.get("value", "") or "")
    return None

def cache_path_pdf(barcode: str) -> str:
    return os.path.join(CACHE_DIR, f"{barcode}.pdf")

def is_probably_pdf_url(url: str) -> bool:
    return bool(re.search(r"\.pdf(\?|$)", url, flags=re.I))

def fetch_bytes(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Return (content, content_type) or (None, None)."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        if r.ok:
            data = r.content
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            return data, ctype
        log.warning(f"fetch_bytes: {url} status={r.status_code}")
        return None, None
    except Exception as e:
        log.warning(f"fetch_bytes failed: {url} err={e}")
        return None, None

def png_to_pdf(png_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PDF")
    return out.getvalue()

def placeholder_png(barcode: str) -> bytes:
    W, H = 1200, 1800  # 4x6in @300dpi
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.multiline_text((60, 60), f"BARCODE:\n{barcode}\n(NO LABEL URL)", fill="black", spacing=10)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ---------- health ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}

# ---------- label serving (machine will fetch the PDF one) ----------
@app.get("/label/{barcode}.pdf")
def serve_label_pdf(barcode: str):
    rec = db_fetch(barcode)
    if not rec:
        return JSONResponse({"code": 0, "msg": "Not found"}, status_code=200)

    cpath = cache_path_pdf(barcode)
    try:
        # serve from cache if fresh (<10 min)
        if os.path.exists(cpath) and time.time() - os.path.getmtime(cpath) < 600:
            with open(cpath, "rb") as f:
                return Response(f.read(), media_type="application/pdf",
                                headers={"Cache-Control": "public, max-age=300",
                                         "Content-Disposition": f'inline; filename="{barcode}.pdf"'})

        label_url = rec["label_url"]
        content, ctype = (None, None)

        if label_url:
            content, ctype = fetch_bytes(label_url)

        if content and (ctype == "application/pdf" or is_probably_pdf_url(label_url)):
            pdf_bytes = content
        else:
            # need to convert from PNG → PDF (or create placeholder)
            if not content:
                png_bytes = placeholder_png(barcode)
            elif ctype in ("image/png", "image/jpeg", "image/jpg"):
                png_bytes = content
            else:
                # unknown type: try convert anyway
                png_bytes = content
            pdf_bytes = png_to_pdf(png_bytes)

        with open(cpath, "wb") as f:
            f.write(pdf_bytes)

        return Response(pdf_bytes, media_type="application/pdf",
                        headers={"Cache-Control": "public, max-age=300",
                                 "Content-Disposition": f'inline; filename="{barcode}.pdf"'})
    except Exception:
        log.exception("serve_label_pdf failed")
        return JSONResponse({"code": 0, "msg": "Server error"}, status_code=200)

# Optional: for your own testing
@app.get("/label/{barcode}.png")
def serve_label_png(barcode: str):
    rec = db_fetch(barcode)
    if not rec:
        return JSONResponse({"code": 0, "msg": "Not found"}, status_code=200)

    label_url = rec["label_url"]
    if label_url:
        content, ctype = fetch_bytes(label_url)
        if content and ctype in ("image/png", "image/jpeg", "image/jpg"):
            return Response(content, media_type=ctype, headers={"Cache-Control": "public, max-age=120"})
    # fall back to placeholder
    return Response(placeholder_png(barcode), media_type="image/png", headers={"Cache-Control": "public, max-age=60"})

# ---------- factory-facing endpoint ----------
@app.get("/api/scanbarcode/getpdfurl")
def get_pdfurl(
    barcode: str = Query(..., min_length=1),
    mac: Optional[str] = None,
    prefer: Optional[str] = Query(None, description="set to 'png' to also include PNGUrl (for testing only)")
):
    """
    Contract the factory expects:
    - Always return code=1/msg='OK' with PDFUrl when found
    - Return code=0/msg='Not found' when barcode is unknown
    - Include Quantity; Order/Color/Size if available from vars
    """
    rec = db_fetch(barcode)
    if not rec:
        return {"code": 0, "msg": "Not found"}

    base = FALLBACK_PDF_BASE or "https://example.invalid/label"
    pdf_url = f"{base}/{barcode}.pdf"

    color = extract_var(rec["vars"], "Color")
    size  = extract_var(rec["vars"], "Size")
    order = extract_var(rec["vars"], "Order")  # if present in your data

    resp: Dict[str, Any] = {
        "code": 1,
        "msg": "OK",
        "Quantity": int(rec["quantity"] or 1),
        "Barcode": rec["barcode"],
        "PDFUrl": pdf_url,
    }
    if order: resp["Order"] = order
    if color: resp["Color"] = color
    if size:  resp["Size"]  = size
    if prefer == "png":
        resp["PNGUrl"] = f"{base}/{barcode}.png"

    return resp
