import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
import psycopg2
import psycopg2.extras

# ---------- Config ----------
DATABASE_URL = os.environ["DATABASE_URL"]                 # e.g. postgres://user:pass@host:5432/podunited
PGSSL = os.getenv("PGSSL", "disable")                     # "require" or "disable"
FALLBACK_PDF_BASE = os.getenv("FALLBACK_PDF_BASE", "")    # optional, e.g. https://labels.customhub.io/labels

app = FastAPI(title="CustomHub Label API", version="1.0.0")


# ---------- DB ----------
def get_conn():
    # If PGSSL=require, Python's psycopg2 uses system CAs; DO provides them.
    return psycopg2.connect(DATABASE_URL, sslmode=PGSSL)

SQL_LOOKUP = """
SELECT
  o."OrderNumber"                           AS "Barcode",
  COALESCE(SUM(oi."Quantity"), 1)           AS "Quantity",
  MAX(s."LabelUrl") FILTER (WHERE s."LabelUrl" <> '') AS "LabelUrl",
  MAX(s."LabelDate")                         AS "LabelDate"
FROM "Orders" o
LEFT JOIN "OrderItems" oi    ON oi."OrderId" = o."Id"
LEFT JOIN "OrderShipments" s ON s."OrderId" = o."Id"
WHERE o."OrderNumber" = %s
GROUP BY o."OrderNumber"
ORDER BY MAX(s."LabelDate") DESC NULLS LAST
LIMIT 1;
"""

def lookup_label_by_barcode(barcode: str):
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(SQL_LOOKUP, (barcode,))
        return cur.fetchone()


# ---------- Helpers ----------
def choose_pdf_url(raw_url: str | None, barcode: str) -> str | None:
    """Return a usable PDF URL or None if we cannot supply one.
       If your LabelUrl is already a PDF, use it.
       If it ends with .png and you host a converted PDF at FALLBACK_PDF_BASE, point to that.
    """
    if not raw_url:
        return None
    lower = raw_url.lower()
    if lower.endswith(".pdf"):
        return raw_url
    if lower.endswith(".png") and FALLBACK_PDF_BASE:
        return f"{FALLBACK_PDF_BASE.rstrip('/')}/{barcode}.pdf"
    return None  # PNG without fallback; machine requires PDF


def build_response(barcode: str):
    row = lookup_label_by_barcode(barcode)
    if not row:
        return {"code": 0, "msg": "Not found"}

    pdf_url = choose_pdf_url(row.get("LabelUrl"), barcode)
    if not pdf_url:
        return {"code": 0, "msg": "Not found (need PDF)"}

    qty = int(row.get("Quantity") or 1)
    return {
        "code": 1,
        "msg": "OK",
        "Quantity": qty,         # if >1 their machine will wait for N pieces before closing the bag
        "Barcode": barcode,
        "PDFUrl": pdf_url
        # You can also add Order/Color/Size if you later map them
    }


# ---------- API ----------
class ScanRequest(BaseModel):
    barcode: str
    mac: str | None = None  # optional; we ignore it

@app.get("/api/scanbarcode/getpdfurl")
def get_pdfurl(barcode: str = Query(...), mac: str | None = None):
    return build_response(barcode.strip())

@app.post("/api/scanbarcode/getpdfurl")
def post_pdfurl(body: ScanRequest):
    return build_response(body.barcode.strip())

@app.get("/healthz")
def health():
    return {"ok": True}
