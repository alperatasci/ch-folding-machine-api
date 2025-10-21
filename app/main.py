import os
import io
import time
import logging
import asyncio
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Tuple, Dict, Any, List, Iterator
from dataclasses import dataclass

import requests
from fastapi import FastAPI, Query, Body, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from cachetools import TTLCache
from PIL import Image, ImageDraw
import structlog

# ----------------------
# Configuration with Validation
# ----------------------
class Settings(BaseSettings):
    database_url: str = Field("", env="DATABASE_URL")
    pgssl: str = Field("require", env="PGSSL")
    fallback_pdf_base: str = Field("", env="FALLBACK_PDF_BASE")
    include_png_url: bool = Field(False, env="INCLUDE_PNG_URL")
    png_to_pdf_mode: str = Field("auto", env="PNG_TO_PDF_MODE")  # auto|force|off
    http_timeout: float = Field(30.0, env="HTTP_TIMEOUT")
    cache_ttl_meta: int = Field(600, env="CACHE_TTL_META_SEC")  # 10 min
    cache_ttl_pdf: int = Field(600, env="CACHE_TTL_PDF_SEC")   # 10 min
    cache_max_size: int = Field(1000, env="CACHE_MAX_SIZE")
    pg_min_conn: int = Field(1, env="PG_MIN")
    pg_max_conn: int = Field(20, env="PG_MAX")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # PDF generation settings
    fallback_pdf_width: int = Field(600, env="FALLBACK_PDF_WIDTH")
    fallback_pdf_height: int = Field(900, env="FALLBACK_PDF_HEIGHT")
    
    # Streaming settings
    stream_chunk_size: int = Field(8192, env="STREAM_CHUNK_SIZE")

    class Config:
        env_file = ".env"

settings = Settings()

# Order statuses to skip processing
BAD_STATUSES = {4, 5}  # 4: Shipped, 5: Cancelled

# ----------------------
# Structured Logging Setup
# ----------------------
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("ch-folding-api")

# ----------------------
# Database Models & Queries
# ----------------------
@dataclass
class OrderData:
    barcode: str
    label_url: str
    quantity: int
    vars: List[Dict[str, Any]]
    status: Optional[int]

# Optimized SQL with better indexing hints and performance improvements
ORDER_QUERY = """
SELECT
  o."OrderNumber"               AS barcode,
  COALESCE(os."LabelUrl", os."TrackingUrl", '') AS label_url,
  COALESCE((
    SELECT SUM(oi."Quantity") 
    FROM "OrderItems" oi 
    WHERE oi."OrderId" = o."Id"
  ), 1) AS quantity,
  COALESCE((
    SELECT jsonb_agg(DISTINCT jsonb_build_object(
      'name',  oiv."FormattedName",
      'value', oiv."FormattedValue"
    )) FILTER (WHERE oiv."FormattedName" IS NOT NULL)
    FROM "OrderItems" oi
    LEFT JOIN "OrderItemVariations" oiv ON oiv."OrderItemId" = oi."Id"
    WHERE oi."OrderId" = o."Id"
  ), '[]'::jsonb) AS vars,
  o."OrderStatusId"             AS status
FROM "Orders" o
LEFT JOIN "OrderShipments" os ON os."OrderId" = o."Id" AND os."CreatedAt" = (
  SELECT MAX(os2."CreatedAt") 
  FROM "OrderShipments" os2 
  WHERE os2."OrderId" = o."Id"
)
WHERE o."OrderNumber" = %s
LIMIT 1;

"""

# ----------------------
# Request/Response Models
# ----------------------
class ScanBody(BaseModel):
    barcode: str = Field(..., min_length=1, max_length=100)
    mac: Optional[str] = Field(None, max_length=17)

class FactoryResponse(BaseModel):
    code: int
    msg: str
    Quantity: int
    Order: str
    Color: str
    Size: str
    Barcode: str
    PDFUrl: str
    PNGUrl: Optional[str] = None

# ----------------------
# Enhanced Connection Pool Management
# ----------------------
class DatabasePool:
    def __init__(self):
        self.pool: Optional[SimpleConnectionPool] = None
        self._initialized = False
    
    def initialize(self):
        if not settings.database_url:
            logger.warning("DATABASE_URL not set. API will run in FALLBACK mode only.")
            return
            
        try:
            dsn = settings.database_url
            if "sslmode=" not in dsn and settings.pgssl:
                sep = "&" if "?" in dsn else "?"
                dsn = f"{dsn}{sep}sslmode={settings.pgssl}"
            
            self.pool = SimpleConnectionPool(
                minconn=settings.pg_min_conn,
                maxconn=settings.pg_max_conn,
                dsn=dsn
            )
            self._initialized = True
            logger.info("Database pool initialized", 
                       min_conn=settings.pg_min_conn, 
                       max_conn=settings.pg_max_conn)
        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise
    
    def close(self):
        if self.pool:
            self.pool.closeall()
            logger.info("Database pool closed")
            self._initialized = False
    
    @contextmanager
    def get_connection(self):
        if not self.pool or not self._initialized:
            yield None
            return
        
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            logger.error("Database connection error", error=str(e))
            raise
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except Exception as e:
                    logger.error("Failed to return connection to pool", error=str(e))
    
    def health_check(self) -> bool:
        """Check if database is accessible"""
        if not self.pool or not self._initialized:
            return False
        
        try:
            with self.get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        return cur.fetchone() is not None
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
        return False

db_pool = DatabasePool()

# ----------------------
# Enhanced Caching with Auto-Cleanup
# ----------------------
class CacheManager:
    def __init__(self):
        self.meta_cache: TTLCache = TTLCache(
            maxsize=settings.cache_max_size, 
            ttl=settings.cache_ttl_meta
        )
        self.pdf_cache: TTLCache = TTLCache(
            maxsize=settings.cache_max_size // 2,  # PDFs are larger
            ttl=settings.cache_ttl_pdf
        )
    
    def get_meta(self, barcode: str) -> Optional[Dict[str, Any]]:
        return self.meta_cache.get(barcode)
    
    def put_meta(self, barcode: str, data: Dict[str, Any]) -> None:
        self.meta_cache[barcode] = data
    
    def get_pdf(self, key: Tuple[str, str]) -> Optional[bytes]:
        return self.pdf_cache.get(key)
    
    def put_pdf(self, key: Tuple[str, str], pdf_bytes: bytes) -> None:
        # Only cache if not too large (prevent memory issues)
        if len(pdf_bytes) < 5 * 1024 * 1024:  # 5MB limit
            self.pdf_cache[key] = pdf_bytes
    
    def clear(self):
        self.meta_cache.clear()
        self.pdf_cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        return {
            "meta_cache": {
                "size": len(self.meta_cache),
                "max_size": self.meta_cache.maxsize,
                "hits": getattr(self.meta_cache, 'hits', 0),
                "misses": getattr(self.meta_cache, 'misses', 0)
            },
            "pdf_cache": {
                "size": len(self.pdf_cache),
                "max_size": self.pdf_cache.maxsize,
                "hits": getattr(self.pdf_cache, 'hits', 0),
                "misses": getattr(self.pdf_cache, 'misses', 0)
            }
        }

cache_manager = CacheManager()

# ----------------------
# Enhanced App Lifecycle
# ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting CH Folding Machine API", version="2.0.0")
    db_pool.initialize()
    yield
    # Shutdown
    logger.info("Shutting down CH Folding Machine API")
    db_pool.close()
    cache_manager.clear()

app = FastAPI(
    title="CH Folding Machine API", 
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Request Middleware for Logging
# ----------------------
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = f"{int(time.time() * 1000000) % 1000000:06d}"
    
    logger.info("Request started",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                client=request.client.host if request.client else None)
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info("Request completed",
                    request_id=request_id,
                    status_code=response.status_code,
                    process_time=f"{process_time:.3f}s")
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error("Request failed",
                     request_id=request_id,
                     error=str(e),
                     process_time=f"{process_time:.3f}s")
        raise

# ----------------------
# Enhanced Database Operations
# ----------------------
def db_get_order(barcode: str) -> Optional[OrderData]:
    """Fetch order info from DB with enhanced error handling and timeout monitoring"""
    start_time = time.time()
    
    with db_pool.get_connection() as conn:
        if not conn:
            logger.warning("Database connection not available", barcode=barcode)
            return None
        
        try:
            with conn.cursor() as cur:
                logger.debug("Database query starting", barcode=barcode)
                cur.execute(ORDER_QUERY, (barcode,))
                row = cur.fetchone()
                duration = time.time() - start_time
                
                if not row:
                    logger.info("Order not found in database", barcode=barcode, query_duration=f"{duration:.2f}s")
                    return None
                
                logger.debug("Database query successful", barcode=barcode, query_duration=f"{duration:.2f}s")
                return OrderData(
                    barcode=row[0],
                    label_url=row[1] or "",
                    quantity=int(row[2] or 1),
                    vars=row[3] or [],
                    status=int(row[4]) if row[4] is not None else None
                )
        except psycopg2.Error as e:
            duration = time.time() - start_time
            logger.error("Database query failed", barcode=barcode, query_duration=f"{duration:.2f}s", error=str(e))
            raise HTTPException(status_code=503, detail="Database temporarily unavailable")
        except Exception as e:
            duration = time.time() - start_time
            logger.error("Unexpected database error", barcode=barcode, query_duration=f"{duration:.2f}s", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")

# ----------------------
# Enhanced Helper Functions
# ----------------------
def extract_var(vars_list: List[Dict[str, Any]], names: List[str]) -> str:
    """Extract variable value with case-insensitive matching"""
    if not vars_list:
        return ""
    
    lname = [n.lower() for n in names]
    for item in vars_list:
        n = str(item.get("name", "")).lower()
        if n in lname:
            return str(item.get("value", "") or "")
    return ""

def build_urls(barcode: str, label_url: str) -> Tuple[str, Optional[str]]:
    """Build PDF and PNG URLs"""
    base = settings.fallback_pdf_base.rstrip("/") if settings.fallback_pdf_base else "/label"
    pdf_url = f"{base}/{barcode}.pdf"
    png_url = None
    
    if settings.include_png_url and label_url:
        lower_url = label_url.lower()
        if any(lower_url.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            png_url = label_url
    
    return pdf_url, png_url

def http_get_with_retry(url: str, stream: bool = False, retries: int = 2):
    """HTTP GET with retry logic"""
    last_exception = None
    start_time = time.time()
    
    for attempt in range(retries + 1):
        try:
            logger.info("HTTP request starting", url=url, attempt=attempt + 1, timeout=settings.http_timeout)
            response = requests.get(
                url, 
                timeout=settings.http_timeout, 
                stream=stream,
                headers={'User-Agent': 'CH-Folding-Machine-API/2.0.0'}
            )
            response.raise_for_status()
            duration = time.time() - start_time
            logger.info("HTTP request successful", url=url, duration=f"{duration:.2f}s")
            return response
        except requests.Timeout as e:
            duration = time.time() - start_time
            logger.error("HTTP request timeout", url=url, duration=f"{duration:.2f}s", timeout=settings.http_timeout)
            last_exception = e
            if attempt < retries:
                wait_time = 2 ** attempt
                logger.warning(f"HTTP timeout, retrying in {wait_time}s", 
                             url=url, attempt=attempt + 1, error=str(e))
                time.sleep(wait_time)
            else:
                logger.error("HTTP request failed after all retries due to timeout", 
                           url=url, retries=retries, total_duration=f"{duration:.2f}s")
        except requests.RequestException as e:
            duration = time.time() - start_time
            logger.error("HTTP request failed", url=url, duration=f"{duration:.2f}s", error=str(e))
            last_exception = e
            if attempt < retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"HTTP request failed, retrying in {wait_time}s", 
                             url=url, attempt=attempt + 1, error=str(e))
                time.sleep(wait_time)
            else:
                logger.error("HTTP request failed after all retries", 
                           url=url, retries=retries, error=str(e))
    
    raise last_exception

def image_to_pdf_bytes(img_bytes: bytes) -> bytes:
    """Convert image to PDF with error handling and optimization"""
    if not img_bytes:
        raise ValueError("Empty image data")
    
    try:
        # Try img2pdf first (faster, smaller files)
        import img2pdf
        return img2pdf.convert(img_bytes)
    except ImportError:
        logger.info("img2pdf not available, using PIL fallback")
    except Exception as e:
        logger.warning("img2pdf conversion failed, using PIL fallback", error=str(e))
    
    # PIL fallback
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            # Handle transparency
            if im.mode in ("RGBA", "LA", "P"):
                # Create white background
                if im.mode == "P":
                    im = im.convert("RGBA")
                background = Image.new("RGB", im.size, (255, 255, 255))
                if im.mode == "RGBA":
                    background.paste(im, mask=im.split()[-1])
                else:
                    background.paste(im)
                im = background
            elif im.mode != "RGB":
                im = im.convert("RGB")
            
            # Optimize size if too large
            max_size = (2000, 2000)
            if im.size[0] > max_size[0] or im.size[1] > max_size[1]:
                orig_size = im.size
                im.thumbnail(max_size, Image.Resampling.LANCZOS)
                logger.info("Image resized for PDF conversion", 
                          original_size=orig_size, new_size=im.size)
            
            out = io.BytesIO()
            im.save(out, format="PDF", optimize=True)
            return out.getvalue()
    except Exception as e:
        logger.error("PIL PDF conversion failed", error=str(e))
        raise HTTPException(status_code=500, detail="Image to PDF conversion failed")

def make_fallback_pdf(barcode: str) -> bytes:
    """Generate fallback PDF with better formatting"""
    try:
        img = Image.new("RGB", (settings.fallback_pdf_width, settings.fallback_pdf_height), "white")
        draw = ImageDraw.Draw(img)
        
        # Try to load a better font
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except:
            font = None
        
        msg = f"NO LABEL FOUND\n\nBARCODE: {barcode}\n\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Center the text
        bbox = draw.textbbox((0, 0), msg, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (settings.fallback_pdf_width - text_width) // 2
        y = (settings.fallback_pdf_height - text_height) // 2
        
        draw.multiline_text((x, y), msg, fill="black", font=font, align="center", spacing=10)
        
        out = io.BytesIO()
        img.save(out, format="PDF")
        return out.getvalue()
    except Exception as e:
        logger.error("Fallback PDF generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="PDF generation failed")

def stream_bytes(data: bytes, chunk_size: int | None = None) -> Iterator[bytes]:
    """Stream bytes in chunks to reduce memory usage"""
    chunk_size = chunk_size or settings.stream_chunk_size
    bio = io.BytesIO(data)
    while True:
        chunk = bio.read(chunk_size)
        if not chunk:
            break
        yield chunk

# ----------------------
# Enhanced Business Logic
# ----------------------
def resolve_label(barcode: str) -> Dict[str, Any]:
    """Resolve label information with enhanced caching and error handling"""
    # Check cache first
    cached_data = cache_manager.get_meta(barcode)
    if cached_data:
        logger.debug("Cache hit for barcode", barcode=barcode)
        return cached_data
    
    logger.debug("Cache miss for barcode", barcode=barcode)
    
    # Get from database
    order_data = db_get_order(barcode)
    
    # Check if order is blocked
    if order_data and order_data.status in BAD_STATUSES:
        data = {
            "barcode": barcode,
            "label_url": "",
            "quantity": 0,
            "vars": [],
            "status": order_data.status,
            "order": "",
            "color": "",
            "size": "",
            "pdf_url": "",
            "png_url": None,
            "blocked": True,
        }
        cache_manager.put_meta(barcode, data)
        logger.info("Order blocked due to status", barcode=barcode, status=order_data.status)
        return data
    
    # Extract order information
    label_url = order_data.label_url if order_data else ""
    quantity = order_data.quantity if order_data else 1
    vars_list = order_data.vars if order_data else []
    
    size = extract_var(vars_list, ["Size", "SIZE", "size"])
    color = extract_var(vars_list, ["Color", "COLOR", "color", "Colorway"])
    
    pdf_url, png_url = build_urls(barcode, label_url)
    
    data = {
        "barcode": barcode,
        "label_url": label_url,
        "quantity": int(quantity),
        "vars": vars_list,
        "status": order_data.status if order_data else None,
        "order": "",
        "color": color,
        "size": size,
        "pdf_url": pdf_url,
        "png_url": png_url,
        "blocked": False,
    }
    
    cache_manager.put_meta(barcode, data)
    logger.info("Label resolved", barcode=barcode, has_label_url=bool(label_url))
    return data

# ----------------------
# API Routes
# ----------------------
@app.get("/healthz", response_class=PlainTextResponse)
def health_check():
    """Enhanced health check with database connectivity"""
    try:
        db_healthy = db_pool.health_check()
        if settings.database_url and not db_healthy:
            return PlainTextResponse("unhealthy - database connection failed", status_code=503)
        return PlainTextResponse("healthy")
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return PlainTextResponse("unhealthy - internal error", status_code=503)

@app.get("/health/detailed")
def detailed_health_check():
    """Detailed health check with system information"""
    try:
        db_healthy = db_pool.health_check()
        cache_stats = cache_manager.stats()
        
        return {
            "status": "healthy" if (not settings.database_url or db_healthy) else "unhealthy",
            "timestamp": time.time(),
            "database": {
                "configured": bool(settings.database_url),
                "healthy": db_healthy
            },
            "cache": cache_stats,
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        return {"status": "error", "error": str(e)}

@app.get("/")
def root():
    return {"ok": True, "service": "ch-folding-machine-api", "version": "2.0.0"}

def _ship_order_in_db(barcode: str) -> bool:
    """Ship order by updating database directly to OrderStatusId=4"""
    with db_pool.get_connection() as conn:
        if not conn:
            logger.warning("Database connection not available for shipping", barcode=barcode)
            return False

        try:
            with conn.cursor() as cur:
                # Update order to shipped status
                cur.execute("""
                    UPDATE "Orders"
                    SET
                        "OrderStatusId" = 4,
                        "ScanToShippedAt" = NOW(),
                        "ScanToShippedUser" = 'AutoPacking'
                    WHERE "OrderNumber" = %s
                    AND "OrderStatusId" < 4
                    RETURNING "Id"
                """, (barcode,))

                result = cur.fetchone()

                if not result:
                    # Already shipped or not found - not an error
                    return True

                order_id = result[0]

                # Insert order history
                cur.execute("""
                    INSERT INTO "OrderHistories"
                    ("Id", "OrderId", "Description", "CreatedBy", "CreatedAt")
                    VALUES (gen_random_uuid(), %s, 'Order is shipped (AutoPacking)', 'AutoPacking', NOW())
                """, (order_id,))

                conn.commit()
                logger.info("Order shipped successfully", barcode=barcode, order_id=str(order_id))
                return True

        except Exception as e:
            conn.rollback()
            logger.error("Error shipping order", barcode=barcode, error=str(e))
            return False

def _create_factory_payload(barcode: str) -> FactoryResponse:
    """Create factory response payload with enhanced error handling"""
    try:
        data = resolve_label(barcode)

        # If shipped/cancelled: tell the machine not to proceed
        if data.get("blocked"):
            return FactoryResponse(
                code=0,
                msg="Order is not eligible (shipped/cancelled)",
                Quantity=0,
                Order="",
                Color="",
                Size="",
                Barcode=barcode,
                PDFUrl="",
            )

        payload = FactoryResponse(
            code=1,
            msg="OK",
            Quantity=data["quantity"],
            Order=data["order"],
            Color=data["color"],
            Size=data["size"],
            Barcode=barcode,
            PDFUrl=data["pdf_url"],
            PNGUrl=data["png_url"] if settings.include_png_url else None
        )

        # If no DB row and no FALLBACK_PDF_BASE, mark as not found
        if not data.get("label_url") and not settings.fallback_pdf_base:
            payload.code = 0
            payload.msg = "Not found"

        # NEW: Ship the order automatically when returning PDF URL
        # (Only if order was found and not blocked)
        if payload.code == 1:
            _ship_order_in_db(barcode)

        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating factory payload", barcode=barcode, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# API Endpoints
@app.get("/api/scanbarcode/getpdfurl", response_model=FactoryResponse)
def get_pdfurl(barcode: str = Query(..., description="order number / barcode"),
               mac: Optional[str] = Query(None)):
    return _create_factory_payload(barcode)

@app.get("/api/printdata", response_model=FactoryResponse)
def get_pdfurl_alias(barcode: str = Query(...), mac: Optional[str] = Query(None)):
    return _create_factory_payload(barcode)

@app.post("/api/scanbarcode/getpdfurl", response_model=FactoryResponse)
def post_pdfurl(body: ScanBody):
    return _create_factory_payload(body.barcode)

@app.post("/api/printdata", response_model=FactoryResponse)
def post_pdfurl_alias(body: ScanBody):
    return _create_factory_payload(body.barcode)

@app.get("/label/{barcode}.pdf")
async def serve_pdf(barcode: str):
    """Serve PDF with enhanced streaming and caching"""
    try:
        data = resolve_label(barcode)
        if data.get("blocked"):
            raise HTTPException(status_code=409, detail="Order not eligible")
        
        src = data.get("label_url", "")
        cache_key = (barcode, src or "fallback")
        
        # Check PDF cache
        cached_pdf = cache_manager.get_pdf(cache_key)
        if cached_pdf:
            logger.debug("PDF cache hit", barcode=barcode)
            return StreamingResponse(
                stream_bytes(cached_pdf), 
                media_type="application/pdf",
                headers={
                    "Cache-Control": f"public, max-age={settings.cache_ttl_pdf}",
                    "Content-Length": str(len(cached_pdf))
                }
            )
        
        logger.debug("PDF cache miss", barcode=barcode)
        
        if src:
            src_lower = src.lower()
            
            # Direct PDF streaming (no conversion needed)
            if src_lower.endswith(".pdf") and settings.png_to_pdf_mode != "force":
                try:
                    response = http_get_with_retry(src, stream=True)
                    logger.info("Streaming PDF directly", barcode=barcode, source=src)
                    return StreamingResponse(
                        response.iter_content(chunk_size=settings.stream_chunk_size),
                        media_type="application/pdf",
                        headers={
                            "Cache-Control": f"public, max-age={settings.cache_ttl_pdf}",
                            "Content-Length": response.headers.get("Content-Length", "")
                        },
                        background=BackgroundTask(response.close)  # Close upstream connection
                    )
                except Exception as e:
                    logger.error("Direct PDF streaming failed", barcode=barcode, source=src, error=str(e))
                    raise HTTPException(status_code=502, detail="PDF source not accessible")
            
            # Download and convert
            try:
                response = http_get_with_retry(src, stream=False)
                
                if (src_lower.endswith((".png", ".jpg", ".jpeg")) or 
                    settings.png_to_pdf_mode in ("auto", "force")):
                    pdf_bytes = image_to_pdf_bytes(response.content)
                else:
                    # Unknown extension: validate and convert if needed
                    content = response.content
                    if content.startswith(b"%PDF-"):
                        pdf_bytes = content
                    else:
                        pdf_bytes = image_to_pdf_bytes(content)
                
                logger.info("PDF converted successfully", barcode=barcode, 
                          source=src, size=len(pdf_bytes))
                
            except requests.RequestException as e:
                logger.error("Failed to fetch PDF source", barcode=barcode, source=src, error=str(e))
                raise HTTPException(status_code=502, detail="PDF source not reachable")
        else:
            # Generate fallback PDF
            pdf_bytes = make_fallback_pdf(barcode)
            logger.info("Fallback PDF generated", barcode=barcode, size=len(pdf_bytes))
        
        # Cache the PDF
        cache_manager.put_pdf(cache_key, pdf_bytes)
        
        return StreamingResponse(
            stream_bytes(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Cache-Control": f"public, max-age={settings.cache_ttl_pdf}",
                "Content-Length": str(len(pdf_bytes))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error serving PDF", barcode=barcode, error=str(e))
        raise HTTPException(status_code=500, detail="PDF generation error")

# ----------------------
# Admin/Debug Endpoints
# ----------------------
@app.get("/admin/cache/stats")
def cache_stats():
    """Get cache statistics"""
    return cache_manager.stats()

@app.post("/admin/cache/clear")
def clear_cache():
    """Clear all caches"""
    cache_manager.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)