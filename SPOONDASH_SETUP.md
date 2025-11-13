# Spoondash Database Integration

## Overview

The API now supports **two separate databases** for two different companies:
1. **United Pod** - Original database (existing endpoints)
2. **Spoondash** - New database (new endpoints)

Each company has its own dedicated API endpoints and database connection pool.

---

## Configuration

### Environment Variables

Add these to your `.env` file or Digital Ocean App Platform environment:

```bash
# United Pod Database (existing - no changes needed)
DATABASE_URL=postgresql://USER:PASS@HOST:5432/podunited
PGSSL=disable

# Spoondash Database (new)
DATABASE_URL_SPOONDASH=postgresql://USER:PASS@HOST:5432/Spoondash
PGSSL_SPOONDASH=require
```

### Digital Ocean App Platform Setup

1. Go to your App's **Settings** → **Environment Variables**
2. Add the following new environment variables:
   - `DATABASE_URL_SPOONDASH` - Your Spoondash PostgreSQL connection string
   - `PGSSL_SPOONDASH` - Set to `require` or `disable` based on your DB config

---

## API Endpoints

### United Pod Endpoints (Unchanged)

These continue to work exactly as before:

**PDF Endpoints:**
```
GET  /api/scanbarcode/getpdfurl?barcode=XXX&mac=YYY
POST /api/scanbarcode/getpdfurl
GET  /api/printdata?barcode=XXX&mac=YYY
POST /api/printdata
```

**PNG Endpoints (NEW - no PDF conversion):**
```
GET  /api/scanbarcode/getpngurl?barcode=XXX&mac=YYY
POST /api/scanbarcode/getpngurl
```

**Example:**
```bash
# Get PDF URL
curl "https://your-api.com/api/scanbarcode/getpdfurl?barcode=ORDER123"

# Get PNG URL (no conversion)
curl "https://your-api.com/api/scanbarcode/getpngurl?barcode=ORDER123"
```

---

### Spoondash Endpoints (New)

Use these endpoints for Spoondash orders:

**PDF Endpoints:**
```
GET  /api/spoondash/scanbarcode/getpdfurl?barcode=XXX&mac=YYY
POST /api/spoondash/scanbarcode/getpdfurl
GET  /api/spoondash/printdata?barcode=XXX&mac=YYY
POST /api/spoondash/printdata
```

**PNG Endpoints (NEW - no PDF conversion):**
```
GET  /api/spoondash/scanbarcode/getpngurl?barcode=XXX&mac=YYY
POST /api/spoondash/scanbarcode/getpngurl
```

**Example:**
```bash
# Get PDF URL
curl "https://your-api.com/api/spoondash/scanbarcode/getpdfurl?barcode=ORDER456"

# Get PNG URL (no conversion)
curl "https://your-api.com/api/spoondash/scanbarcode/getpngurl?barcode=ORDER456"
```

---

### Direct Image Serving Endpoints

**Serve PNG/JPG directly (no conversion):**
```
GET  /label/{barcode}.png?db=unitedpod
GET  /label/{barcode}.png?db=spoondash
```

**Examples:**
```bash
# United Pod PNG
curl "https://your-api.com/label/ORDER123.png?db=unitedpod"

# Spoondash PNG
curl "https://your-api.com/label/ORDER456.png?db=spoondash"
```

**Features:**
- Serves the original image from `LabelUrl` without any conversion
- Supports PNG, JPG, and JPEG formats
- Automatically detects image type
- Cached for performance
- Use `db` query parameter to specify database (defaults to `unitedpod`)

---

## Request/Response Format

Both United Pod and Spoondash endpoints use the **same request/response format**.

### Request (GET)
```
GET /api/spoondash/scanbarcode/getpdfurl?barcode=ORDER123&mac=00:11:22:33:44:55
```

### Request (POST)
```json
{
  "barcode": "ORDER123",
  "mac": "00:11:22:33:44:55"  // optional
}
```

### Response (Success)
```json
{
  "code": 1,
  "msg": "OK",
  "Quantity": 5,
  "Order": "",
  "Color": "Red",
  "Size": "Large",
  "Barcode": "ORDER123",
  "PDFUrl": "/label/ORDER123.pdf",
  "PNGUrl": null
}
```

### Response (Blocked/Cancelled)
```json
{
  "code": 0,
  "msg": "Order is not eligible (shipped/cancelled)",
  "Quantity": 0,
  "Order": "",
  "Color": "",
  "Size": "",
  "Barcode": "ORDER123",
  "PDFUrl": ""
}
```

---

## Database Schema

Both databases must have the **same schema** (CustomHub format):

**Required Tables:**
- `Orders` - Main order table
- `OrderShipments` - Shipment/label URLs
- `OrderItems` - Order line items
- `OrderItemVariations` - Product variations (Size, Color, etc.)
- `OrderHistories` - Order status history

**Key Columns:**
- `Orders.OrderNumber` - Barcode lookup key
- `Orders.OrderStatusId` - Status (4=Shipped, 5=Cancelled)
- `OrderShipments.LabelUrl` - PDF/label URL

---

## Automatic Order Shipping

**IMPORTANT:** When a barcode is scanned and successfully returns a label:

1. The order status is automatically updated to `OrderStatusId = 4` (Shipped)
2. `ScanToShippedAt` is set to current timestamp
3. `ScanToShippedUser` is set to `'AutoPacking'`
4. A record is inserted into `OrderHistories` table

This happens for **both United Pod and Spoondash** databases.

---

## Health Check Endpoints

### Basic Health Check
```
GET /healthz
```

Returns `200 OK` with text "healthy" if both databases are accessible.

### Detailed Health Check
```
GET /health/detailed
```

Returns JSON with detailed status:
```json
{
  "status": "healthy",
  "timestamp": 1699564800.123,
  "databases": {
    "unitedpod": {
      "configured": true,
      "healthy": true
    },
    "spoondash": {
      "configured": true,
      "healthy": true
    }
  },
  "cache": {
    "meta_cache": {
      "size": 45,
      "max_size": 1000
    },
    "pdf_cache": {
      "size": 12,
      "max_size": 500
    }
  },
  "version": "2.0.0"
}
```

---

## Caching

The API caches data separately for each database:

- **Cache Key Format:** `{DatabaseName}_{Barcode}`
  - United Pod: `UnitedPod_ORDER123`
  - Spoondash: `Spoondash_ORDER456`

- **TTL:** 600 seconds (10 minutes) by default
- **Cache Types:**
  - Metadata cache (order info, color, size, etc.)
  - PDF cache (actual PDF bytes)

---

## Logging

All operations are logged with the database name:

```json
{
  "event": "Database query starting",
  "barcode": "ORDER123",
  "db_name": "Spoondash",
  "timestamp": "2024-11-09T10:30:00Z"
}
```

This helps distinguish which database is being accessed in the logs.

---

## Migration Guide

### For Existing United Pod Users
- **No changes required**
- All existing endpoints continue to work
- No configuration changes needed

### For New Spoondash Users
1. Add `DATABASE_URL_SPOONDASH` environment variable
2. Add `PGSSL_SPOONDASH` environment variable
3. Update your machines to use `/api/spoondash/...` endpoints
4. Test with a sample barcode

---

## Testing

### Test United Pod Endpoint
```bash
curl "http://localhost:8000/api/scanbarcode/getpdfurl?barcode=TEST123"
```

### Test Spoondash Endpoint
```bash
curl "http://localhost:8000/api/spoondash/scanbarcode/getpdfurl?barcode=TEST456"
```

### Test Health Check
```bash
curl "http://localhost:8000/health/detailed"
```

---

## Troubleshooting

### Database Connection Issues

**Problem:** `unhealthy - Spoondash database connection failed`

**Solution:**
1. Check `DATABASE_URL_SPOONDASH` is set correctly
2. Verify database credentials
3. Check `PGSSL_SPOONDASH` setting matches your DB config
4. Ensure database is accessible from Digital Ocean

### Order Not Found

**Problem:** `{"code": 0, "msg": "Not found"}`

**Solution:**
1. Verify barcode exists in the correct database
2. Check `OrderNumber` field matches exactly
3. Ensure order status is not 4 (Shipped) or 5 (Cancelled)

### Wrong Database

**Problem:** Spoondash order returns from United Pod endpoint

**Solution:**
- Use the correct endpoint:
  - United Pod: `/api/scanbarcode/getpdfurl`
  - Spoondash: `/api/spoondash/scanbarcode/getpdfurl`

---

## Support

For issues or questions:
1. Check Digital Ocean logs: `App Settings → Runtime Logs`
2. Verify environment variables are set correctly
3. Test health check endpoint: `/health/detailed`
4. Review structured logs for error details
