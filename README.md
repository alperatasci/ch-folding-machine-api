# CH Folding Machine Label API

FastAPI microservice for the folding machine to fetch a PDF label by barcode.

- GET/POST `/api/scanbarcode/getpdfurl?barcode=...`
- Returns JSON with { code, msg, Quantity, Barcode, PDFUrl }
- Requires Postgres (United Pod DB)
- PDF is required by the machine
