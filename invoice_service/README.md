# Invoice Processing Service

A FastAPI service that processes invoice documents (images, PDFs, or DOCX files) and extracts structured data using Google's Gemini 2.5 AI model.

## Features

- Process invoice images (JPG, PNG)
- Process PDF documents using Microsoft's markitdown library for text extraction and rendering
- Process DOCX documents using Microsoft's markitdown library for text extraction
- Extract structured invoice data in JSON format
- Log token usage to files
- SQLite database storage for all processing inputs and outputs
- REST API endpoints for querying processing history

## Requirements

- Python 3.8+
- Google Gemini API key

## Installation

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:

```bash
# On Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# On Linux/macOS
export GEMINI_API_KEY="your-api-key-here"
```

## Usage

Start the service:

```bash
uvicorn main:app --reload
```

The service will be available at http://localhost:8000

### API Endpoints

#### POST /invoice

Upload an invoice file to be processed:

```bash
curl -X POST -F "file=@/path/to/invoice.pdf" -F "file_id=your-file-id" http://localhost:8000/invoice
```

#### GET /healthcheck

Check if the service is running correctly:

```bash
curl http://localhost:8000/healthcheck
```

#### GET /history

Retrieve processing history:

```bash
# Get the latest 50 records
curl http://localhost:8000/history

# Pagination with limit and offset
curl http://localhost:8000/history?limit=10&offset=20

# Filter by file_id
curl http://localhost:8000/history?file_id=your-file-id
```

#### DELETE /history/{record_id}

Delete a specific record from the history:

```bash
curl -X DELETE http://localhost:8000/history/123
```

## API Documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker

You can also run this service using Docker:

```bash
# Build the Docker image
docker build -t invoice-service .

# Run the Docker container
docker run -p 8000:8000 -e GEMINI_API_KEY="your-api-key-here" invoice-service
```
