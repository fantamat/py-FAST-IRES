import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from main import app, process_image, process_pdf, process_docx


class TestInvoiceService(unittest.TestCase):
    """Test cases for the invoice service."""

    def setUp(self):
        """Set up test environment."""
        self.client = TestClient(app)
        # Mock the Gemini client for testing
        self.gemini_patcher = patch('invoce_service.main.client')
        self.mock_gemini = self.gemini_patcher.start()
        
        # Mock the response from Gemini
        self.mock_response = MagicMock()
        self.mock_response.parsed.model_dump.return_value = {
            "customer": "DEYMED",
            "billing_account": {
                "account_name": "Test Company",
                "company_id": "12345678",
                "vat_id": "CZ12345678",
                "adress": {
                    "street": "Test Street 123",
                    "city": "Test City",
                    "postalcode": "12345",
                    "country": "Test Country"
                },
                "account_phone": "123456789",
                "account_email": "test@example.com"
            },
            "order_total_price": 1000.0,
            "order_currency": "EUR",
            "items": [
                {
                    "part_number": "P001",
                    "description": "Test Item",
                    "quantity": 2,
                    "unit_price": 500.0,
                    "total_price": 1000.0
                }
            ]
        }
        self.mock_response.usage_metadata.total_token_count = 100
        
        self.mock_gemini.models.generate_content.return_value = self.mock_response

    def tearDown(self):
        """Clean up after tests."""
        self.gemini_patcher.stop()

    @patch('main.MarkItDown')
    def test_process_image(self, mock_markitdown):
        """Test processing an image file."""
        # REPLACE WITH ACTUAL IMAGE PATH
        image_path = "test/data/faktura.png"  
        
        # Skip test if file doesn't exist
        if not Path(image_path).exists():
            self.skipTest(f"Test image file not found: {image_path}")
        
        # Test image processing
        result = process_image(image_path)
        
        # Verify the result
        self.assertIn("invoice", result)
        self.assertIn("total_token_count", result)
        self.assertEqual(result["total_token_count"], 100)
        self.mock_gemini.models.generate_content.assert_called_once()

    @patch('main.MarkItDown')
    @patch('main.convert_from_path')
    def test_process_pdf(self, mock_convert_from_path, mock_markitdown):
        """Test processing a PDF file."""
        # REPLACE WITH ACTUAL PDF PATH
        pdf_path = "test/data/matejfanta-2505001.pdf"  # Replace with your test PDF file
        
        # Skip test if file doesn't exist
        if not Path(pdf_path).exists():
            self.skipTest(f"Test PDF file not found: {pdf_path}")
        
        # Mock the PDF to image conversion
        mock_image = MagicMock()
        mock_image.save.return_value = None
        mock_convert_from_path.return_value = [mock_image]
        
        # Mock markitdown conversion result
        mock_result = MagicMock()
        mock_result.text_content = "Test markdown content"
        mock_markitdown_instance = MagicMock()
        mock_markitdown_instance.convert_stream.return_value = mock_result
        mock_markitdown.return_value = mock_markitdown_instance
        
        # Test PDF processing
        result = process_pdf(pdf_path)
        
        # Verify the result
        self.assertIn("invoice", result)
        self.assertIn("total_token_count", result)
        self.assertEqual(result["total_token_count"], 100)
        self.mock_gemini.models.generate_content.assert_called_once()
        mock_markitdown.assert_called_once()
        mock_markitdown_instance.convert_stream.assert_called_once()

    @patch('main.MarkItDown')
    def test_process_docx(self, mock_markitdown):
        """Test processing a DOCX file."""
        # REPLACE WITH ACTUAL DOCX PATH
        docx_path = "test/data/Downloadable-Word-Invoice-Template.docx"  # Replace with your test DOCX file
        
        # Skip test if file doesn't exist
        if not Path(docx_path).exists():
            self.skipTest(f"Test DOCX file not found: {docx_path}")
        
        # Mock markitdown conversion result
        mock_result = MagicMock()
        mock_result.text_content = "Test markdown content"
        mock_markitdown_instance = MagicMock()
        mock_markitdown_instance.convert_stream.return_value = mock_result
        mock_markitdown.return_value = mock_markitdown_instance
        
        # Test DOCX processing
        result = process_docx(docx_path)
        
        # Verify the result
        self.assertIn("invoice", result)
        self.assertIn("total_token_count", result)
        self.assertEqual(result["total_token_count"], 100)
        self.mock_gemini.models.generate_content.assert_called_once()
        mock_markitdown.assert_called_once()
        mock_markitdown_instance.convert_stream.assert_called_once()

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/healthcheck")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_invoice_endpoint_image(self):
        """Test the invoice endpoint with an image file."""
        # REPLACE WITH ACTUAL IMAGE PATH
        image_path = "test/data/faktura.png"  # Replace with your test image file
        
        # Skip test if file doesn't exist
        if not Path(image_path).exists():
            self.skipTest(f"Test image file not found: {image_path}")
        
        # Create test image data
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Send request to the endpoint
        response = self.client.post(
            "/invoice",
            files={"file": ("test_image.jpg", image_data, "image/jpeg")}
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertIn("invoice", response.json())
        self.assertIn("total_token_count", response.json())

    def test_invoice_endpoint_pdf(self):
        """Test the invoice endpoint with a PDF file."""
        # REPLACE WITH ACTUAL PDF PATH
        pdf_path = "test/data/matejfanta-2505001.pdf"  # Replace with your test PDF file
        
        # Skip test if file doesn't exist
        if not Path(pdf_path).exists():
            self.skipTest(f"Test PDF file not found: {pdf_path}")
        
        # Create test PDF data
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        # Send request to the endpoint
        response = self.client.post(
            "/invoice",
            files={"file": ("test_invoice.pdf", pdf_data, "application/pdf")}
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertIn("invoice", response.json())
        self.assertIn("total_token_count", response.json())

    def test_invoice_endpoint_docx(self):
        """Test the invoice endpoint with a DOCX file."""
        # REPLACE WITH ACTUAL DOCX PATH
        docx_path = "test/data/Downloadable-Word-Invoice-Template.docx"  # Replace with your test DOCX file
        
        # Skip test if file doesn't exist
        if not Path(docx_path).exists():
            self.skipTest(f"Test DOCX file not found: {docx_path}")
        
        # Create test DOCX data
        with open(docx_path, "rb") as f:
            docx_data = f.read()
        
        # Send request to the endpoint
        response = self.client.post(
            "/invoice",
            files={"file": ("test_invoice.docx", docx_data, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertIn("invoice", response.json())
        self.assertIn("total_token_count", response.json())

    def test_invalid_file_format(self):
        """Test the invoice endpoint with an unsupported file format."""
        # Create a simple text file
        text_data = b"This is not a valid invoice file"
        
        # Send request to the endpoint
        response = self.client.post(
            "/invoice",
            files={"file": ("test.txt", text_data, "text/plain")}
        )
        
        # Check that we get an error response
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file format", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
