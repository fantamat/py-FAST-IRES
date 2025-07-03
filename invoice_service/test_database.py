import os
import unittest
import sqlite3
import json
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

# Import from the correct location
from invoice_service.main import app, setup_database, save_to_database, DB_PATH


class TestDatabaseFunctionality(unittest.TestCase):
    """Test cases for the database functionality."""

    def setUp(self):
        """Set up test environment."""
        self.client = TestClient(app)
        
        # Use a temporary database file for testing
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.db_path_patcher = patch('invoice_service.main.DB_PATH', Path(self.temp_db))
        self.mock_db_path = self.db_path_patcher.start()
        
        # Initialize the test database
        setup_database()
        
        # Mock the Gemini client for testing
        self.gemini_patcher = patch('invoice_service.main.client')
        self.mock_gemini = self.gemini_patcher.start()
        
        # Setup mock response
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
        self.db_path_patcher.stop()
        
        # Remove the temporary database
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)

    def test_save_to_database(self):
        """Test saving data to the database."""
        # Test data
        file_id = "test-file-id"
        file_name = "test-invoice.pdf"
        file_type = "pdf"
        token_count = 100
        request_data = {"filename": file_name, "file_id": file_id}
        response_data = {"invoice": {"customer": "DEYMED"}, "token_count": token_count}
        
        # Call the function
        save_to_database(
            file_id=file_id,
            file_name=file_name,
            file_type=file_type,
            token_count=token_count,
            request_data=request_data,
            response_data=response_data
        )
        
        # Query the database to verify
        conn = sqlite3.connect(self.temp_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM invoice_processes WHERE file_id = ?", [file_id])
        row = cursor.fetchone()
        conn.close()
        
        # Verify the data
        self.assertIsNotNone(row)
        self.assertEqual(row["file_id"], file_id)
        self.assertEqual(row["file_name"], file_name)
        self.assertEqual(row["file_type"], file_type)
        self.assertEqual(row["token_count"], token_count)
        self.assertEqual(json.loads(row["request_json"]), request_data)
        self.assertEqual(json.loads(row["response_json"]), response_data)

    def test_error_saving(self):
        """Test saving error data to the database."""
        # Test data
        file_id = "error-file-id"
        file_name = "error-file.txt"
        file_type = "txt"
        error_message = "Unsupported file format"
        
        # Call the function
        save_to_database(
            file_id=file_id,
            file_name=file_name,
            file_type=file_type,
            error_message=error_message
        )
        
        # Query the database to verify
        conn = sqlite3.connect(self.temp_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM invoice_processes WHERE file_id = ?", [file_id])
        row = cursor.fetchone()
        conn.close()
        
        # Verify the data
        self.assertIsNotNone(row)
        self.assertEqual(row["file_id"], file_id)
        self.assertEqual(row["file_name"], file_name)
        self.assertEqual(row["file_type"], file_type)
        self.assertEqual(row["error_message"], error_message)
        self.assertIsNone(row["token_count"])
        self.assertIsNone(row["request_json"])
        self.assertIsNone(row["response_json"])

    def test_history_endpoint(self):
        """Test the history endpoint."""
        # Add some test data
        for i in range(5):
            save_to_database(
                file_id=f"file-{i}",
                file_name=f"invoice-{i}.pdf",
                file_type="pdf",
                token_count=100 + i,
                request_data={"file_id": f"file-{i}"},
                response_data={"invoice": {"customer": "DEYMED"}}
            )
        
        # Test the endpoint
        response = self.client.get("/history")
        data = response.json()
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["total"], 5)
        self.assertEqual(len(data["results"]), 5)
        
        # Test with limit
        response = self.client.get("/history?limit=2")
        data = response.json()
        self.assertEqual(len(data["results"]), 2)
        
        # Test with file_id filter
        response = self.client.get("/history?file_id=file-3")
        data = response.json()
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["file_id"], "file-3")

    def test_delete_endpoint(self):
        """Test the delete endpoint."""
        # Add test data
        save_to_database(
            file_id="file-to-delete",
            file_name="delete-me.pdf",
            file_type="pdf",
            token_count=100,
            request_data={"file_id": "file-to-delete"},
            response_data={"invoice": {"customer": "DEYMED"}}
        )
        
        # Get the ID of the added record
        conn = sqlite3.connect(self.temp_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM invoice_processes WHERE file_id = ?", ["file-to-delete"])
        row = cursor.fetchone()
        record_id = row["id"]
        conn.close()
        
        # Delete the record
        response = self.client.delete(f"/history/{record_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully", response.json()["message"])
        
        # Verify it's deleted
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM invoice_processes WHERE id = ?", [record_id])
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)
        
        # Test deleting non-existent record
        response = self.client.delete(f"/history/99999")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
