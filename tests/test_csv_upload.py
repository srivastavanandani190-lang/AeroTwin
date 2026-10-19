"""
AeroTwin - Custom CSV Upload & Streamlined Data Control Toolbar Tests
====================================================================
Tests for:
1. Removal of cluttered inputs (CITY, START DATE, END DATE, PREVIEW ROWS).
2. Modern CSV upload zone accepting strictly .csv files.
3. Streamlined action buttons ("Process Dataset & Run Twin", "Export Cleaned CSV").
4. Dynamic binding of uploaded CSV to global state, charts, and twin forecasting.
5. REST API endpoint /api/upload-csv validation and edge cases.
"""

import io
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import create_flask_app, PROCESSED_DIR


class TestCsvUploadAndToolbarRefactor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_flask_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_toolbar_clutter_removed(self):
        """Verifies that CITY dropdown, START DATE, END DATE, and PREVIEW ROWS are removed."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Confirm old inputs are completely removed
        self.assertNotIn('for="city"', html)
        self.assertNotIn('for="start_date"', html)
        self.assertNotIn('for="end_date"', html)
        self.assertNotIn('for="limit"', html)
        self.assertNotIn('PREVIEW ROWS', html)

    def test_upload_zone_and_action_buttons_rendered(self):
        """Verifies modern upload zone, Process Dataset & Run Twin, and Export Cleaned CSV."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Upload dropzone and file input
        self.assertIn('csv-upload-dropzone', html)
        self.assertIn('name="csv_file"', html)
        self.assertIn('accept=".csv"', html)

        # Action buttons
        self.assertIn("Process Dataset &amp; Run Twin", html)
        self.assertIn("Export Cleaned CSV", html)
        self.assertIn('btn-process-twin', html)
        self.assertIn('btn-export-csv', html)

    def test_post_csv_upload_to_dashboard(self):
        """Verifies uploading custom CSV via POST / updates dashboard state and twin forecasts."""
        sample_csv_data = (
            "Date,City,PM2.5,PM10,NO2,SO2,CO,O3,AQI\n"
            "2026-01-01,CustomCity,145.2,210.5,35.0,18.0,1.8,42.0,295.0\n"
            "2026-01-02,CustomCity,178.6,245.0,42.0,22.0,2.1,48.0,335.0\n"
            "2026-01-03,CustomCity,215.0,290.0,50.0,25.0,2.5,52.0,380.0\n"
        )
        file_storage = (io.BytesIO(sample_csv_data.encode("utf-8")), "custom_sensor_stream.csv")

        res = self.client.post(
            "/",
            data={"csv_file": file_storage},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Verify recognized filename and active state
        self.assertIn("custom_sensor_stream.csv", html)
        self.assertIn("Custom Ingested", html)

        # Cleanup uploaded test file
        test_file = os.path.join(PROCESSED_DIR, "custom_sensor_stream.csv")
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_api_upload_csv_endpoint(self):
        """Verifies /api/upload-csv REST endpoint."""
        sample_csv = "Date,PM2.5,AQI\n2026-01-01,95.0,180.0\n2026-01-02,110.0,210.0\n"
        file_storage = (io.BytesIO(sample_csv.encode("utf-8")), "api_test_feed.csv")

        res = self.client.post(
            "/api/upload-csv",
            data={"csv_file": file_storage},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["filename"], "api_test_feed.csv")
        self.assertEqual(data["rows"], 2)
        self.assertIn("PM2.5", data["columns"])

        # Non-csv rejection
        bad_storage = (io.BytesIO(b"bad content"), "malicious.exe")
        bad_res = self.client.post(
            "/api/upload-csv",
            data={"csv_file": bad_storage},
            content_type="multipart/form-data"
        )
        self.assertEqual(bad_res.status_code, 400)

        # Cleanup
        test_file = os.path.join(PROCESSED_DIR, "api_test_feed.csv")
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_header_profile_icon_removed(self):
        """Verifies that the user/avatar profile icon is completely removed from the header."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertNotIn('class="user-avatar"', html)
        self.assertNotIn('title="Environmental Intelligence Officer"', html)

    def test_header_location_badge_dynamic(self):
        """Verifies that the header weather/location badge dynamically updates with dataset or region."""
        # 1. Test region Ghaziabad
        sample_ghaziabad_csv = (
            "Date,City,PM2.5,AQI\n"
            "2026-01-01,Ghaziabad,195.0,340.0\n"
        )
        file_storage = (io.BytesIO(sample_ghaziabad_csv.encode("utf-8")), "ghaziabad_feed.csv")
        res = self.client.post("/", data={"csv_file": file_storage}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn("Ghaziabad", html)
        self.assertIn("Hazy Smog", html)

        # Cleanup
        test_file = os.path.join(PROCESSED_DIR, "ghaziabad_feed.csv")
        if os.path.exists(test_file):
            os.remove(test_file)

        # 2. Test region Ahmedabad
        sample_ahmedabad_csv = (
            "Date,City,PM2.5,AQI\n"
            "2026-01-01,Ahmedabad,72.0,140.0\n"
        )
        file_storage2 = (io.BytesIO(sample_ahmedabad_csv.encode("utf-8")), "ahmedabad_feed.csv")
        res2 = self.client.post("/", data={"csv_file": file_storage2}, content_type="multipart/form-data")
        self.assertEqual(res2.status_code, 200)
        html2 = res2.get_data(as_text=True)

        self.assertIn("Ahmedabad", html2)
        self.assertIn("Clear Sky", html2)

        # Cleanup
        test_file2 = os.path.join(PROCESSED_DIR, "ahmedabad_feed.csv")
        if os.path.exists(test_file2):
            os.remove(test_file2)


if __name__ == "__main__":
    unittest.main()
