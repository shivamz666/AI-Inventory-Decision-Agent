"""
Unit tests for Gemini Market Intelligence and live NewsAPI integration (src/market_intelligence.py).
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import urllib.error

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.market_intelligence import (
    load_product_catalog,
    get_product_metadata,
    get_product_keywords,
    fetch_news,
    analyze_news_with_gemini,
    get_market_intelligence,
    VALID_EVIDENCE_TYPES
)


class TestMarketIntelligence(unittest.TestCase):

    def setUp(self):
        self.sample_product = {
            "product_id": "P001",
            "product_name": "Wireless Gaming Mouse",
            "category": "Computer Accessories",
            "brand": "NovaTech"
        }
        self.sample_articles = [
            {
                "title": "NovaTech launches new gaming gear line",
                "description": "High demand expected for new low-latency wireless mice.",
                "source": "TechDaily",
                "published_at": "2025-09-15T10:00:00Z",
                "url": "https://news.example.com/p001"
            },
            {
                "title": "Computer accessories sales surge in Q3",
                "description": "Peripherals market shows positive trajectory.",
                "source": "Hardware Weekly",
                "published_at": "2025-09-14T08:30:00Z",
                "url": "https://news.example.com/hardware"
            }
        ]

    # --- CATALOG & KEYWORDS TESTS ---

    def test_product_catalog_and_keywords(self):
        """Test loading product metadata and generating news search keywords."""
        meta = get_product_metadata("P001")
        self.assertEqual(meta["product_id"], "P001")
        self.assertIn("product_name", meta)
        
        keywords = get_product_keywords("P001")
        self.assertIsInstance(keywords, list)
        self.assertTrue(len(keywords) > 0)

    # --- NEWS API TESTS ---

    @patch("urllib.request.urlopen")
    def test_successful_newsapi_response(self, mock_urlopen):
        """Test parsing successful NewsAPI.org response into normalized article list."""
        mock_response = MagicMock()
        mock_data = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "title": "New Gaming Mouse Launched",
                    "description": "NovaTech introduces high precision gaming mouse.",
                    "source": {"id": "tech-crunch", "name": "TechCrunch"},
                    "publishedAt": "2025-09-20T12:00:00Z",
                    "url": "https://techcrunch.com/mouse-launch"
                },
                {
                    "title": "Peripheral Market Trends 2025",
                    "description": "Demand for wireless accessories increases.",
                    "source": {"id": "reuters", "name": "Reuters"},
                    "publishedAt": "2025-09-19T15:30:00Z",
                    "url": "https://reuters.com/peripherals"
                }
            ]
        }
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        articles = fetch_news("P001", ["Wireless Gaming Mouse"], max_articles=2, use_cache=False, api_key="dummy_news_key")
        
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0]["title"], "New Gaming Mouse Launched")
        self.assertEqual(articles[0]["source"], "TechCrunch")
        self.assertEqual(articles[0]["url"], "https://techcrunch.com/mouse-launch")
        self.assertEqual(articles[1]["title"], "Peripheral Market Trends 2025")

    @patch("urllib.request.urlopen")
    def test_empty_newsapi_response(self, mock_urlopen):
        """Test that empty NewsAPI response falls back safely without crashing."""
        mock_response = MagicMock()
        mock_data = {"status": "ok", "totalResults": 0, "articles": []}
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        articles = fetch_news("P001", ["UnmatchedNonExistentQuery"], max_articles=2, use_cache=False, api_key="dummy_news_key")
        self.assertIsInstance(articles, list)
        self.assertTrue(len(articles) > 0)

    @patch("urllib.request.urlopen")
    def test_newsapi_error_response(self, mock_urlopen):
        """Test handling NewsAPI error response status."""
        mock_response = MagicMock()
        mock_data = {"status": "error", "code": "apiKeyInvalid", "message": "Your API key is invalid."}
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        articles = fetch_news("P001", ["Wireless Gaming Mouse"], max_articles=2, use_cache=False, api_key="invalid_key")
        self.assertIsInstance(articles, list)
        self.assertTrue(len(articles) > 0)

    @patch("urllib.request.urlopen")
    def test_newsapi_rate_limit(self, mock_urlopen):
        """Test handling HTTP 429 rate limit error gracefully."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://newsapi.org", code=429, msg="Too Many Requests", hdrs={}, fp=None
        )
        articles = fetch_news("P001", ["Wireless Gaming Mouse"], max_articles=2, use_cache=False, api_key="dummy_key")
        self.assertIsInstance(articles, list)
        self.assertTrue(len(articles) > 0)

    @patch.dict(os.environ, {"NEWS_API_KEY": ""}, clear=False)
    def test_missing_news_api_key(self):
        """Test fallback when NEWS_API_KEY is missing."""
        articles = fetch_news("P001", ["Wireless Gaming Mouse"], max_articles=2, use_cache=False, api_key="")
        self.assertIsInstance(articles, list)
        self.assertTrue(len(articles) > 0)

    @patch("urllib.request.urlopen")
    def test_malformed_newsapi_response(self, mock_urlopen):
        """Test handling non-JSON malformed HTTP response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>502 Bad Gateway</html>"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        articles = fetch_news("P001", ["Wireless Gaming Mouse"], max_articles=2, use_cache=False, api_key="dummy_key")
        self.assertIsInstance(articles, list)
        self.assertTrue(len(articles) > 0)

    @patch("urllib.request.urlopen")
    def test_duplicate_articles_deduplication(self, mock_urlopen):
        """Test that duplicate articles are filtered out."""
        mock_response = MagicMock()
        mock_data = {
            "status": "ok",
            "articles": [
                {"title": "Duplicate Title", "description": "Desc 1", "source": {"name": "S1"}, "publishedAt": "2025-09-20", "url": "https://dup.com"},
                {"title": "Duplicate Title", "description": "Desc 2", "source": {"name": "S2"}, "publishedAt": "2025-09-20", "url": "https://dup.com"},
                {"title": "Unique Title", "description": "Desc 3", "source": {"name": "S3"}, "publishedAt": "2025-09-19", "url": "https://unique.com"}
            ]
        }
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        articles = fetch_news("P001", ["Wireless Gaming Mouse"], max_articles=5, use_cache=False, api_key="dummy_key")
        self.assertEqual(len(articles), 2)
        titles = [a["title"] for a in articles]
        self.assertEqual(titles.count("Duplicate Title"), 1)

    # --- GEMINI API TESTS ---

    def test_empty_articles_gemini(self):
        """Test that passing an empty articles list to Gemini returns a safe fallback neutral response."""
        res = analyze_news_with_gemini(self.sample_product, [], api_key="dummy_key")
        self.assertEqual(res["product_id"], "P001")
        self.assertEqual(res["market_signal"], "neutral")
        self.assertEqual(res["market_impact"], 0.0)
        self.assertEqual(res["confidence"], 0.0)
        self.assertEqual(res["relevance"], 0.0)
        self.assertEqual(res["evidence_strength"], 0.0)
        self.assertIn("irrelevant", res["evidence_type"])
        self.assertIn("evidence_summary", res)
        self.assertIn("unavailable", res["reason"].lower())

    def test_missing_gemini_api_key(self):
        """Test that missing GEMINI_API_KEY triggers a safe structured fallback without crashing."""
        res = analyze_news_with_gemini(self.sample_product, self.sample_articles, api_key="")
        self.assertEqual(res["product_id"], "P001")
        self.assertEqual(res["market_signal"], "neutral")
        self.assertEqual(res["confidence"], 0.0)
        self.assertIn("missing gemini_api_key", res["reason"].lower())

    @patch("src.market_intelligence.genai.Client")
    def test_valid_gemini_response(self, mock_client_cls):
        """Test parsing valid Gemini API response with evidence_summary into structured schema."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        
        valid_json_payload = json.dumps({
            "product_id": "P001",
            "market_signal": "positive",
            "market_impact": 0.50,
            "confidence": 0.85,
            "relevance": 0.90,
            "evidence_strength": 0.80,
            "evidence_type": ["product_specific", "technology"],
            "evidence_summary": "Articles confirm NovaTech launched a new product line with positive reception.",
            "reason": "Strong direct evidence found in product launch announcements.",
            "key_events": ["NovaTech product line launch [Date: 2025-09-15]"],
            "risks": ["Potential market competition"],
            "opportunities": ["Q4 holiday demand surge"]
        })
        
        mock_response.text = f"```json\n{valid_json_payload}\n```"
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client
        
        res = analyze_news_with_gemini(self.sample_product, self.sample_articles, api_key="test_key")
        
        self.assertEqual(res["product_id"], "P001")
        self.assertEqual(res["market_signal"], "positive")
        self.assertEqual(res["market_impact"], 0.50)
        self.assertEqual(res["confidence"], 0.85)
        self.assertEqual(res["relevance"], 0.90)
        self.assertEqual(res["evidence_strength"], 0.80)
        self.assertIn("product_specific", res["evidence_type"])
        self.assertIn("Articles confirm", res["evidence_summary"])
        self.assertIsInstance(res["key_events"], list)
        self.assertIsInstance(res["risks"], list)
        self.assertIsInstance(res["opportunities"], list)

    @patch("src.market_intelligence.genai.Client")
    def test_competitor_pricing_evidence(self, mock_client_cls):
        """Test evidence classification when articles contain competitor pricing or resale deals."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        
        json_payload = json.dumps({
            "product_id": "P001",
            "market_signal": "negative",
            "market_impact": 0.50,
            "confidence": 0.30,
            "relevance": 0.60,
            "evidence_strength": 0.40,
            "evidence_type": ["competitor_pricing", "competitor_product"],
            "evidence_summary": "Articles report competitor discount deals on Slickdeals.",
            "reason": "Articles describe competitive pressure from competitor discounts on resale listings.",
            "key_events": ["Competitor Logitech G305 discount listed [Date: 2026-09-24]"],
            "risks": ["Price matching pressure"],
            "opportunities": []
        })
        
        mock_response.text = json_payload
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client
        
        res = analyze_news_with_gemini(self.sample_product, self.sample_articles, api_key="test_key")
        self.assertEqual(res["market_signal"], "negative")
        self.assertEqual(res["market_impact"], 0.50)
        self.assertLessEqual(res["confidence"], 0.30)
        self.assertIn("competitor_pricing", res["evidence_type"])
        self.assertIn("competitive pressure", res["reason"].lower())

    @patch("src.market_intelligence.genai.Client")
    def test_malformed_gemini_response(self, mock_client_cls):
        """Test graceful error handling when Gemini returns invalid non-JSON text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is plain text response not valid JSON format."
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client
        
        res = analyze_news_with_gemini(self.sample_product, self.sample_articles, api_key="test_key")
        self.assertEqual(res["product_id"], "P001")
        self.assertEqual(res["market_signal"], "neutral")
        self.assertEqual(res["confidence"], 0.0)
        self.assertIn("malformed json response", res["reason"].lower())

    @patch("src.market_intelligence.genai.Client")
    def test_gemini_api_failure(self, mock_client_cls):
        """Test fallback handling when Gemini API client raises an exception."""
        mock_client_cls.side_effect = RuntimeError("API Connection Timeout")
        
        res = analyze_news_with_gemini(self.sample_product, self.sample_articles, api_key="test_key")
        self.assertEqual(res["product_id"], "P001")
        self.assertEqual(res["market_signal"], "neutral")
        self.assertEqual(res["confidence"], 0.0)
        self.assertIn("gemini api failure", res["reason"].lower())


if __name__ == "__main__":
    unittest.main()
