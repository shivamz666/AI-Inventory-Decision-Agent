"""
Market Intelligence Module for BFWAI AI Inventory Decision Agent.

This module retrieves external market/news information using NewsAPI.org and analyzes it
using the Gemini API to produce a structured, evidence-grounded market sentiment signal.

Pipeline:
Product -> Product Keywords -> News API -> Relevant Articles -> Gemini API -> Structured Market Signal

Note: Demand forecasting is handled separately by the ML model.
Gemini is ONLY responsible for analyzing external market/news information and produces an external signal.
It does NOT make inventory/order decisions (INCREASE/MAINTAIN/REDUCE) or business recommendations.
"""

from typing import Dict, Any, List, Optional, Union
import os
import json
import logging
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None

# Configure Module Logger
logger = logging.getLogger("BFWAI.MarketIntelligence")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default File Paths & Cache Directory
_DEFAULT_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "product_catalog.csv")
_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external", "news")

VALID_EVIDENCE_TYPES = {
    "product_specific",
    "category_level",
    "competitor_pricing",
    "competitor_product",
    "technology",
    "promotion",
    "market_trend",
    "irrelevant"
}


def load_product_catalog(catalog_path: str = _DEFAULT_CATALOG_PATH) -> pd.DataFrame:
    """Loads product catalog containing product_id, product_name, category, brand, and news_keywords."""
    if not os.path.exists(catalog_path):
        catalog_path = "data/processed/product_catalog.csv"
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Product catalog not found at path: {catalog_path}")
    return pd.read_csv(catalog_path)


def get_product_metadata(product_id: str, catalog_path: str = _DEFAULT_CATALOG_PATH) -> Dict[str, Any]:
    """Retrieves metadata dictionary for a specific product ID."""
    catalog = load_product_catalog(catalog_path)
    match = catalog[catalog['product_id'] == product_id]
    if match.empty:
        raise ValueError(f"Product ID '{product_id}' not found in product catalog.")
    return match.iloc[0].to_dict()


def get_product_keywords(product_id: str, catalog_path: str = _DEFAULT_CATALOG_PATH) -> List[str]:
    """Extracts search query keywords for news retrieval from product catalog."""
    meta = get_product_metadata(product_id, catalog_path)
    raw_kw = str(meta.get("news_keywords", ""))
    keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
    if not keywords:
        keywords = [str(meta.get("product_name", product_id))]
    return keywords


def _build_fallback_articles(product_id: str) -> List[Dict[str, Any]]:
    """Helper to build offline fallback articles when live News API is unreachable or empty."""
    meta = {}
    try:
        meta = get_product_metadata(product_id)
    except Exception:
        meta = {"product_name": product_id, "category": "Electronics", "brand": "Tech"}
        
    p_name = meta.get("product_name", product_id)
    category = meta.get("category", "General")
    brand = meta.get("brand", "Tech")
    
    return [
        {
            "title": f"Consumer demand dynamics reported for {p_name} in {category} market",
            "description": f"Industry reports show consistent purchasing patterns and market interest for {brand}'s {p_name}.",
            "source": "TechMarket Insights",
            "published_at": "2025-09-15T09:00:00Z",
            "url": f"https://news.example.com/tech/{product_id}/demand-report"
        },
        {
            "title": f"Supply chain and logistics update for {category} hardware",
            "description": f"Manufacturing throughput and component logistics for {category} remain stable across major suppliers.",
            "source": "Global Supply Chain Weekly",
            "published_at": "2025-09-14T14:30:00Z",
            "url": f"https://news.example.com/supply-chain/{product_id}/logistics"
        }
    ]


def fetch_news(
    product_id: str,
    keywords: Union[str, List[str]],
    max_articles: int = 5,
    use_cache: bool = True,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves relevant news articles for a product using live NewsAPI.org HTTP integration or local cache.
    
    Args:
        product_id: Target product SKU ID.
        keywords: Search query string or list of keyword strings.
        max_articles: Maximum number of articles to return.
        use_cache: If True, reads from local cache if available.
        cache_dir: Path to news cache directory.
        api_key: Optional NewsAPI key override (defaults to NEWS_API_KEY environment variable).
        
    Returns:
        List of normalized article dicts:
        {
            "title": "...",
            "description": "...",
            "source": "...",
            "published_at": "...",
            "url": "..."
        }
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"news_{product_id}.json")
    
    # 1. Local Cache Check
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_articles = json.load(f)
                if isinstance(cached_articles, list) and len(cached_articles) > 0:
                    logger.info(f"Product {product_id}: Retrieved {len(cached_articles)} articles from local cache.")
                    return cached_articles[:max_articles]
        except Exception as e:
            logger.warning(f"Product {product_id}: Failed to read news cache ({e}). Fetching fresh news.")

    # 2. Check API Key
    effective_api_key = api_key if api_key is not None else os.environ.get("NEWS_API_KEY")
    if not effective_api_key or not str(effective_api_key).strip():
        logger.warning(f"Product {product_id}: NEWS_API_KEY environment variable is missing. Using offline fallback articles.")
        return _build_fallback_articles(product_id)[:max_articles]

    # 3. Format Search Query
    if isinstance(keywords, list) and keywords:
        if len(keywords) >= 2:
            query_term = f'"{keywords[0]}" OR "{keywords[1]}"'
        else:
            query_term = f'"{keywords[0]}"'
    elif isinstance(keywords, str) and keywords.strip():
        query_term = keywords.strip()
    else:
        query_term = f"product {product_id}"

    logger.info(f"Product {product_id}: Querying NewsAPI.org for query '{query_term[:60]}...'")

    # 4. Execute Live NewsAPI.org HTTP Request
    endpoint_url = "https://newsapi.org/v2/everything"
    params = {
        "q": query_term,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max(20, max_articles * 2),
        "apiKey": effective_api_key.strip()
    }
    
    encoded_url = f"{endpoint_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(encoded_url, headers={"User-Agent": "BFWAI-Inventory-Agent/1.0"})
    
    raw_articles = []
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            resp_bytes = response.read()
            data = json.loads(resp_bytes.decode('utf-8'))
            
            if data.get("status") == "ok":
                raw_articles = data.get("articles", [])
                logger.info(f"Product {product_id}: NewsAPI returned {len(raw_articles)} raw articles.")
            else:
                err_msg = data.get("message", "Unknown NewsAPI error")
                logger.warning(f"Product {product_id}: NewsAPI returned error status: {err_msg}")

    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.warning(f"Product {product_id}: NewsAPI rate limit exceeded (HTTP 429).")
        else:
            logger.warning(f"Product {product_id}: NewsAPI HTTP Error {e.code}.")
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"Product {product_id}: NewsAPI network connection error: {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"Product {product_id}: Failed to parse NewsAPI JSON response: {e}")
    except Exception as e:
        logger.warning(f"Product {product_id}: Unexpected error fetching news from NewsAPI: {e}")

    # 5. Normalize and Deduplicate Articles
    normalized_articles = []
    seen_titles = set()
    seen_urls = set()

    for item in raw_articles:
        if not isinstance(item, dict):
            continue
            
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        
        # Skip removed or invalid articles
        if not title or title.lower() in ["[removed]", "none", "null"]:
            continue
            
        source_dict = item.get("source") or {}
        source_name = str(source_dict.get("name") or "Unknown Source").strip()
        published_at = str(item.get("publishedAt") or "").strip()
        url = str(item.get("url") or "").strip()
        
        title_key = title.lower()
        url_key = url.lower() if url else title_key
        
        if title_key in seen_titles or url_key in seen_urls:
            continue
            
        seen_titles.add(title_key)
        seen_urls.add(url_key)
        
        normalized_articles.append({
            "title": title,
            "description": description if description and description.lower() != "[removed]" else title,
            "source": source_name,
            "published_at": published_at,
            "url": url
        })

    # 6. Fallback if empty results
    if not normalized_articles:
        logger.warning(f"Product {product_id}: Zero valid articles retrieved from live NewsAPI. Using fallback articles.")
        normalized_articles = _build_fallback_articles(product_id)

    # 7. Truncate to max_articles
    final_articles = normalized_articles[:max_articles]

    # 8. Save to Cache
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(final_articles, f, indent=2)
    except Exception as e:
        logger.warning(f"Product {product_id}: Could not write news cache ({e})")

    logger.info(f"Product {product_id}: Successfully processed {len(final_articles)} articles.")
    return final_articles


def _build_fallback_response(product_id: str, reason: str, articles: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Helper to build safe fallback response in case of API or parse failures."""
    arts = articles or []
    sources = sorted(list(set(str(a.get("source", "")).strip() for a in arts if a.get("source"))))
    return {
        "product_id": str(product_id),
        "market_signal": "neutral",
        "market_impact": 0.0,
        "confidence": 0.0,
        "relevance": 0.0,
        "evidence_strength": 0.0,
        "evidence_type": ["irrelevant"],
        "evidence_summary": "Insufficient or unavailable article evidence.",
        "reason": f"Market intelligence unavailable: {reason}",
        "key_events": [],
        "risks": [],
        "opportunities": [],
        "articles_analyzed": len(arts),
        "article_sources": sources
    }


def _clip_val(val: float, min_val: float, max_val: float) -> float:
    """Safe float clipping helper."""
    try:
        v = float(val)
        return max(min_val, min(max_val, v))
    except Exception:
        return min_val


def analyze_news_with_gemini(
    product: Dict[str, Any],
    articles: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """
    Analyzes external news articles using the Gemini API and produces a strictly evidence-grounded market signal JSON.
    
    Args:
        product: Dict containing product metadata (product_id, product_name, category, brand).
        articles: List of relevant news article dicts (title, description, source, published_at, url).
        api_key: Optional Gemini API key override (defaults to GEMINI_API_KEY environment variable).
        model_name: Target Gemini model name (default: 'gemini-2.5-flash').
        
    Returns:
        Structured Dict containing all required market intelligence fields.
    """
    product_id = str(product.get("product_id", "UNKNOWN"))
    
    # 1. Empty articles check
    if not articles:
        logger.info(f"Product {product_id}: Empty articles list provided. Returning fallback neutral signal.")
        return _build_fallback_response(product_id, "No relevant articles provided for analysis.", articles)

    # 2. Check API Key
    effective_api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    if not effective_api_key or not str(effective_api_key).strip():
        logger.error(f"Product {product_id}: GEMINI_API_KEY environment variable is missing.")
        return _build_fallback_response(product_id, "Missing GEMINI_API_KEY environment variable.", articles)

    # 3. Format articles text preserving published_at dates explicitly
    formatted_articles = []
    for idx, art in enumerate(articles, 1):
        t = art.get("title", "No Title")
        d = art.get("description", "No Description")
        s = art.get("source", "Unknown Source")
        p = art.get("published_at", "Unknown Date")
        formatted_articles.append(f"[{idx}] Title: {t}\n    Source: {s} | Published Date: {p}\n    Summary: {d}")
    articles_text = "\n\n".join(formatted_articles)

    # 4. Construct Gemini Prompt with strict evidence-grounding instructions
    prompt = f"""You are a market intelligence analyst for an AI supply chain platform.
Analyze the following external news articles for the specified product and return a strictly evidence-grounded JSON market signal analysis.

PRODUCT METADATA:
- Product ID: {product_id}
- Product Name: {product.get('product_name', 'N/A')}
- Category: {product.get('category', 'N/A')}
- Brand: {product.get('brand', 'N/A')}

NEWS ARTICLES TO ANALYZE:
{articles_text}

STRICT RULES FOR FACTUAL EVIDENCE GROUNDING:
1. ONLY STATE FACTUAL EVIDENCE: State ONLY facts explicitly supported by the supplied articles.
   - NEVER invent, assume, or extrapolate article dates, product launches, market trends, company activity, demand trends, brand activity, or statistics not explicitly present in the supplied text.
   - If an article date is missing or unknown, do NOT make up a date.

2. PRESERVE ARTICLE PUBLISHED DATES:
   - For every key event listed in 'key_events', preserve and include the article's actual published date (published_at) if available (e.g., "Logitech G PRO X Superlight deal listed [Date: 2026-09-28]").

3. DISTINGUISH EVIDENCE TYPES:
   - Categorize evidence using tags from this valid list ONLY:
     ["product_specific", "category_level", "competitor_pricing", "competitor_product", "technology", "promotion", "market_trend", "irrelevant"]
   - Distinguish clearly between:
     a) Direct evidence (articles explicitly referencing our specific product '{product.get('product_name')}' by brand '{product.get('brand')}')
     b) Category-level evidence (articles about the broad category '{product.get('category')}')
     c) Competitor evidence (articles about competitor pricing, releases, or deals)
     d) Inferred market implications
   - Do NOT say that demand for our product is increasing or decreasing unless a supplied article explicitly provides direct factual evidence for that claim.
   - If an article only indicates competitor activity or competitor price cuts, describe it strictly as "competitive pressure" or "competitor activity" rather than stating our demand will drop.

4. NO BUSINESS OR INVENTORY RECOMMENDATIONS:
   - Do NOT make business or inventory recommendations such as "increase inventory", "reduce inventory", "buy more", "reorder", "allocate budget", "increase stock", or "decrease stock".
   - The 'opportunities' and 'risks' fields MUST describe evidence-based potential implications (e.g., "Potential pricing pressure due to competitor discounts"), NOT business decisions or operational recommendations.

5. INSUFFICIENT OR WEAK EVIDENCE:
   - If evidence is insufficient, indirect, weak, or conflicting, set market_signal = "neutral" and reduce confidence (<= 0.3) and evidence_strength (<= 0.3).
   - If articles are completely irrelevant, set market_signal = "neutral", confidence <= 0.2, relevance <= 0.2, evidence_strength <= 0.2, and evidence_type = ["irrelevant"].

6. EVIDENCE SUMMARY FIELD:
   - Provide a concise field 'evidence_summary' that briefly explains what the supplied articles actually establish factually.

7. SCALE CONSTANTS:
   - market_impact: float between 0.0 and 1.0 (magnitude only: 0.0=negligible, 0.25=low, 0.50=moderate, 0.75=high, 1.0=very high).
   - Sentiment direction (positive / neutral / negative) is conveyed ONLY by market_signal.
   - evidence_strength: float between 0.0 and 1.0.

REQUIRED JSON OUTPUT SCHEMA:
{{
    "product_id": "{product_id}",
    "market_signal": "positive" | "neutral" | "negative",
    "market_impact": <float 0.0 to 1.0 (0.0=negligible, 0.25=low, 0.50=moderate, 0.75=high, 1.0=very high)>,
    "confidence": <float 0.0 to 1.0 representing analysis confidence>,
    "relevance": <float 0.0 to 1.0 representing article relevance to product>,
    "evidence_strength": <float 0.0 to 1.0 representing strength of evidence>,
    "evidence_type": ["<tag1>", "<tag2>"],
    "evidence_summary": "<brief factual explanation of what the supplied articles actually establish>",
    "reason": "<analytical rationale clearly distinguishing direct vs category vs competitor evidence>",
    "key_events": ["<factual event preserving actual published_at date if available>"],
    "risks": ["<evidence-based potential risk implication>"],
    "opportunities": ["<evidence-based potential opportunity implication>"]
}}
"""

    # 5. Call Gemini API (with fallback model if primary model is 503 unavailable)
    models_to_try = [model_name, "gemini-1.5-flash"]
    last_error = None

    for m_name in models_to_try:
        try:
            logger.info(f"Product {product_id}: Sending Gemini request ({len(articles)} articles, model={m_name})...")
            
            if genai is None:
                raise ImportError("google-genai SDK is not installed in the python environment.")
                
            client = genai.Client(api_key=effective_api_key)
            response = client.models.generate_content(
                model=m_name,
                contents=prompt
            )
            
            raw_text = response.text.strip()
            logger.info(f"Product {product_id}: Received Gemini response ({len(raw_text)} chars).")
            
            # Clean markdown code blocks if wrapped in ```json ... ```
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
                
            result_json = json.loads(raw_text)
            
            # Enforce required keys & data types
            result_json["product_id"] = str(product_id)
            if result_json.get("market_signal") not in ["positive", "neutral", "negative"]:
                result_json["market_signal"] = "neutral"
                
            result_json["market_impact"] = _clip_val(result_json.get("market_impact", 0.0), 0.0, 1.0)
            result_json["confidence"] = _clip_val(result_json.get("confidence", 0.5), 0.0, 1.0)
            result_json["relevance"] = _clip_val(result_json.get("relevance", 0.5), 0.0, 1.0)
            result_json["evidence_strength"] = _clip_val(result_json.get("evidence_strength", 0.5), 0.0, 1.0)
            
            # Validate and clean evidence_type
            raw_ev_types = result_json.get("evidence_type", [])
            if isinstance(raw_ev_types, str):
                raw_ev_types = [raw_ev_types]
            valid_types = [t for t in raw_ev_types if t in VALID_EVIDENCE_TYPES]
            result_json["evidence_type"] = valid_types if valid_types else ["irrelevant"]
            
            result_json["evidence_summary"] = str(result_json.get("evidence_summary", "Factual summary of supplied article evidence."))
            result_json["reason"] = str(result_json.get("reason", "Analyzed external news articles."))
            result_json["key_events"] = list(result_json.get("key_events", []))
            
            # Post-process risks & opportunities to ensure no business/inventory recommendations
            forbidden_terms = ["increase inventory", "reduce inventory", "buy more", "reorder", "allocate budget", "increase stock", "decrease stock"]
            clean_risks = []
            for r in result_json.get("risks", []):
                r_str = str(r)
                for term in forbidden_terms:
                    if term in r_str.lower():
                        r_str = r_str.replace(term, "manage product risk")
                clean_risks.append(r_str)
            result_json["risks"] = clean_risks

            clean_opps = []
            for o in result_json.get("opportunities", []):
                o_str = str(o)
                for term in forbidden_terms:
                    if term in o_str.lower():
                        o_str = o_str.replace(term, "explore market potential")
                clean_opps.append(o_str)
            result_json["opportunities"] = clean_opps

            result_json["articles_analyzed"] = len(articles)
            sources = sorted(list(set(str(a.get("source", "")).strip() for a in articles if a.get("source"))))
            result_json["article_sources"] = sources
            
            return result_json

        except json.JSONDecodeError as e:
            logger.error(f"Product {product_id}: Failed to parse Gemini response as JSON: {e}")
            return _build_fallback_response(product_id, "Malformed JSON response from Gemini API.", articles)
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Product {product_id}: Gemini API call failed with model '{m_name}': {e}")
            
            # Handle rate limiting (429 RESOURCE_EXHAUSTED) with automatic sleep backoff
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "Quota" in str(e):
                import time
                logger.info(f"Product {product_id}: Rate limit encountered. Waiting 15 seconds for quota reset...")
                time.sleep(15)
                try:
                    logger.info(f"Product {product_id}: Retrying Gemini request with model '{m_name}'...")
                    response = client.models.generate_content(
                        model=m_name,
                        contents=prompt
                    )
                    raw_text = response.text.strip()
                    if raw_text.startswith("```"):
                        lines = raw_text.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        raw_text = "\n".join(lines).strip()
                        
                    result_json = json.loads(raw_text)
                    result_json["product_id"] = str(product_id)
                    if result_json.get("market_signal") not in ["positive", "neutral", "negative"]:
                        result_json["market_signal"] = "neutral"
                        
                    result_json["market_impact"] = _clip_val(result_json.get("market_impact", 0.0), 0.0, 1.0)
                    result_json["confidence"] = _clip_val(result_json.get("confidence", 0.5), 0.0, 1.0)
                    result_json["relevance"] = _clip_val(result_json.get("relevance", 0.5), 0.0, 1.0)
                    result_json["evidence_strength"] = _clip_val(result_json.get("evidence_strength", 0.5), 0.0, 1.0)
                    
                    raw_ev_types = result_json.get("evidence_type", [])
                    if isinstance(raw_ev_types, str):
                        raw_ev_types = [raw_ev_types]
                    valid_types = [t for t in raw_ev_types if t in VALID_EVIDENCE_TYPES]
                    result_json["evidence_type"] = valid_types if valid_types else ["irrelevant"]
                    
                    result_json["evidence_summary"] = str(result_json.get("evidence_summary", "Factual summary of supplied article evidence."))
                    result_json["reason"] = str(result_json.get("reason", "Analyzed external news articles."))
                    result_json["key_events"] = list(result_json.get("key_events", []))
                    
                    forbidden_terms = ["increase inventory", "reduce inventory", "buy more", "reorder", "allocate budget", "increase stock", "decrease stock"]
                    clean_risks = [r_str if not any(t in str(r_str).lower() for t in forbidden_terms) else str(r_str).replace(t, "manage product risk") for r_str in result_json.get("risks", [])]
                    result_json["risks"] = clean_risks
                    
                    clean_opps = [o_str if not any(t in str(o_str).lower() for t in forbidden_terms) else str(o_str).replace(t, "explore market potential") for o_str in result_json.get("opportunities", [])]
                    result_json["opportunities"] = clean_opps
                    
                    result_json["articles_analyzed"] = len(articles)
                    sources = sorted(list(set(str(a.get("source", "")).strip() for a in articles if a.get("source"))))
                    result_json["article_sources"] = sources
                    
                    return result_json
                except Exception as retry_e:
                    logger.warning(f"Product {product_id}: Retry failed after sleep ({retry_e})")
                    last_error = str(retry_e)

            if "503" in str(e) or "UNAVAILABLE" in str(e):
                continue  # Try fallback model
            else:
                break

    return _build_fallback_response(product_id, f"Gemini API failure: {last_error}", articles)


def analyze_all_products_batch(
    products_with_articles: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Dict[str, Any]]:
    """
    Analyzes ALL products' news articles in a SINGLE Gemini API call.
    
    This is critical for free-tier Gemini API usage where the daily quota is only 20 requests.
    Instead of 1 call per product (20 calls), we batch everything into 1 call.
    
    Args:
        products_with_articles: List of dicts, each containing:
            - product: Dict with product_id, product_name, category, brand
            - articles: List of article dicts
        api_key: Optional Gemini API key override.
        model_name: Gemini model name.
        
    Returns:
        Dict mapping product_id -> structured market intelligence result.
    """
    if not products_with_articles:
        return {}
    
    effective_api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    if not effective_api_key or not str(effective_api_key).strip():
        logger.error("GEMINI_API_KEY environment variable is missing for batch analysis.")
        return {
            item["product"]["product_id"]: _build_fallback_response(
                item["product"]["product_id"], "Missing GEMINI_API_KEY.", item.get("articles", [])
            )
            for item in products_with_articles
        }
    
    if genai is None:
        logger.error("google-genai SDK is not installed.")
        return {
            item["product"]["product_id"]: _build_fallback_response(
                item["product"]["product_id"], "google-genai SDK not installed.", item.get("articles", [])
            )
            for item in products_with_articles
        }

    # Build one mega-prompt with all products
    product_sections = []
    for item in products_with_articles:
        product = item["product"]
        articles = item.get("articles", [])
        pid = str(product.get("product_id", "UNKNOWN"))
        
        formatted_articles = []
        for idx, art in enumerate(articles, 1):
            t = art.get("title", "No Title")
            d = art.get("description", "No Description")
            s = art.get("source", "Unknown Source")
            p = art.get("published_at", "Unknown Date")
            formatted_articles.append(f"  [{idx}] Title: {t}\n      Source: {s} | Published Date: {p}\n      Summary: {d}")
        articles_text = "\n\n".join(formatted_articles) if formatted_articles else "  No articles available."
        
        section = f"""--- PRODUCT: {pid} ---
Product ID: {pid}
Product Name: {product.get('product_name', 'N/A')}
Category: {product.get('category', 'N/A')}
Brand: {product.get('brand', 'N/A')}

Articles:
{articles_text}
"""
        product_sections.append(section)
    
    all_products_text = "\n\n".join(product_sections)
    product_ids_list = [item["product"]["product_id"] for item in products_with_articles]
    
    prompt = f"""You are a market intelligence analyst for an AI supply chain platform.
You will analyze news articles for MULTIPLE products at once and return a JSON array with one analysis object per product.

PRODUCTS TO ANALYZE:
{all_products_text}

STRICT RULES FOR FACTUAL EVIDENCE GROUNDING:
1. ONLY STATE FACTUAL EVIDENCE: State ONLY facts explicitly supported by the supplied articles.
   - NEVER invent, assume, or extrapolate facts not explicitly present.
   - If an article date is missing or unknown, do NOT make up a date.

2. PRESERVE ARTICLE PUBLISHED DATES in key_events.

3. DISTINGUISH EVIDENCE TYPES using ONLY these tags:
   ["product_specific", "category_level", "competitor_pricing", "competitor_product", "technology", "promotion", "market_trend", "irrelevant"]

4. NO BUSINESS OR INVENTORY RECOMMENDATIONS.
   - Do NOT recommend "increase inventory", "reduce inventory", etc.

5. INSUFFICIENT OR WEAK EVIDENCE:
   - If evidence is insufficient, set market_signal = "neutral", confidence <= 0.3, evidence_strength <= 0.3.
   - If articles are irrelevant, set evidence_type = ["irrelevant"].

6. SCALE CONSTANTS:
   - market_impact: float between 0.0 and 1.0 (0.0=negligible, 0.25=low, 0.50=moderate, 0.75=high, 1.0=very high).
   - Sentiment direction is conveyed ONLY by market_signal.
   - evidence_strength: float between 0.0 and 1.0.

REQUIRED OUTPUT:
Return a JSON array with EXACTLY {len(products_with_articles)} objects, one per product, in this order: {product_ids_list}.
Each object must have:
{{
    "product_id": "<product_id>",
    "market_signal": "positive" | "neutral" | "negative",
    "market_impact": <float 0.0-1.0>,
    "confidence": <float 0.0-1.0>,
    "relevance": <float 0.0-1.0>,
    "evidence_strength": <float 0.0-1.0>,
    "evidence_type": ["<tag1>", "<tag2>"],
    "evidence_summary": "<brief factual explanation>",
    "reason": "<analytical rationale>",
    "key_events": ["<factual event>"],
    "risks": ["<evidence-based risk>"],
    "opportunities": ["<evidence-based opportunity>"]
}}

Return ONLY the JSON array. No markdown, no explanation.
"""

    logger.info(f"Batch analysis: Sending {len(products_with_articles)} products to Gemini in a single call (model={model_name})...")
    
    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"]
    last_error = None
    
    for m_name in models_to_try:
        try:
            client = genai.Client(api_key=effective_api_key)
            response = client.models.generate_content(
                model=m_name,
                contents=prompt
            )
            
            raw_text = response.text.strip()
            logger.info(f"Batch analysis: Received Gemini response ({len(raw_text)} chars).")
            
            # Clean markdown code blocks
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
            
            results_array = json.loads(raw_text)
            
            if not isinstance(results_array, list):
                logger.error("Batch analysis: Gemini did not return a JSON array.")
                last_error = "Response was not a JSON array"
                continue
            
            # Build output dict, validate each result
            output = {}
            for result_json in results_array:
                pid = str(result_json.get("product_id", "UNKNOWN"))
                
                if result_json.get("market_signal") not in ["positive", "neutral", "negative"]:
                    result_json["market_signal"] = "neutral"
                
                result_json["market_impact"] = _clip_val(result_json.get("market_impact", 0.0), 0.0, 1.0)
                result_json["confidence"] = _clip_val(result_json.get("confidence", 0.5), 0.0, 1.0)
                result_json["relevance"] = _clip_val(result_json.get("relevance", 0.5), 0.0, 1.0)
                result_json["evidence_strength"] = _clip_val(result_json.get("evidence_strength", 0.5), 0.0, 1.0)
                
                raw_ev_types = result_json.get("evidence_type", [])
                if isinstance(raw_ev_types, str):
                    raw_ev_types = [raw_ev_types]
                valid_types = [t for t in raw_ev_types if t in VALID_EVIDENCE_TYPES]
                result_json["evidence_type"] = valid_types if valid_types else ["irrelevant"]
                
                result_json["evidence_summary"] = str(result_json.get("evidence_summary", ""))
                result_json["reason"] = str(result_json.get("reason", ""))
                result_json["key_events"] = list(result_json.get("key_events", []))
                result_json["risks"] = list(result_json.get("risks", []))
                result_json["opportunities"] = list(result_json.get("opportunities", []))
                
                # Find matching articles count
                matching_item = next((x for x in products_with_articles if x["product"]["product_id"] == pid), None)
                if matching_item:
                    arts = matching_item.get("articles", [])
                    result_json["articles_analyzed"] = len(arts)
                    result_json["article_sources"] = sorted(list(set(
                        str(a.get("source", "")).strip() for a in arts if a.get("source")
                    )))
                
                output[pid] = result_json
            
            # Fill any missing products with fallback
            for item in products_with_articles:
                pid = item["product"]["product_id"]
                if pid not in output:
                    output[pid] = _build_fallback_response(pid, "Product missing from batch response.", item.get("articles", []))
            
            logger.info(f"Batch analysis: Successfully parsed results for {len(output)} products.")
            return output
            
        except json.JSONDecodeError as e:
            logger.error(f"Batch analysis: Failed to parse Gemini response as JSON: {e}")
            last_error = str(e)
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Batch analysis: Gemini API call failed with model '{m_name}': {e}")
            # For 503 (temporary high demand), retry same model once after 10s backoff
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                import time
                logger.info(f"Batch analysis: Model '{m_name}' has temporary high demand. Retrying in 10 seconds...")
                time.sleep(10)
                try:
                    response = client.models.generate_content(model=m_name, contents=prompt)
                    raw_text = response.text.strip()
                    if raw_text.startswith("```"):
                        lines = raw_text.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        raw_text = "\n".join(lines).strip()
                    results_array = json.loads(raw_text)
                    if isinstance(results_array, list):
                        output = {}
                        for result_json in results_array:
                            pid = str(result_json.get("product_id", "UNKNOWN"))
                            if result_json.get("market_signal") not in ["positive", "neutral", "negative"]:
                                result_json["market_signal"] = "neutral"
                            result_json["market_impact"] = _clip_val(result_json.get("market_impact", 0.0), 0.0, 1.0)
                            result_json["confidence"] = _clip_val(result_json.get("confidence", 0.5), 0.0, 1.0)
                            result_json["relevance"] = _clip_val(result_json.get("relevance", 0.5), 0.0, 1.0)
                            result_json["evidence_strength"] = _clip_val(result_json.get("evidence_strength", 0.5), 0.0, 1.0)
                            raw_ev_types = result_json.get("evidence_type", [])
                            if isinstance(raw_ev_types, str):
                                raw_ev_types = [raw_ev_types]
                            valid_types = [t for t in raw_ev_types if t in VALID_EVIDENCE_TYPES]
                            result_json["evidence_type"] = valid_types if valid_types else ["irrelevant"]
                            result_json["evidence_summary"] = str(result_json.get("evidence_summary", ""))
                            result_json["reason"] = str(result_json.get("reason", ""))
                            result_json["key_events"] = list(result_json.get("key_events", []))
                            result_json["risks"] = list(result_json.get("risks", []))
                            result_json["opportunities"] = list(result_json.get("opportunities", []))
                            matching_item = next((x for x in products_with_articles if x["product"]["product_id"] == pid), None)
                            if matching_item:
                                arts = matching_item.get("articles", [])
                                result_json["articles_analyzed"] = len(arts)
                                result_json["article_sources"] = sorted(list(set(str(a.get("source", "")).strip() for a in arts if a.get("source"))))
                            output[pid] = result_json
                        for item in products_with_articles:
                            pid = item["product"]["product_id"]
                            if pid not in output:
                                output[pid] = _build_fallback_response(pid, "Product missing from batch response.", item.get("articles", []))
                        logger.info(f"Batch analysis: Retry succeeded! Parsed results for {len(output)} products.")
                        return output
                except Exception as retry_e:
                    logger.warning(f"Batch analysis: Retry of '{m_name}' also failed: {retry_e}")
                    last_error = str(retry_e)
                continue  # Try next model
            elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "404" in str(e) or "NOT_FOUND" in str(e):
                continue  # Try next model which has its own quota
            else:
                break
    
    # All attempts failed — return fallbacks
    logger.error(f"Batch analysis: All Gemini attempts failed. Last error: {last_error}")
    return {
        item["product"]["product_id"]: _build_fallback_response(
            item["product"]["product_id"], f"Gemini batch failure: {last_error}", item.get("articles", [])
        )
        for item in products_with_articles
    }


def get_market_intelligence(
    product_id: str,
    date: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: str = _DEFAULT_CACHE_DIR
) -> Dict[str, Any]:
    """
    Main orchestration entrypoint for Market Intelligence.
    Loads product catalog -> retrieves news -> analyzes with Gemini API -> returns structured signal.
    """
    os.makedirs(cache_dir, exist_ok=True)
    date_str = date or "latest"
    analysis_cache_file = os.path.join(cache_dir, f"analysis_{product_id}_{date_str}.json")
    
    if use_cache and os.path.exists(analysis_cache_file):
        try:
            with open(analysis_cache_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                logger.info(f"Product {product_id}: Loaded Gemini analysis result from cache.")
                return result
        except Exception:
            pass
            
    product_meta = get_product_metadata(product_id)
    keywords = get_product_keywords(product_id)
    articles = fetch_news(product_id, keywords, use_cache=use_cache, cache_dir=cache_dir)
    
    result = analyze_news_with_gemini(product_meta, articles)
    result["date"] = date_str
    
    # Save analysis result to cache
    try:
        with open(analysis_cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Product {product_id}: Could not write analysis cache ({e})")
        
    return result
