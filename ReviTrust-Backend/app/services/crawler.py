import requests
import re
import hashlib
from datetime import datetime, timezone
from app.services.database import db

class CrawlerService:
    def __init__(self):
        self.tiki_headers = {
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        self.ali_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.aliexpress.com/",
            "Cookie": "aep_usuc_f=site=glo&c_tp=USD&region=VN&b_locale=en_US;"
        }
        # Thêm header Ali...

    def _get_platform(self, url: str):
        if "tiki.vn" in url: return "tiki"
        if "aliexpress" in url: return "aliexpress"
        return None

    def _extract_id(self, url: str, platform: str):
        if platform == "tiki":
            match = re.search(r'-p(\d+)\.html', url)
            return match.group(1) if match else None
        # Logic Ali...
        if platform == "aliexpress":
            match = re.search(r"item/(\d+)\.html", str(url))
            if not match: return None
            return str(match.group(1))

    def crawl(self, product_url: str):
        platform = self._get_platform(product_url)
        pid = self._extract_id(product_url, platform)
        
        if not pid:
            raise ValueError("Invalid URL or Platform")

        # 1. Crawl Product Info & Reviews (Logic from notebook)
        # Giả lập logic cào
        product_info = {"id": pid, "name": "Demo Product", "product_link": product_url} # Placeholder
        reviews = [] # Placeholder: Thực hiện requests tới Tiki/Ali API thật ở đây
        
        # 2. Save to DB
        db.upsert_product(product_info)
        # db.upsert_reviews(reviews) -> Cần transform data đúng schema trước khi save
        
        return {"status": "success", "product_id": pid, "reviews_count": len(reviews)}