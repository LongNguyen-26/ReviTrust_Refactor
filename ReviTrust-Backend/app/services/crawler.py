import requests
import re
import hashlib
from datetime import datetime, timezone
from app.services.database import db
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil import parser
from datetime import datetime, timezone
import time
import random

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

    def _get_product_info(self, pid: str, product_url: str, platform: str):
        if platform == "tiki":
            try:
                res = requests.get(product_url, headers=self.tiki_headers)
                data = res.json()
                return {
                    "id": str(data.get("id")),
                    "name": data.get("name", "Unknown Product"),
                    "shop_name": data.get("current_seller", {}).get("name", "Unknown Shop"),
                    "product_link": f"https://tiki.vn/{data.get('url_path', f'p{pid}.html')}",
                    "product_images": [img.get("base_url") for img in data.get("images", [])]
                }
            except: return None

        if platform == "aliexpress":
            try:
                res = requests.get(product_url, headers=self.ali_headers, timeout=15)
                html = res.text
                title_match = re.search(r'<meta property="og:title" content="(.*?)"', html)
                name = title_match.group(1) if title_match else f"AliExpress Product {pid}"
                shop_match = re.search(r'"storeName":"(.*?)"', html)
                shop_name = shop_match.group(1) if shop_match else "AliExpress Seller"
                images = []
                img_list_match = re.search(r'"imagePathList":\s*(\[.*?\])', html)
                if img_list_match:
                    try: images = json.loads(img_list_match.group(1))
                    except: pass
                if not images:
                    img_match = re.search(r'<meta property="og:image" content="(.*?)"', html)
                    if img_match: images.append(img_match.group(1))
                return {
                    "id": str(pid), "name": name, "shop_name": shop_name,
                    "product_link": str(product_url), "product_images": images
                }
            # Unlock to testing and debug
            # except: return {"id": str(pid), "name": f"AliExpress {pid}", "shop_name": "Unknown", "product_link": product_url, "product_images": []}
            except: return None

        return None

    # FOR TIKI CRAWLING
    def _crawl_tiki_page(self, pid, page):
        url = f"https://tiki.vn/api/v2/reviews?product_id={product_id}&limit=20&page={page}"
        try:
            res = requests.get(url, headers=self.tiki_headers, timeout=10)
            if res.status_code == 200: return res.json().get("data", [])
        except: pass
        return []

    # FOR ALIPAY CRAWLING
    def _generate_deterministic_id(product_id, content, date_str):
        raw_str = f"{product_id}_{content[:50]}_{date_str}"
        return hashlib.md5(raw_str.encode()).hexdigest()

    def _norm_url(u: str):
        if not u: return ""
        u = u.replace("\\u002F", "/")
        if u.startswith("//"): u = "https:" + u
        return u

    def _fetch_ali_page(self, pid, page):
        url = "https://feedback.aliexpress.com/pc/searchEvaluation.do"
        params = {"productId": pid, "lang": "en_US", "country": "US", "page": page, "pageSize": 20, "filter": "all", "sort": "complex_default"}
        try: return requests.get(url, params=params, headers=self.ali_headers, timeout=20).json()
        except: return {}

    
    # MAIN CRAWL REVIEWS
    def _crawl_reviews(self, pid: str, platform: str):
        if platform == "tiki":
            try:
                url = f"https://tiki.vn/api/v2/reviews?product_id={pid}&limit=20&page=1"
                data = requests.get(url, headers=self.tiki_headers).json()
                total = data.get("paging", {}).get("total", 0)
                total_pages = max((total // 20) + 1, 1)
            except: total_pages = 1; total = 0

            all_reviews = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._crawl_tiki_page, pid, p) for p in range(1, total_pages + 1)]
                for f in as_completed(futures):
                    res = f.result()
                    if res: all_reviews.extend(res)

            normalized = []
            for r in all_reviews:
                ts = r.get("created_at")
                c_at = datetime.fromtimestamp(int(ts), timezone.utc).isoformat() if ts else datetime.now(timezone.utc).isoformat()
                normalized.append({
                    "id": str(r.get("id")), "product_id": str(pid), "rating": r.get("rating", 0),
                    "content": r.get("content", ""), "created_at": c_at,
                    "images": [img.get("full_path", "") for img in r.get("images", [])]
                })
            return normalized, total

        if platform == "aliexpress":
            all_reviews = []
            seen_ids = set()
            first_page = self._fetch_ali_page(pid, 1)
            if not first_page: return [], 0

            data_node = first_page.get("data", {})
            total_count = data_node.get("totalNum", 0)
            total_pages = int(data_node.get("totalPage") or 1)
            actual_pages = min(total_pages, 15)

            def process(payload):
                items = []
                raw = payload.get("data", {}).get("evaViewList") or []
                for r in raw:
                    content = (r.get("buyerFeedback") or r.get("feedback") or "").strip()
                    date_str = r.get("evalDate") or r.get("date")
                    r_id = str(r.get("feedbackId")) if r.get("feedbackId") else self._generate_deterministic_id(pid, content, date_str)

                    if r_id in seen_ids: continue
                    seen_ids.add(r_id)

                    try: created_at = parser.parse(date_str).isoformat() if date_str else datetime.now(timezone.utc).isoformat()
                    except: created_at = datetime.now(timezone.utc).isoformat()

                    try: rating = int(float(r.get("buyerEval", 5)) / 20)
                    except: rating = 5

                    imgs = []
                    if "images" in r: imgs = r["images"] # simple case
                    else: # complex ali structure
                        if "picList" in r: imgs = r["picList"]

                    # Extract URLs safely
                    clean_imgs = []
                    for im in imgs:
                        if isinstance(im, str): clean_imgs.append(self._norm_url(im))
                        elif isinstance(im, dict) and "url" in im: clean_imgs.append(self._norm_url(im["url"]))

                    items.append({
                        "id": r_id, "product_id": str(pid), "rating": max(1, min(5, rating)),
                        "content": content, "created_at": created_at, "images": list(set(clean_imgs))
                    })
                return items

            all_reviews.extend(process(first_page))
            for p in range(2, actual_pages + 1):
                res = self._fetch_ali_page(pid, p)
                all_reviews.extend(process(res))
                time.sleep(random.uniform(0.5, 1.0))

            return all_reviews, total_count

    # Supabase Checking Duplicate Logic
    def _check_product_status(product_id):
        try:
            res = db.client.table("products").select("id").eq("id", str(product_id)).execute()
            if not res.data: return False, 0
            count = db.client.table("raw_comment").select("id", count="exact", head=True).eq("product_id", str(product_id)).execute()
            return True, count.count
        except: return False, 0

    def _save_product_info(product_data):
        if not product_data: return False, "No Data"
        try:
            product_data["id"] = str(product_data["id"])
            data = {
                "id": product_data["id"], "name": product_data["name"],
                "shop_name": product_data["shop_name"], "product_link": product_data["product_link"],
                "product_images": product_data["product_images"], "created_at": datetime.now(timezone.utc).isoformat()
            }
            db.client.table("products").upsert(data).execute()
            return True, "Success"
        except Exception as e: return False, str(e)

    def crawl(self, product_url: str):
        platform = self._get_platform(product_url)
        pid = self._extract_id(product_url, platform)
        
        if not pid:
            raise ValueError("Invalid URL or Platform")

        exists, db_count = self._check_product_status(pid)

        live_total = 0
        if platform == "tiki":
            try:
                r = requests.get(f"https://tiki.vn/api/v2/reviews?product_id={pid}&limit=1&page=1", headers=self.tiki_headers).json()
                live_total = r.get("paging", {}).get("total", 0)
            except: pass
        else:
            try:
                r = self._fetch_ali_page(pid, 1)
                live_total = r.get("data", {}).get("totalNum", 0)
            except: pass

        # Cache logic: Nếu DB có data và chênh lệch không quá lớn (hoặc Ali ảo)
        if exists and db_count > 0:
            if db_count >= live_total or (live_total - db_count < 5):
                print(f"✅ Cache Hit: PID {pid}")
                return {"status": "already_processed", "product_id": pid, "total_reviews": db_count}


        info = self._get_product_info(pid, product_url, platform)
        revs, _ = self._crawl_reviews(pid, platform)

        if not info: raise ValueError("Product Info Not Found")

        imgs_saved, _ = db.save_raw_reviews_and_images(pid, revs)

        return {"status": "success", "product_id": pid, "total_reviews_crawled": len(revs), "total_images_saved": imgs_saved}

        # 1. Crawl Product Info & Reviews (Logic from notebook)
        # Giả lập logic cào
        product_info = {"id": pid, "name": "Demo Product", "product_link": product_url} # Placeholder
        reviews = [] # Placeholder: Thực hiện requests tới Tiki/Ali API thật ở đây
        
        # 2. Save to DB
        db.upsert_product(product_info)
        # db.upsert_reviews(reviews) -> Cần transform data đúng schema trước khi save
        
        return {"status": "success", "product_id": pid, "reviews_count": len(reviews)}