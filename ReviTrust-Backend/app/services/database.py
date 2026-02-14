from supabase import create_client, Client
from app.config import settings
from datetime import datetime, timezone


class DatabaseService:
    def __init__(self):
        self.client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    # For Crawler
    def save_raw_reviews_and_images(self, product_id, reviews):
        if not reviews: return 0, []
        errors = []
        saved_count = 0

        comments_batch = []
        images_to_insert = []

        # Chuẩn bị comments
        for r in reviews:
            comments_batch.append({
                "id": str(r["id"]), "product_id": str(product_id),
                "rating": r["rating"], "content": r["content"],
                "created_at": r["created_at"], "processed": False
            })
            for img_url in r["images"]:
                images_to_insert.append({"comment_id": str(r["id"]), "image_url": img_url})

        # Upsert comments
        for i in range(0, len(comments_batch), 50):
            try: self.client.table("raw_comment").upsert(comments_batch[i:i+50], on_conflict="id").execute()
            except Exception as e: errors.append(str(e))

        # Insert images (Strict check)
        if images_to_insert:
            try:
                # 1. Lấy tất cả ảnh đã có trong DB của các comment này
                cids = list(set([c["id"] for c in comments_batch]))
                existing_set = set()
                for i in range(0, len(cids), 500):
                    chunk = cids[i:i+500]
                    res = self.client.table("comment_images").select("comment_id, image_url").in_("comment_id", chunk).execute()
                    for item in res.data: existing_set.add((str(item["comment_id"]), item["image_url"]))

                # 2. Lọc ảnh mới
                final_imgs = []
                seen_local = set()
                for item in images_to_insert:
                    key = (item["comment_id"], item["image_url"])
                    if key not in existing_set and key not in seen_local:
                        final_imgs.append(item)
                        seen_local.add(key)

                # 3. Insert
                for i in range(0, len(final_imgs), 50):
                    try:
                        self.client.table("comment_images").insert(final_imgs[i:i+50]).execute()
                        saved_count += len(final_imgs[i:i+50])
                    except Exception as e: errors.append(str(e))
            except Exception as e: errors.append(str(e))

        return saved_count, errors

    # Supabase Checking Duplicate Logic
    def check_product_status(self, product_id):
        try:
            res = self.client.table("products").select("id").eq("id", str(product_id)).execute()
            if not res.data: return False, 0
            count = self.client.table("raw_comment").select("id", count="exact", head=True).eq("product_id", str(product_id)).execute()
            return True, count.count
        except: return False, 0

    def save_product_info(self, product_data):
        if not product_data: return False, "No Data"
        try:
            product_data["id"] = str(product_data["id"])
            data = {
                "id": product_data["id"], "name": product_data["name"],
                "shop_name": product_data["shop_name"], "product_link": product_data["product_link"],
                "product_images": product_data["product_images"], "created_at": datetime.now(timezone.utc).isoformat()
            }
            self.client.table("products").upsert(data).execute()
            return True, "Success"
        except Exception as e: return False, str(e)

    # For Analytics
    # --- HELPER: FETCH ALL DATA ---
    def fetch_all_rows_pagination(self, table_name, col_name, col_value, select="*"):
        all_data = []
        offset = 0
        limit = 1000
        while True:
            try:
                res = self.client.table(table_name).select(select).eq(col_name, str(col_value)).range(offset, offset + limit - 1).execute()
                batch_data = res.data
                if not batch_data: break
                all_data.extend(batch_data)
                if len(batch_data) < limit: break
                offset += limit
            except Exception as e:
                print(f"   ⚠️ Error fetching {table_name}: {e}")
                break
        return all_data

    

    def get_product_name(self, product_id: str):
        try:
            res = self.client.table("products").select("name").eq("id", str(product_id)).execute()
            if res.data: return res.data[0]['name']
        except: pass
        return "Unknown Product"

    def fetch_all_in_list(self, table_name, col_in, list_values, select="*"):
        all_data = []
        CHUNK = 500
        list_values = [str(v) for v in list_values]
        # Deduplicate list_values trước khi query để tối ưu
        list_values = list(set(list_values))

        for i in range(0, len(list_values), CHUNK):
            chunk = list_values[i:i+CHUNK]
            try:
                res = self.client.table(table_name).select(select).in_(col_in, chunk).execute()
                if res.data: all_data.extend(res.data)
            except Exception as e:
                print(f"Error fetch_all_in_list: {e}")
        return all_data

    #-----------------------------------------------------------------
    # def upsert_product(self, data: dict):
    #     return self.client.table("products").upsert(data).execute()

    # def upsert_reviews(self, reviews: list):
    #     # Chia batch để insert nếu danh sách quá dài
    #     BATCH_SIZE = 50
    #     results = []
    #     for i in range(0, len(reviews), BATCH_SIZE):
    #         batch = reviews[i:i+BATCH_SIZE]
    #         results.append(self.client.table("raw_comment").upsert(batch, on_conflict="id").execute())
    #     return results

    # def insert_review_images(self, images: list):
    #     # Logic check trùng lặp nên thực hiện ở crawler hoặc dùng upsert
    #     if not images: return
    #     BATCH_SIZE = 50
    #     for i in range(0, len(images), BATCH_SIZE):
    #         try:
    #             self.client.table("comment_images").insert(images[i:i+BATCH_SIZE]).execute()
    #         except Exception as e:
    #             print(f"Image insert error (duplicate/other): {e}")

    def get_product_reviews(self, product_id: str):
        # Fetch data phục vụ analytics
        # Lưu ý: Cần implement pagination nếu data lớn (như trong notebook cũ)
        return self.client.table("raw_comment").select("*").eq("product_id", str(product_id)).execute()

    def get_ai_results(self, product_id: str, ai_type="text"):
        # Helper lấy kết quả AI để tính toán
        pass # Implement logic query bảng text_ai_results / image_ai_results

    def save_metrics(self, metrics: dict):
        return self.client.table("product_trust_metrics").upsert(metrics).execute()


db = DatabaseService()