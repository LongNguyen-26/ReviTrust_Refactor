from supabase import create_client, Client
from app.config import settings

class DatabaseService:
    def __init__(self):
        self.client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    def upsert_product(self, data: dict):
        return self.client.table("products").upsert(data).execute()

    def upsert_reviews(self, reviews: list):
        # Chia batch để insert nếu danh sách quá dài
        BATCH_SIZE = 50
        results = []
        for i in range(0, len(reviews), BATCH_SIZE):
            batch = reviews[i:i+BATCH_SIZE]
            results.append(self.client.table("raw_comment").upsert(batch, on_conflict="id").execute())
        return results

    def insert_review_images(self, images: list):
        # Logic check trùng lặp nên thực hiện ở crawler hoặc dùng upsert
        if not images: return
        BATCH_SIZE = 50
        for i in range(0, len(images), BATCH_SIZE):
            try:
                self.client.table("comment_images").insert(images[i:i+BATCH_SIZE]).execute()
            except Exception as e:
                print(f"Image insert error (duplicate/other): {e}")

    def get_product_reviews(self, product_id: str):
        # Fetch data phục vụ analytics
        # Lưu ý: Cần implement pagination nếu data lớn (như trong notebook cũ)
        return self.client.table("raw_comment").select("*").eq("product_id", str(product_id)).execute()

    def get_ai_results(self, product_id: str, ai_type="text"):
        # Helper lấy kết quả AI để tính toán
        pass # Implement logic query bảng text_ai_results / image_ai_results

    def save_metrics(self, metrics: dict):
        return self.client.table("product_trust_metrics").upsert(metrics).execute()

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

db = DatabaseService()