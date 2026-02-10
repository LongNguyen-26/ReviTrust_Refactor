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

db = DatabaseService()