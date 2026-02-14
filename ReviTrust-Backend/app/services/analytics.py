import pandas as pd
import numpy as np
from app.services.database import db
from datetime import datetime, timezone

class AnalyticsService:
    # def __init__(self, db_service) -> None:
    #     self.db = db_service
    # --- DATA MAPPING HELPERS (ĐÃ SỬA LỖI) ---

    def map_reviews_detail(self, comment_ids, df_merged, df_images, limit=10, filter_bad_images=False):
        """
        Map chi tiết review.
        - filter_bad_images=True: Sẽ loại bỏ ảnh TRASH/SPAM khỏi danh sách ảnh hiển thị của review đó.
        """
        if not comment_ids or df_merged.empty: return []
        results = []
        comment_ids = [str(x) for x in comment_ids]

        # Filter df_merged
        df_subset = df_merged[df_merged['comment_id'].isin(comment_ids)]
        records = df_subset.to_dict('records')
        record_map = {str(r['comment_id']): r for r in records}

        # Danh sách nhãn xấu cần lọc bỏ nếu filter_bad_images=True
        BAD_LABELS = ['TRASH', 'SHOP_IMAGE_SPAM', 'DUPLICATE_SPAM']

        for cid in comment_ids:
            row = record_map.get(cid)
            if not row: continue

            img_urls = []
            if not df_images.empty:
                # Lấy ảnh của comment này
                imgs_df = df_images[df_images['comment_id'] == cid]

                if not imgs_df.empty:
                    if filter_bad_images and 'final_classification' in imgs_df.columns:
                        # 🛠️ FIX 2: Lọc bỏ ảnh rác khỏi review tốt
                        imgs_df = imgs_df[~imgs_df['final_classification'].isin(BAD_LABELS)]

                    if not imgs_df.empty and 'image_url' in imgs_df.columns:
                        # 🛠️ FIX 1: Dùng set() để loại bỏ URL trùng lặp
                        img_urls = list(set(imgs_df['image_url'].tolist()))

            results.append({
                "comment_id": cid,
                "rating": int(row.get("rating", 0)),
                "content": row.get("content", ""),
                "images": img_urls,
                "created_at": row.get("created_at", ""),
            })
            if len(results) >= limit: break
        return results

    def count_aspect_reviews_detailed(self, df_merged, df_images):
        aspect_stats = {}
        aspects = {"packaging": "sentiment_packaging", "quality": "sentiment_quality", "price": "sentiment_price", "delivery": "sentiment_delivery"}

        # Xác định review nào có ảnh HỢP LỆ (không phải spam/rác)
        valid_img_comments = []
        if not df_images.empty:
            # Chỉ lấy ảnh tốt
            good_imgs = df_images[~df_images['final_classification'].isin(['TRASH', 'SHOP_IMAGE_SPAM', 'DUPLICATE_SPAM'])]
            valid_img_comments = good_imgs['comment_id'].unique().tolist()

        df_merged['has_valid_images'] = df_merged['comment_id'].isin(valid_img_comments)

        def get_top_comments(df, col_name, sentiment_value, limit=5):
            filtered = df[df[col_name] == sentiment_value]
            # Ưu tiên có ảnh HỢP LỆ + nội dung dài
            priority = filtered.sort_values(by=['has_valid_images', 'content'], ascending=[False, True])
            return priority['comment_id'].head(limit).tolist()

        for aspect_name, col_name in aspects.items():
            if col_name in df_merged.columns:
                aspect_reviews = df_merged[df_merged[col_name] != 0]
                total_count = len(aspect_reviews)
                pos_reviews = aspect_reviews[aspect_reviews[col_name] == 1]
                neg_reviews = aspect_reviews[aspect_reviews[col_name] == -1]

                # Khi map review cho aspect, bật filter_bad_images=True để không hiện ảnh rác
                pos_details = self.map_reviews_detail(get_top_comments(aspect_reviews, col_name, 1, 5), df_merged, df_images, 5, filter_bad_images=True)
                neg_details = self.map_reviews_detail(get_top_comments(aspect_reviews, col_name, -1, 5), df_merged, df_images, 5, filter_bad_images=True)

                aspect_stats[aspect_name] = {
                    "total": int(total_count),
                    "positive": {"count": int(len(pos_reviews)), "ratio": float(len(pos_reviews)/total_count) if total_count else 0, "reviews": pos_details},
                    "negative": {"count": int(len(neg_reviews)), "ratio": float(len(neg_reviews)/total_count) if total_count else 0, "reviews": neg_details}
                }
            else:
                aspect_stats[aspect_name] = {"total": 0, "positive": {"count":0,"ratio":0,"reviews":[]}, "negative": {"count":0,"ratio":0,"reviews":[]}}
        return aspect_stats

    def calculate_trust_score(self, metrics: dict, aspect_stats: dict) -> tuple:
        R_avg = metrics['avg_rating']
        R_norm = (R_avg - 1.0) / 4.0 if R_avg >= 1.0 else 0.0

        S_ratio = metrics['spam_review_ratio']
        F_ratio = metrics['fake_image_ratio']

        aspect_positive_ratios = []
        for aspect_name in ["quality", "price", "delivery", "packaging"]:
            if aspect_stats[aspect_name]["total"] > 0:
                aspect_positive_ratios.append(aspect_stats[aspect_name]["positive"]["ratio"])

        avg_aspect_pos = np.mean(aspect_positive_ratios) if aspect_positive_ratios else 0.0
        overall_pos = metrics.get('positive_ratio', 0.0)
        overall_neg = metrics.get('negative_ratio', 0.0)

        score = 0.30
        score += R_norm * 0.25
        score += avg_aspect_pos * 0.35
        score += overall_pos * 0.10

        score -= S_ratio * 0.20
        score -= F_ratio * 0.20
        score -= overall_neg * 0.10

        final_score = max(0.20, min(1.0, score))

        if final_score > 0.75: label = "SAFE"
        elif final_score > 0.50: label = "LOW_RISK"
        elif final_score > 0.30: label = "WARNING"
        else: label = "SCAM"

        return final_score, label

    def classify_review_sentiment(self, rating: int, sentiment_overall: float) -> str:
        if rating >= 4 and sentiment_overall >= 0.3: return "positive"
        elif rating <= 2 or sentiment_overall <= -0.3: return "negative"
        else: return "neutral"

    def fetch_data_as_dataframe(self, product_id: str):
        raw_comments_data = db.fetch_all_rows_pagination("raw_comment", "product_id", str(product_id), select="id, rating, content, created_at")
        df_comments = pd.DataFrame(raw_comments_data)
        if df_comments.empty: return None, None, None

        df_comments['id'] = df_comments['id'].astype(str)
        # Deduplicate comments nếu có
        df_comments.drop_duplicates(subset=['id'], inplace=True)
        comment_ids = df_comments['id'].tolist()

        text_ai_data = db.fetch_all_in_list("text_ai_results", "comment_id", comment_ids, select="comment_id, is_spam, spam_score, sentiment_overall, sentiment_delivery, sentiment_packaging, sentiment_quality, sentiment_price")
        df_text = pd.DataFrame(text_ai_data)
        if not df_text.empty:
            df_text['comment_id'] = df_text['comment_id'].astype(str)
            df_text.drop_duplicates(subset=['comment_id'], inplace=True)

        img_map_data = db.fetch_all_in_list("comment_images", "comment_id", comment_ids, select="id, comment_id, image_url")
        df_img_map = pd.DataFrame(img_map_data)

        df_img_ai = pd.DataFrame()
        if not df_img_map.empty:
            image_ids = df_img_map['id'].tolist()
            img_ai_data = db.fetch_all_in_list("image_ai_results", "image_id", image_ids, select="image_id, final_classification, similarity_score")
            df_img_ai = pd.DataFrame(img_ai_data)
            # Deduplicate AI results
            df_img_ai.drop_duplicates(subset=['image_id'], inplace=True)

        df_images_full = pd.DataFrame(columns=["comment_id", "final_classification", "similarity_score", "image_url"])
        if not df_img_ai.empty and not df_img_map.empty:
            df_images_full = pd.merge(df_img_map, df_img_ai, left_on="id", right_on="image_id", how="left")
            # Fill NA cho những ảnh chưa chạy AI xong để tránh lỗi
            df_images_full['final_classification'] = df_images_full['final_classification'].fillna('UNKNOWN')

        if not df_images_full.empty:
            df_images_full['comment_id'] = df_images_full['comment_id'].astype(str)
            # 🛠️ FIX DUPLICATE ROWS IN MERGE:
            df_images_full.drop_duplicates(subset=['comment_id', 'image_url'], inplace=True)

        return df_comments, df_text, df_images_full

    # MAIN PROCESS

    def process_analytics(self, product_id: str):
        pid_str = str(product_id)
        print(f"📄 Calculating for Product {pid_str}...")
        product_name = db.get_product_name(pid_str)
        df_comments, df_text, df_images = self.fetch_data_as_dataframe(pid_str)

        if df_comments is None: return {"status": "error", "msg": "No data found."}

        total_reviews = len(df_comments)
        df_comments.rename(columns={'id': 'comment_id'}, inplace=True)
        df_comments['comment_id'] = df_comments['comment_id'].astype(str)

        if not df_text.empty:
            df_merged = pd.merge(df_comments, df_text, on="comment_id", how="left")
        else:
            df_merged = df_comments.copy()
            for col in ['sentiment_overall', 'sentiment_delivery', 'sentiment_packaging', 'sentiment_quality', 'sentiment_price', 'spam_score']: df_merged[col] = 0.0
            df_merged['is_spam'] = False

        df_merged.fillna(0, inplace=True)
        df_merged['is_spam'] = df_merged['is_spam'].astype(bool)
        df_merged['content'] = df_merged['content'].fillna("")

        # --- XỬ LÝ PHÂN LOẠI REVIEW (VALID) ---
        df_valid = df_merged[~df_merged['is_spam']].copy()

        # Chỉ đánh dấu "has_images" nếu review đó có ảnh HỢP LỆ (Không phải rác)
        valid_comments_with_images = []
        if not df_images.empty:
            # Lọc ảnh tốt
            good_imgs = df_images[~df_images['final_classification'].isin(['TRASH', 'SHOP_IMAGE_SPAM', 'DUPLICATE_SPAM'])]
            valid_comments_with_images = good_imgs['comment_id'].unique().tolist()

        df_valid['has_valid_images'] = df_valid['comment_id'].isin(valid_comments_with_images)

        # Biến thống kê cho tất cả ảnh (bao gồm cả ảnh rác để tính fake_ratio)
        all_comments_with_images = df_images['comment_id'].unique().tolist() if not df_images.empty else []

        highlight_list = []
        pos_ratio = neg_ratio = neu_ratio = pos_img_ratio = 0.0
        pos_count = neg_count = neu_count = 0
        pos_reviews = neg_reviews = neu_reviews = []

        if not df_valid.empty:
            df_valid['sentiment_class'] = df_valid.apply(lambda row: self.classify_review_sentiment(row['rating'], row['sentiment_overall']), axis=1)

            pos_df = df_valid[df_valid['sentiment_class'] == 'positive']
            neg_df = df_valid[df_valid['sentiment_class'] == 'negative']
            neu_df = df_valid[df_valid['sentiment_class'] == 'neutral']

            total_valid = len(df_valid)
            pos_count = len(pos_df); neg_count = len(neg_df); neu_count = len(neu_df)
            pos_ratio = pos_count/total_valid if total_valid else 0
            neg_ratio = neg_count/total_valid if total_valid else 0
            neu_ratio = neu_count/total_valid if total_valid else 0

            # Tỉ lệ review tốt có ảnh (Dùng ảnh hợp lệ)
            pos_img_count = len(pos_df[pos_df['has_valid_images'] == True])
            pos_img_ratio = pos_img_count / total_reviews if total_reviews > 0 else 0.0

            # Highlights Logic: Positive + Có Text + Có Ảnh Hợp Lệ
            mask_potential = (df_valid['sentiment_class'] == 'positive') & (df_valid['content'].str.strip() != '')
            df_highlights = df_valid[mask_potential].copy()

            if not df_highlights.empty:
                if not df_images.empty:
                    # Lấy max score của ảnh HỢP LỆ
                    good_imgs = df_images[~df_images['final_classification'].isin(['TRASH', 'SHOP_IMAGE_SPAM', 'DUPLICATE_SPAM'])]
                    if not good_imgs.empty:
                        img_scores = good_imgs.groupby('comment_id')['similarity_score'].max().reset_index().rename(columns={'similarity_score': 'max_img_score'})
                        df_highlights = pd.merge(df_highlights, img_scores, on='comment_id', how='left')
                        df_highlights['max_img_score'] = df_highlights['max_img_score'].fillna(0)
                    else:
                        df_highlights['max_img_score'] = 0.0
                else:
                    df_highlights['max_img_score'] = 0.0

                # Sort: Có ảnh hợp lệ > Điểm cao
                df_highlights['ranking_score'] = df_highlights['sentiment_overall'] + df_highlights['max_img_score']
                df_highlights = df_highlights.sort_values(by=['has_valid_images', 'ranking_score'], ascending=[False, False])

                # Gọi map với filter_bad_images=True
                top_ids = df_highlights['comment_id'].head(10).tolist()
                highlight_list = self.map_reviews_detail(top_ids, df_merged, df_images, filter_bad_images=True)

            # Strict Lists (Bắt buộc Text + Ảnh Hợp Lệ)
            def get_strict_ids(df):
                mask = (df['has_valid_images'] == True) & (df['content'].str.strip() != '')
                sorted_df = df[mask].sort_values(by='content', key=lambda x: x.str.len(), ascending=False)
                return sorted_df['comment_id'].head(10).tolist()

            # Gọi map với filter_bad_images=True
            pos_reviews = self.map_reviews_detail(get_strict_ids(pos_df), df_merged, df_images, filter_bad_images=True)
            neg_reviews = self.map_reviews_detail(get_strict_ids(neg_df), df_merged, df_images, filter_bad_images=True)
            neu_reviews = self.map_reviews_detail(get_strict_ids(neu_df), df_merged, df_images, filter_bad_images=True)

        aspect_stats = self.count_aspect_reviews_detailed(df_merged[~df_merged['is_spam']], df_images)

        # --- SPAM & FAKE LISTS (Ở đây KHÔNG lọc, vì ta muốn show bằng chứng xấu) ---
        df_has_text = df_merged[df_merged['content'].str.strip() != '']
        df_spam = df_has_text[df_has_text['is_spam'] == True]
        spam_ratio = float(len(df_spam) / len(df_has_text)) if not df_has_text.empty else 0.0

        # Spam list: show bình thường (có thể chứa ảnh rác cũng đc)
        spam_details_list = self.map_reviews_detail(df_spam.sort_values(by='spam_score', ascending=False)['comment_id'].head(10).tolist(), df_merged, df_images, filter_bad_images=False) if not df_spam.empty else []

        total_images = len(df_images)
        # Fake Images: Lấy danh sách ảnh xấu
        df_fake_images = df_images[df_images['final_classification'].fillna('N/A').isin(['SHOP_IMAGE_SPAM', 'DUPLICATE_SPAM', 'TRASH'])]
        fake_ratio = float(len(df_fake_images) / total_images) if total_images > 0 else 0.0

        # De-duplicate Fake Images trước khi trả về list
        fake_details_list = []
        if not df_fake_images.empty:
            df_fake_sorted = df_fake_images.sort_values(by='similarity_score', ascending=False)
            # Drop duplicate url để danh sách Fake Image clean hơn
            df_fake_sorted.drop_duplicates(subset=['image_url'], inplace=True)
            fake_details_list = df_fake_sorted[['image_url', 'final_classification']].head(10).to_dict(orient='records')

        # True Rating
        df_text_valid = df_merged[(df_merged['content'].str.strip() != '') & (~df_merged['is_spam'])]
        avg_rating = float(df_text_valid['rating'].mean()) if not df_text_valid.empty else 0.0

        metrics_db = {
            "product_id": pid_str,
            "total_reviews": total_reviews,
            "avg_rating": avg_rating,
            "rating_distribution": df_comments['rating'].value_counts().to_dict(),
            "review_with_text_ratio": float(len(df_has_text) / total_reviews) if total_reviews > 0 else 0.0,
            "spam_review_ratio": spam_ratio,
            "fake_image_ratio": fake_ratio,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        metrics_extended = metrics_db.copy()
        metrics_extended.update({"positive_ratio": pos_ratio, "negative_ratio": neg_ratio, "neutral_ratio": neu_ratio})
        score, label = self.calculate_trust_score(metrics_extended, aspect_stats)
        metrics_db["machine_learning_trust_score"] = score
        metrics_db["risk_label"] = label

        try: db.client.table("product_trust_metrics").upsert(metrics_db).execute()
        except Exception as e: print(f"Error saving metrics: {e}")

        return {
            "status": "success",
            "product_name": product_name,
            "metrics": {
                "product_id": pid_str,
                "total_reviews": total_reviews,
                "avg_rating": avg_rating,
                "pos_img_ratio": pos_img_ratio,
                "rating_distribution": metrics_db["rating_distribution"],
                "spam_review_ratio": {"ratio": spam_ratio, "details": spam_details_list},
                "fake_image_ratio": {"ratio": fake_ratio, "details": fake_details_list},
                "machine_learning_trust_score": metrics_db["machine_learning_trust_score"],
                "risk_label": metrics_db["risk_label"],
                "updated_at": metrics_db["updated_at"]
            },
            "highlights": {"positive_with_images": highlight_list},
            "sentiment_breakdown": {
                "positive": {"count": pos_count, "ratio": pos_ratio, "reviews": pos_reviews},
                "negative": {"count": neg_count, "ratio": neg_ratio, "reviews": neg_reviews},
                "neutral": {"count": neu_count, "ratio": neu_ratio, "reviews": neu_reviews}
            },
            "aspect_statistics": aspect_stats
        }

if __name__ == "__main__":
    analytics = AnalyticsService()
    final_result = analytics.process_analytics("product_id")
    print(final_result)