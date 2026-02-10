import os
import shutil
import requests
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from supabase import create_client, Client

# --- CONFIG ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TEMP_DIR = "temp_ai_processing"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="ReviTrust Image Core")

# --- SUPABASE CLIENT ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- MODEL LOADING ---
print(f"🚀 Loading Models on {DEVICE}...")

# 1. CLIP Model
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"❌ Error loading CLIP: {e}")

# 2. MobileNetV2 (Custom Trained)
CNN_CLASS_NAMES = ['POTENTIAL', 'TRASH']
cnn_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_mobilenet_model(path, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    try:
        # Load weight từ file local (được upload lên HF Space)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"❌ Error loading MobileNet: {e}")
        return None

# Đảm bảo bạn upload file best.pt lên root của HF Space
cnn_model = load_mobilenet_model("best.pt", DEVICE)

# --- UTILS ---
# [NEW] Hàm này giúp lấy toàn bộ dữ liệu (vượt giới hạn 1000 dòng của Supabase)
def fetch_all_rows(table_name, col_name, col_value, select="*"):
    all_data = []
    offset = 0
    limit = 1000 # Supabase thường giới hạn 1000 row/request
    
    print(f"📡 Fetching {table_name} where {col_name}={col_value}...")
    
    while True:
        try:
            # Dùng .range() để phân trang
            res = supabase.table(table_name)\
                .select(select)\
                .eq(col_name, str(col_value))\
                .range(offset, offset + limit - 1)\
                .execute()
            
            data = res.data
            if not data: 
                break # Hết dữ liệu
            
            all_data.extend(data)
            
            if len(data) < limit: 
                break # Đã lấy đến trang cuối
                
            offset += limit # Tăng offset để lấy trang tiếp theo
            
        except Exception as e:
            print(f"⚠️ Error fetching {table_name} (offset {offset}): {e}")
            break
            
    return all_data

def download_image(url, path):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            with open(path, "wb") as f:
                f.write(res.content)
            return True
        return False
    except: return False

def get_clip_emb(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
            return clip_model.get_image_features(**inputs).cpu().numpy()[0]
    except: return None

def classify_cnn(img_path):
    if not cnn_model: return "POTENTIAL", 0.0
    try:
        image = Image.open(img_path).convert("RGB")
        t = cnn_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = cnn_model(t)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, 0)
            return CNN_CLASS_NAMES[idx.item()], float(conf.item())
    except: return "POTENTIAL", 0.0

# --- CORE LOGIC ---
# --- CORE LOGIC ---
def process_product_images(product_id: str):
    pid_str = str(product_id)
    work_dir = os.path.join(TEMP_DIR, pid_str)
    os.makedirs(work_dir, exist_ok=True)

    print(f"🚀 START Processing Product: {pid_str}")

    # ==============================================================================
    # 1. Get Shop Images (Ảnh mẫu từ shop) - Giữ nguyên Code 1 vì thường ít
    # ==============================================================================
    try:
        prod_res = supabase.table("products").select("product_images").eq("id", pid_str).execute()
        shop_urls = prod_res.data[0].get('product_images', []) if prod_res.data else []
    except: shop_urls = []

    shop_embs = []
    if shop_urls:
        for i, url in enumerate(shop_urls):
            p = os.path.join(work_dir, f"shop_{i}.jpg")
            if download_image(url, p):
                emb = get_clip_emb(p)
                if emb is not None: shop_embs.append(emb)
    shop_matrix = np.vstack(shop_embs) if shop_embs else None

    # ==============================================================================
    # 2. Get Review Images (Ảnh đánh giá) - [THAY ĐỔI QUAN TRỌNG]
    # ==============================================================================
    try:
        # [NEW] Dùng fetch_all_rows thay vì query thường để đảm bảo lấy hết comment ID
        # Code cũ: c_res = supabase.table("raw_comment")... (Chỉ lấy đc 1000 row đầu)
        raw_comments = fetch_all_rows("raw_comment", "product_id", pid_str, select="id")
        
        c_ids = [str(c['id']) for c in raw_comments]
        if not c_ids: return {"status": "No comments found"}
        
        print(f"🔎 Found {len(c_ids)} comments. Fetching images...")

        # Batch fetch images (Lấy ảnh theo từng cụm 500 ID để không làm quá tải DB)
        db_images = []
        for i in range(0, len(c_ids), 500):
            chunk = c_ids[i:i+500]
            # Lưu ý: Ở đây không dùng fetch_all_rows được vì ta dùng toán tử .in_()
            img_res = supabase.table("comment_images").select("*").in_("comment_id", chunk).execute()
            if img_res.data:
                db_images.extend(img_res.data)
                
        print(f"📸 Total images to process: {len(db_images)}")
        
    except Exception as e:
        return {"error": str(e)}

    if not db_images: return {"status": "No images found in comments"}

    # ==============================================================================
    # 3. Process AI (Xử lý song song)
    # ==============================================================================
    processed_results = []
    valid_items = []

    def process_one(img_row):
        fname = f"img_{img_row['id']}.jpg"
        fpath = os.path.join(work_dir, fname)
        if not download_image(img_row['image_url'], fpath): return None
        
        cnn_st, cnn_conf = classify_cnn(fpath)
        item = {
            "image_id": img_row['id'],
            "cnn_status": cnn_st,
            "final_classification": cnn_st,
            "similarity_score": cnn_conf,
            "clip_embedding": None
        }
        
        # Chỉ chạy CLIP nếu ảnh không phải là rác (TRASH)
        if cnn_st != "TRASH":
            emb = get_clip_emb(fpath)
            if emb is not None:
                item["clip_embedding"] = emb
                item["final_classification"] = "VALID"
        return item

    # [UPDATE] Tăng max_workers lên 10 để nhanh hơn (Code 1 là 5, Code 2 là 25)
    with ThreadPoolExecutor(max_workers=10) as ex: 
        futures = [ex.submit(process_one, row) for row in db_images]
        for f in as_completed(futures):
            res = f.result()
            if res:
                processed_results.append(res)
                if res.get("clip_embedding") is not None: valid_items.append(res)

    # ==============================================================================
    # 4. Compare Logic (So sánh trùng lặp) - Giữ nguyên Code 1
    # ==============================================================================
    if valid_items:
        rv_mtx = np.vstack([x['clip_embedding'] for x in valid_items])
        
        # Internal Duplicates (Ảnh trùng nhau giữa các review)
        sim = cosine_similarity(rv_mtx)
        for i in range(len(valid_items)):
            for j in range(i+1, len(valid_items)):
                if sim[i, j] > 0.98:
                    valid_items[j]['final_classification'] = "DUPLICATE_SPAM"

        # Shop Duplicates (Ảnh review lấy trộm từ shop)
        if shop_matrix is not None:
            scores = np.max(cosine_similarity(rv_mtx, shop_matrix), axis=1)
            for i, obj in enumerate(valid_items):
                if obj['final_classification'] == "DUPLICATE_SPAM": continue
                sc = float(scores[i])
                obj['similarity_score'] = sc
                if sc > 0.95: obj['final_classification'] = "SHOP_IMAGE_SPAM"
                elif sc > 0.65: obj['final_classification'] = "VALID"
                elif sc > 0.45: obj['final_classification'] = "SUSPICIOUS"
                else: obj['final_classification'] = "MISMATCH"

    # ==============================================================================
    # 5. Save DB (Lưu kết quả) - Giữ nguyên Code 1 (QUAN TRỌNG: Dùng UPSERT)
    # ==============================================================================
    db_payload = []
    for p in processed_results:
        emb_val = p['clip_embedding'].tolist() if p['clip_embedding'] is not None else None
        db_payload.append({
            "image_id": int(p['image_id']),
            "cnn_status": p['cnn_status'],
            "final_classification": p['final_classification'],
            "similarity_score": p['similarity_score'],
            "clip_embedding": emb_val,
            "processed_at": datetime.now(timezone.utc).isoformat()
        })

    # Upsert batch 200 items (An toàn hơn insert của Code 2)
    saved_count = 0
    for i in range(0, len(db_payload), 200):
        try:
            supabase.table("image_ai_results").upsert(db_payload[i:i+200]).execute()
            saved_count += len(db_payload[i:i+200])
        except Exception as e: print(f"DB Error batch {i}: {e}")

    shutil.rmtree(work_dir, ignore_errors=True)
    return {"status": "success", "processed": len(db_payload), "saved": saved_count}

# --- API ---
class AIRequest(BaseModel):
    product_id: str

@app.post("/analyze")
async def run_analysis(req: AIRequest):
    return process_product_images(req.product_id)