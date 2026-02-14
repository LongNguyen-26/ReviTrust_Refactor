import os
import shutil
import requests
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from supabase import create_client, Client

# --- CONFIG ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TEMP_DIR = "temp_ai_processing"

# Model Paths (Đảm bảo upload file đã quantize lên cùng thư mục)
MOBI_MODEL_PATH = "mobilenet_quant.onnx"
CLIP_MODEL_PATH = "clip_vision_quant.onnx"

app = FastAPI(title="ReviTrust Image Core (ONNX Quantized)")

# --- SUPABASE CLIENT ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- ONNX INFERENCE SESSIONS ---
print(f"🚀 Loading ONNX Models (CPU Quantized)...")

# 1. CLIP Setup
try:
    # Processor vẫn cần để normalize ảnh đúng chuẩn CLIP
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_session = ort.InferenceSession(CLIP_MODEL_PATH, providers=["CPUExecutionProvider"])
    clip_input_name = clip_session.get_inputs()[0].name
    print("✅ CLIP ONNX Loaded")
except Exception as e:
    print(f"❌ Error loading CLIP ONNX: {e}")
    clip_session = None

# 2. MobileNet Setup
try:
    cnn_session = ort.InferenceSession(MOBI_MODEL_PATH, providers=["CPUExecutionProvider"])
    cnn_input_name = cnn_session.get_inputs()[0].name
    print("✅ MobileNet ONNX Loaded")
except Exception as e:
    print(f"❌ Error loading MobileNet ONNX: {e}")
    cnn_session = None

CNN_CLASS_NAMES = ['POTENTIAL', 'TRASH']

# --- PREPROCESSING UTILS ---
def preprocess_mobilenet(image: Image.Image):
    """
    Chuẩn hóa ảnh cho MobileNet khớp với torchvision.transforms:
    Resize(256) (cạnh ngắn = 256, giữ tỷ lệ) -> CenterCrop(224) -> Normalize
    """
    image = image.convert("RGB")
    
    # 1. Resize: Cạnh nhỏ nhất = 256, giữ nguyên tỷ lệ
    w, h = image.size
    if w < h:
        new_w = 256
        new_h = int(256 * h / w)
    else:
        new_h = 256
        new_w = int(256 * w / h)
        
    image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # 2. Center Crop 224
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = (new_w + 224) / 2
    bottom = (new_h + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    # 3. To Numpy & Normalize
    img_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # (H, W, C) -> Normalize -> (1, C, H, W)
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np

def get_clip_emb_onnx(img_path):
    if clip_session is None: return None
    try:
        image = Image.open(img_path).convert("RGB")
        
        # SỬA ĐỔI: Dùng return_tensors="pt" như file test_inference_v2.py 
        # để tránh lỗi dimension khi dùng "np" trực tiếp với transformers bản mới
        inputs = clip_processor(images=image, return_tensors="pt")
        
        # Convert từ Tensor -> Numpy
        pixel_values = inputs['pixel_values'].detach().cpu().numpy().astype(np.float32)
        
        # Run ONNX Inference
        outputs = clip_session.run(None, {clip_input_name: pixel_values})
        
        # SỬA ĐỔI: Theo claude_quantize.py, output gồm [last_hidden_state, pooler_output]
        # Ta cần pooler_output (embedding vector) nằm ở index 1
        if len(outputs) > 1:
            embedding = outputs[1][0] # Lấy batch đầu tiên của pooler_output
        else:
            # Fallback nếu model chỉ có 1 output (ít xảy ra với code export kia)
            embedding = outputs[0][0] 
            
        return embedding
    except Exception as e:
        print(f"CLIP Error: {e}")
        return None

def classify_cnn_onnx(img_path):
    if cnn_session is None: return "POTENTIAL", 0.0
    try:
        image = Image.open(img_path)
        # Hàm preprocess_mobilenet mới đã handle việc resize đúng tỷ lệ
        input_tensor = preprocess_mobilenet(image)
        
        # Run ONNX Inference
        # MobileNet export thường chỉ có 1 output là logits
        logits = cnn_session.run(None, {cnn_input_name: input_tensor})[0][0]
        
        # Softmax thủ công
        exp_logits = np.exp(logits - np.max(logits)) 
        probs = exp_logits / np.sum(exp_logits)
        
        idx = np.argmax(probs)
        conf = probs[idx]
        
        return CNN_CLASS_NAMES[idx], float(conf)
    except Exception as e:
        print(f"CNN Error: {e}")
        return "POTENTIAL", 0.0

# --- HELPER FUNCTIONS ---
def download_image(url, path):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            with open(path, "wb") as f:
                f.write(res.content)
            return True
        return False
    except: return False

def fetch_all_rows(table_name, col_name, col_value, select="*"):
    all_data = []
    offset = 0
    limit = 1000 
    
    print(f"📡 Fetching {table_name} where {col_name}={col_value}...")
    while True:
        try:
            res = supabase.table(table_name)\
                .select(select)\
                .eq(col_name, str(col_value))\
                .range(offset, offset + limit - 1)\
                .execute()
            
            data = res.data
            if not data: break
            all_data.extend(data)
            if len(data) < limit: break
            offset += limit
        except Exception as e:
            print(f"⚠️ Error fetching {table_name}: {e}")
            break
    return all_data

# --- CORE LOGIC (Logic luồng giữ nguyên, chỉ thay hàm gọi AI) ---
def process_product_images(product_id: str):
    pid_str = str(product_id)
    work_dir = os.path.join(TEMP_DIR, pid_str)
    os.makedirs(work_dir, exist_ok=True)

    print(f"🚀 START Processing Product (ONNX): {pid_str}")

    # 1. Get Shop Images
    try:
        prod_res = supabase.table("products").select("product_images").eq("id", pid_str).execute()
        shop_urls = prod_res.data[0].get('product_images', []) if prod_res.data else []
    except: shop_urls = []

    shop_embs = []
    if shop_urls:
        for i, url in enumerate(shop_urls):
            p = os.path.join(work_dir, f"shop_{i}.jpg")
            if download_image(url, p):
                emb = get_clip_emb_onnx(p) # Dùng ONNX
                if emb is not None: shop_embs.append(emb)
    shop_matrix = np.vstack(shop_embs) if shop_embs else None

    # 2. Get Review Images
    try:
        raw_comments = fetch_all_rows("raw_comment", "product_id", pid_str, select="id")
        c_ids = [str(c['id']) for c in raw_comments]
        if not c_ids: return {"status": "No comments found"}
        
        db_images = []
        for i in range(0, len(c_ids), 500):
            chunk = c_ids[i:i+500]
            img_res = supabase.table("comment_images").select("*").in_("comment_id", chunk).execute()
            if img_res.data:
                db_images.extend(img_res.data)
    except Exception as e: return {"error": str(e)}

    if not db_images: return {"status": "No images found in comments"}

    # 3. Process AI (Parallel)
    processed_results = []
    valid_items = []

    def process_one(img_row):
        fname = f"img_{img_row['id']}.jpg"
        fpath = os.path.join(work_dir, fname)
        if not download_image(img_row['image_url'], fpath): return None
        
        cnn_st, cnn_conf = classify_cnn_onnx(fpath) # Dùng ONNX
        
        item = {
            "image_id": img_row['id'],
            "cnn_status": cnn_st,
            "final_classification": cnn_st,
            "similarity_score": cnn_conf,
            "clip_embedding": None
        }
        
        if cnn_st != "TRASH":
            emb = get_clip_emb_onnx(fpath) # Dùng ONNX
            if emb is not None:
                item["clip_embedding"] = emb
                item["final_classification"] = "VALID"
        return item

    # Tăng workers vì ONNX CPU release GIL tốt hơn PyTorch
    with ThreadPoolExecutor(max_workers=8) as ex: 
        futures = [ex.submit(process_one, row) for row in db_images]
        for f in as_completed(futures):
            res = f.result()
            if res:
                processed_results.append(res)
                if res.get("clip_embedding") is not None: valid_items.append(res)

    # 4. Compare Logic
    if valid_items:
        rv_mtx = np.vstack([x['clip_embedding'] for x in valid_items])
        
        # Internal Duplicates
        sim = cosine_similarity(rv_mtx)
        for i in range(len(valid_items)):
            for j in range(i+1, len(valid_items)):
                if sim[i, j] > 0.98:
                    valid_items[j]['final_classification'] = "DUPLICATE_SPAM"

        # Shop Duplicates
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

    # 5. Save DB
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