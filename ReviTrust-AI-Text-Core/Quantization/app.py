import os
import re
import unicodedata
import numpy as np
import emoji
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from transformers import AutoTokenizer
from datetime import datetime, timezone
import onnxruntime as ort

# --- CONFIG ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

app = FastAPI(title="ReviTrust Text Core - ONNX Optimized")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ... (Giữ nguyên các imports)

# --- LOAD ONNX MODELS & TOKENIZERS ---
print("⏳ Initializing ONNX Runtime Text Models...")

# Cấu hình ONNX Runtime tối ưu cho CPU trên HF Space
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = os.cpu_count() or 2 # Tự động lấy số core
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load tokenizers (Lưu ý: PhoBERT cần use_fast=False)
print("📦 Loading tokenizers...")
try:
    tokenizer_vi = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    tokenizer_en = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
except Exception as e:
    print(f"⚠️ Error loading tokenizers: {e}")

# Load ONNX models
def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        print(f"⚠️  Model file {model_path} not found!")
        return None
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        print(f"✅ Loaded {model_path}")
        return session
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return None

print("🔧 Loading quantized ONNX models...")
# SỬA: Trỏ đúng vào file _quantized đã tạo từ script quantize
spam_vi_session = load_onnx_model("onnx_models/spam_vi_quantized.onnx")
sent_vi_session = load_onnx_model("onnx_models/sent_vi_quantized.onnx")
spam_en_session = load_onnx_model("onnx_models/spam_en_quantized.onnx")
sent_en_session = load_onnx_model("onnx_models/sent_en_quantized.onnx")

# --- UTILS ---
def clean_text(text, is_vietnamese=True):
    """Text preprocessing"""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = emoji.replace_emoji(text, "")
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')
    text = re.sub(r"\S+@\S+", "", text)
    if is_vietnamese:
        text = re.sub(r"[^a-zA-Z0-9À-ỹ \.,!?\"']", " ", text)
    else:
        text = re.sub(r"[^a-zA-Z0-9 \.,!?\"']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_platform(product_id):
    """Xác định platform từ product_link"""
    try:
        res = supabase.table("products").select("product_link").eq("id", str(product_id)).execute()
        if res.data:
            link = res.data[0].get("product_link", "")
            if "tiki.vn" in link: 
                return "tiki"
            if "aliexpress" in link: 
                return "aliexpress"
    except: 
        pass
    return "unknown"

# --- INFERENCE WITH ONNX ---
def run_batch_spam_onnx(session, tokenizer, texts, clean_fn, max_length=156):
    """Run spam detection với ONNX model (Binary Classification)"""
    if session is None or not texts:
        return [], []
    
    cleaned = [clean_fn(t) for t in texts]
    enc = tokenizer(
        cleaned, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors="np"
    )
    
    # SỬA: Ép kiểu int64 bắt buộc cho ONNX Runtime
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # SỬA: Export script quy định output_names=["output"], nên ta lấy [0]
    # Output shape spam model thường là (batch_size, 2)
    try:
        logits = session.run(["output"], ort_inputs)[0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        preds = np.argmax(probs, axis=1)
        scores = probs[:, 1] # Lấy probability của lớp Spam (index 1)
        
        return preds, scores
    except Exception as e:
        print(f"❌ Error in spam inference: {e}")
        return [0]*len(texts), [0.0]*len(texts)

def run_batch_sentiment_onnx(session, tokenizer, texts, clean_fn, max_length=128):
    """
    Run sentiment analysis.
    LƯU Ý: Do script export dùng output_names=['output'], model trả về 1 tensor duy nhất.
    Giả định model Sentiment gộp 4 aspect (Giao hàng, Chất lượng, Giá, Đóng gói).
    Shape kỳ vọng: (Batch, 4, 3) hoặc (Batch, 12).
    """
    if session is None or not texts:
        return {}
    
    cleaned = [clean_fn(t) for t in texts]
    enc = tokenizer(
        cleaned, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors="np"
    )
    
    # SỬA: Ép kiểu int64
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    try:
        # Lấy single output
        output_tensor = session.run(["output"], ort_inputs)[0]
        
        # Xử lý shape: Nếu output bị flatten thành (Batch, 12), reshape lại thành (Batch, 4, 3)
        # 4 aspects, 3 classes (0,1,2)
        if len(output_tensor.shape) == 2 and output_tensor.shape[1] == 12:
            output_tensor = output_tensor.reshape(-1, 4, 3)
        elif len(output_tensor.shape) == 2 and output_tensor.shape[1] == 4:
            # Trường hợp regression hoặc binary per aspect (ít khả năng hơn với output cũ)
             pass 

        # Argmax trên dimension cuối cùng để lấy class (0, 1, 2)
        # Shape kết quả: (Batch, 4)
        preds = np.argmax(output_tensor, axis=-1)
        
        # Map kết quả vào từng aspect
        # Thứ tự aspect phụ thuộc vào lúc train, giả định theo code cũ:
        # 0: Giao hàng, 1: Chất lượng, 2: Giá cả, 3: Đóng gói (Cần verify lại model gốc của bạn)
        res = {}
        for i in range(len(texts)):
            # Mapping này phải khớp với thứ tự trong SentimentModelVi/En của bạn
            # Code cũ của bạn: ["giao_hang", "chat_luong", "gia_ca", "dong_goi"]
            # Tôi giả định thứ tự tensor cũng vậy.
            res[i] = {
                "giao_hang": preds[i][0],
                "chat_luong": preds[i][1],
                "gia_ca": preds[i][2],
                "dong_goi": preds[i][3]
            }
        return res

    except Exception as e:
        print(f"❌ Error in sentiment inference: {e}")
        # Return default neutral values on error
        return {i: {"giao_hang":0, "chat_luong":0, "gia_ca":0, "dong_goi":0} for i in range(len(texts))}

# --- API ---
class AnalyzeRequest(BaseModel):
    product_id: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "backend": "ONNX Runtime",
        "optimization": "Dynamic Quantization (QUInt8)",
        "models_loaded": {
            "spam_vi": spam_vi_session is not None,
            "sent_vi": sent_vi_session is not None,
            "spam_en": spam_en_session is not None,
            "sent_en": sent_en_session is not None
        }
    }

@app.post("/analyze_text")
async def analyze_text(req: AnalyzeRequest):
    """
    Phân tích text comments cho một product
    - Spam detection
    - Sentiment analysis (4 aspects: quality, price, delivery, packaging)
    """
    pid = str(req.product_id)
    platform = get_platform(pid)
    
    # Chọn models dựa trên platform
    if platform == "tiki":
        spam_session = spam_vi_session
        sent_session = sent_vi_session
        tokenizer = tokenizer_vi
        clean_fn = lambda x: clean_text(x, True)
        spam_max_len = 156
        sent_max_len = 128
    else:
        spam_session = spam_en_session
        sent_session = sent_en_session
        tokenizer = tokenizer_en
        clean_fn = lambda x: clean_text(x, False)
        spam_max_len = 156
        sent_max_len = 128
    
    # Kiểm tra models đã load
    if spam_session is None or sent_session is None:
        raise HTTPException(
            status_code=500, 
            detail=f"Models for platform '{platform}' not loaded"
        )

    # Fetch data từ Supabase
    try:
        res = supabase.table("raw_comment").select("id, content").eq("product_id", pid).execute()
        raw_data = res.data
    except Exception as e: 
        return {"error": str(e)}

    if not raw_data: 
        return {"msg": "No data", "processed": 0}
    
    # Filter valid texts
    texts, ids = [], []
    for r in raw_data:
        if r.get("content") and len(str(r["content"]).strip()) > 0:
            texts.append(str(r["content"]))
            ids.append(str(r["id"]))

    if not texts: 
        return {"msg": "Empty text", "processed": 0}

    # Processing theo batch
    final_results = []
    BATCH_SIZE = 32
    
    for i in range(0, len(texts), BATCH_SIZE):
        b_texts = texts[i:i+BATCH_SIZE]
        b_ids = ids[i:i+BATCH_SIZE]

        # Spam detection
        preds, scores = run_batch_spam_onnx(
            spam_session, 
            tokenizer, 
            b_texts, 
            clean_fn,
            spam_max_len
        )
        
        # Chỉ phân tích sentiment cho non-spam comments
        non_spam_indices = [idx for idx, p in enumerate(preds) if p == 0]
        non_spam_texts = [b_texts[idx] for idx in non_spam_indices]
        
        sent_results = {}
        if non_spam_texts:
            raw_sent = run_batch_sentiment_onnx(
                sent_session, 
                tokenizer, 
                non_spam_texts, 
                clean_fn,
                sent_max_len
            )
            for loc_idx, real_idx in enumerate(non_spam_indices):
                sent_results[real_idx] = {k: raw_sent[k][loc_idx] for k in raw_sent}

        # Build results
        for idx, rid in enumerate(b_ids):
            # Map sentiment values (0:None, 1:Pos, 2:Neg -> 0, 1, -1)
            def map_sentiment(v): 
                return 1 if v == 1 else (-1 if v == 2 else 0)
            
            item = {
                "comment_id": rid,
                "is_spam": bool(preds[idx] == 1),
                "spam_score": float(scores[idx]),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "sentiment_overall": 0.0,
                "sentiment_quality": 0,
                "sentiment_price": 0,
                "sentiment_delivery": 0,
                "sentiment_packaging": 0
            }

            # Nếu không phải spam, cập nhật sentiment
            if idx in sent_results:
                s = sent_results[idx]
                q = map_sentiment(s["chat_luong"])
                p = map_sentiment(s["gia_ca"])
                d = map_sentiment(s["giao_hang"])
                pk = map_sentiment(s["dong_goi"])
                
                item.update({
                    "sentiment_quality": int(q),
                    "sentiment_price": int(p),
                    "sentiment_delivery": int(d),
                    "sentiment_packaging": int(pk)
                })
                
                # Overall sentiment (weighted average)
                item["sentiment_overall"] = float(
                    (0.4 * q) + (0.2 * p) + (0.2 * d) + (0.2 * pk)
                )
            
            final_results.append(item)

    # Save to database
    cids = [x["comment_id"] for x in final_results]
    
    # Delete existing results (batch delete)
    for k in range(0, len(cids), 500):
        supabase.table("text_ai_results").delete().in_("comment_id", cids[k:k+500]).execute()
    
    # Insert new results (batch insert)
    for k in range(0, len(final_results), 100):
        supabase.table("text_ai_results").insert(final_results[k:k+100]).execute()

    return {
        "status": "success",
        "processed": len(final_results),
        "platform": platform,
        "backend": "ONNX Runtime (Quantized)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)