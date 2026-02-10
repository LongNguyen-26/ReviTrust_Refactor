import os
import re
import unicodedata
import torch
import torch.nn.functional as F
import emoji
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from transformers import AutoTokenizer
from datetime import datetime, timezone
from model_defs import SpamModelVi, SentimentModelVi, SpamModelEn, SentimentModelEn

# --- CONFIG ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="ReviTrust Text Core")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- LOAD MODELS & TOKENIZERS ---
print("⏳ Initializing Text Models...")
tokenizer_vi = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
tokenizer_en = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

# Khởi tạo model architecture
spam_vi = SpamModelVi().to(DEVICE)
sent_vi = SentimentModelVi().to(DEVICE)
spam_en = SpamModelEn().to(DEVICE)
sent_en = SentimentModelEn().to(DEVICE)

# Hàm load weights
def load_weights(model, filename):
    try:
        if os.path.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=DEVICE))
            model.eval()
            print(f"✅ Loaded {filename}")
        else:
            print(f"⚠️ Weight file {filename} not found!")
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")

# Load weights (Đảm bảo file .pth đã được upload lên HF Space)
load_weights(spam_vi, "best_spam_transform-F1-0.89.pth")
load_weights(sent_vi, "best_review_transform-F108435.pth")
load_weights(spam_en, "ReviewSpamEnglish.pth")
load_weights(sent_en, "ReviewEnglishEmotion.pth")

# --- UTILS ---
def clean_text(text, is_vietnamese=True):
    if not isinstance(text, str): return ""
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
    try:
        res = supabase.table("products").select("product_link").eq("id", str(product_id)).execute()
        if res.data:
            link = res.data[0].get("product_link", "")
            if "tiki.vn" in link: return "tiki"
            if "aliexpress" in link: return "aliexpress"
    except: pass
    return "unknown"

# --- INFERENCE ---
def run_batch_spam(model, tokenizer, texts, clean_fn):
    cleaned = [clean_fn(t) for t in texts]
    enc = tokenizer(cleaned, truncation=True, padding="max_length", max_length=156, return_tensors="pt")
    input_ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids, mask)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        scores = probs[:, 1].cpu().numpy()
    return preds, scores

def run_batch_sentiment(model, tokenizer, texts, clean_fn):
    cleaned = [clean_fn(t) for t in texts]
    enc = tokenizer(cleaned, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    input_ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        out = model(input_ids, mask)
    res = {}
    for k in out: res[k] = torch.argmax(out[k], dim=1).cpu().numpy()
    return res

# --- API ---
class AnalyzeRequest(BaseModel):
    product_id: str

@app.post("/analyze_text")
async def analyze_text(req: AnalyzeRequest):
    pid = str(req.product_id)
    platform = get_platform(pid)
    
    if platform == "tiki":
        spam_model, sent_model, tokenizer = spam_vi, sent_vi, tokenizer_vi
        clean_fn = lambda x: clean_text(x, True)
    else:
        spam_model, sent_model, tokenizer = spam_en, sent_en, tokenizer_en
        clean_fn = lambda x: clean_text(x, False)

    # Fetch data
    try:
        res = supabase.table("raw_comment").select("id, content").eq("product_id", pid).execute()
        raw_data = res.data
    except Exception as e: return {"error": str(e)}

    if not raw_data: return {"msg": "No data"}
    
    texts, ids = [], []
    for r in raw_data:
        if r.get("content") and len(str(r["content"]).strip()) > 0:
            texts.append(str(r["content"]))
            ids.append(str(r["id"]))

    if not texts: return {"msg": "Empty text"}

    # Processing
    final_results = []
    BATCH = 32
    for i in range(0, len(texts), BATCH):
        b_texts = texts[i:i+BATCH]
        b_ids = ids[i:i+BATCH]

        preds, scores = run_batch_spam(spam_model, tokenizer, b_texts, clean_fn)
        
        non_spam_indices = [idx for idx, p in enumerate(preds) if p == 0]
        non_spam_texts = [b_texts[idx] for idx in non_spam_indices]
        
        sent_results = {}
        if non_spam_texts:
            raw_sent = run_batch_sentiment(sent_model, tokenizer, non_spam_texts, clean_fn)
            for loc_idx, real_idx in enumerate(non_spam_indices):
                sent_results[real_idx] = {k: raw_sent[k][loc_idx] for k in raw_sent}

        for idx, rid in enumerate(b_ids):
            # Map logic (0:None, 1:Pos, 2:Neg -> 1, -1)
            def m(v): return 1 if v==1 else (-1 if v==2 else 0)
            
            item = {
                "comment_id": rid,
                "is_spam": bool(preds[idx] == 1),
                "spam_score": float(scores[idx]),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "sentiment_overall": 0.0, "sentiment_quality": 0, "sentiment_price": 0,
                "sentiment_delivery": 0, "sentiment_packaging": 0
            }

            if idx in sent_results:
                s = sent_results[idx]
                q, p, d, pk = m(s["chat_luong"]), m(s["gia_ca"]), m(s["giao_hang"]), m(s["dong_goi"])
                item.update({"sentiment_quality": q, "sentiment_price": p, "sentiment_delivery": d, "sentiment_packaging": pk})
                item["sentiment_overall"] = (0.4*q) + (0.2*p) + (0.2*d) + (0.2*pk)
            
            final_results.append(item)

    # Save DB
    cids = [x["comment_id"] for x in final_results]
    for k in range(0, len(cids), 500):
        supabase.table("text_ai_results").delete().in_("comment_id", cids[k:k+500]).execute()
    
    for k in range(0, len(final_results), 100):
        supabase.table("text_ai_results").insert(final_results[k:k+100]).execute()

    return {"status": "success", "processed": len(final_results)}