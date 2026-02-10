from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.services.crawler import CrawlerService
from app.services.ai_client import ai_client
from app.services.analytics import AnalyticsService
import asyncio

app = FastAPI(title="ReviTrust Core Gateway")

class PipelineRequest(BaseModel):
    product_url: str

@app.get("/")
def health_check():
    return {"status": "ReviTrust Core is running"}

@app.post("/pipeline/execute")
async def execute_pipeline(req: PipelineRequest):
    crawler = CrawlerService()
    analytics = AnalyticsService()
    
    print(f"🚀 Starting pipeline for: {req.product_url}")

    # BƯỚC 1: CRAWL
    try:
        crawl_res = crawler.crawl(req.product_url)
        product_id = crawl_res['product_id']
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Crawl failed: {str(e)}")

    # BƯỚC 2: GỌI AI (SONG SONG)
    # Vì DB đã có dữ liệu sau bước Crawl, ta gọi 2 service AI xử lý data đó
    print("⏳ Calling External AI Services...")
    text_task = ai_client.call_text_analysis(product_id)
    image_task = ai_client.call_image_analysis(product_id)
    
    # Chạy song song và đợi kết quả
    text_res, image_res = await asyncio.gather(text_task, image_task)
    
    # Kiểm tra lỗi AI (Optional: có thể log warning thay vì raise error)
    if "error" in text_res: print(f"⚠️ Text AI Warning: {text_res['error']}")
    if "error" in image_res: print(f"⚠️ Image AI Warning: {image_res['error']}")

    # BƯỚC 3: ANALYTICS & SCORING
    print("📊 Computing Analytics...")
    try:
        final_result = analytics.compute_trust_score(product_id)
        return {"status": "success", "data": final_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)