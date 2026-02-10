import httpx
import asyncio
from app.config import settings

class AIClient:
    async def call_text_analysis(self, product_id: str):
        """Gửi request trigger Text AI Service xử lý product_id"""
        async with httpx.AsyncClient() as client:
            try:
                # Giả sử API Text của bạn nhận {"product_id": "..."} và tự query DB xử lý
                response = await client.post(
                    settings.TEXT_AI_API_URL, 
                    json={"product_id": str(product_id)},
                    timeout=300 # Timeout dài vì xử lý batch
                )
                return response.json()
            except Exception as e:
                print(f"Text AI Error: {e}")
                return {"error": str(e)}

    async def call_image_analysis(self, product_id: str):
        """Gửi request trigger Image AI Service xử lý product_id"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    settings.IMAGE_AI_API_URL, 
                    json={"product_id": str(product_id)},
                    timeout=600 # Xử lý ảnh lâu hơn
                )
                return response.json()
            except Exception as e:
                print(f"Image AI Error: {e}")
                return {"error": str(e)}

ai_client = AIClient()