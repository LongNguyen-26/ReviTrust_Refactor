from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Supabase Config
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    # External AI Services (URL của 2 HF Space kia)
    TEXT_AI_API_URL: str  # Ví dụ: https://huggingface.co/spaces/user/text-api/predict
    IMAGE_AI_API_URL: str # Ví dụ: https://huggingface.co/spaces/user/image-api/predict
    
    class Config:
        env_file = ".env"

settings = Settings()