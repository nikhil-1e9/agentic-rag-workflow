import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: str
    firecrawl_api_key: str
    
    # Model Configuration
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    llm_model: str = "gpt-3.5-turbo"
    vector_dim: int = 1024
    
    # Retrieval Configuration
    top_k: int = 5
    batch_size: int = 512
    rerank_top_k: int = 3
    
    # Database Configuration
    milvus_db_path: str = "./data/milvus_binary.db"
    collection_name: str = "legal_documents"
    
    # Cache Configuration
    hf_cache_dir: str = "./cache/hf_cache"
    
    # Workflow Configuration
    temperature: float = 0.6
    max_tokens: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __post_init__(self):
        """Create necessary directories."""
        Path(self.milvus_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.hf_cache_dir).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()