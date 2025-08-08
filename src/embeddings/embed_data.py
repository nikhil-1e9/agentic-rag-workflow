import numpy as np
from typing import List, Iterator
from loguru import logger
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config.settings import settings

def batch_iterate(lst: List, batch_size: int) -> Iterator[List]:
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

class EmbedData:
    """Handles document embedding with binary quantization support."""
    
    def __init__(
        self, 
        embed_model_name: str = None,
        batch_size: int = None,
        cache_folder: str = None
    ):
        self.embed_model_name = embed_model_name or settings.embedding_model
        self.batch_size = batch_size or settings.batch_size
        self.cache_folder = cache_folder or settings.hf_cache_dir
        
        self.embed_model = self._load_embed_model()
        self.embeddings: List[List[float]] = []
        self.binary_embeddings: List[bytes] = []
        self.contexts: List[str] = []

    def _load_embed_model(self) -> HuggingFaceEmbedding:
        """Load the embedding model."""
        logger.info(f"Loading embedding model: {self.embed_model_name}")
        
        embed_model = HuggingFaceEmbedding(
            model_name=self.embed_model_name,
            trust_remote_code=True,
            cache_folder=self.cache_folder
        )
        return embed_model

    def _binary_quantize(self, embeddings: List[List[float]]) -> List[bytes]:
        """Convert float32 embeddings to binary vectors."""
        embeddings_array = np.array(embeddings)
        binary_embeddings = np.where(embeddings_array > 0, 1, 0).astype(np.uint8)
        
        # Pack bits into bytes (8 dimensions per byte)
        packed_embeddings = np.packbits(binary_embeddings, axis=1)
        return [vec.tobytes() for vec in packed_embeddings]

    def generate_embedding(self, contexts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of contexts."""
        return self.embed_model.get_text_embedding_batch(contexts)

    def embed(self, contexts: List[str]) -> None:
        """Generate embeddings for all contexts with binary quantization."""
        self.contexts = contexts
        logger.info(f"Generating embeddings for {len(contexts)} contexts...")

        for batch_context in batch_iterate(contexts, self.batch_size):
            # Generate float32 embeddings
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)

            # Convert to binary and store
            binary_batch = self._binary_quantize(batch_embeddings)
            self.binary_embeddings.extend(binary_batch)

        logger.info(f"Generated {len(self.embeddings)} embeddings with binary quantization")

    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_model.get_query_embedding(query)

    def binary_quantize_query(self, query_embedding: List[float]) -> bytes:
        """Convert query embedding to binary format."""
        embedding_array = np.array([query_embedding])
        binary_embedding = np.where(embedding_array > 0, 1, 0).astype(np.uint8)
        packed_embedding = np.packbits(binary_embedding, axis=1)
        return packed_embedding[0].tobytes()

    def clear(self) -> None:
        """Clear stored embeddings and contexts."""
        self.embeddings.clear()
        self.binary_embeddings.clear()
        self.contexts.clear()
        logger.info("Cleared all embeddings and contexts")