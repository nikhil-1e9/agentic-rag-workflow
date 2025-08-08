from typing import List, Dict, Any, Optional
from loguru import logger
from llama_index.core.schema import NodeWithScore, TextNode
from src.indexing.milvus_vdb import MilvusVDB
from src.embeddings.embed_data import EmbedData
from config.settings import settings

class Retriever:
    """Simplified retriever without reranking for demo purposes."""
    
    def __init__(
        self, 
        vector_db: MilvusVDB, 
        embed_data: EmbedData, 
        top_k: int = None
    ):
        self.vector_db = vector_db
        self.embed_data = embed_data
        self.top_k = top_k or settings.top_k

    def search(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Search for relevant documents using vector similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of NodeWithScore objects
        """
        if top_k is None:
            top_k = self.top_k

        # Generate query embedding and convert to binary
        query_embedding = self.embed_data.get_query_embedding(query)
        binary_query = self.embed_data.binary_quantize_query(query_embedding)

        # Perform vector search
        search_results = self.vector_db.search(
            binary_query=binary_query,
            top_k=top_k,
            output_fields=["context"]
        )

        # Convert to NodeWithScore objects
        nodes_with_scores = []
        for result in search_results:
            node = TextNode(
                text=result["payload"]["context"],
                id_=str(result["id"])
            )
            node_with_score = NodeWithScore(
                node=node,
                score=result["score"]
            )
            nodes_with_scores.append(node_with_score)

        logger.info(f"Retrieved {len(nodes_with_scores)} results for query")
        return nodes_with_scores

    def get_contexts(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Get context strings from search results."""
        nodes_with_scores = self.search(query, top_k)
        return [node.node.text for node in nodes_with_scores]

    def get_combined_context(self, query: str, top_k: Optional[int] = None, separator: str = "\n\n---\n\n") -> str:
        """Get combined context from search results."""
        contexts = self.get_contexts(query, top_k)
        return separator.join(contexts)

    def search_with_scores(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search and return results with detailed scoring information."""
        nodes_with_scores = self.search(query, top_k)
        
        results = []
        for node_with_score in nodes_with_scores:
            results.append({
                "context": node_with_score.node.text,
                "score": node_with_score.score,
                "node_id": node_with_score.node.id_,
                "metadata": node_with_score.node.metadata or {}
            })
        
        return results