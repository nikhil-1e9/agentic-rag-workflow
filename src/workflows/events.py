from typing import List, Optional
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

class RetrieveEvent(Event):
    """Event containing retrieved nodes from vector database."""
    retrieved_nodes: List[NodeWithScore]
    query: str

class EvaluateEvent(Event): 
    """Event for evaluating RAG response quality."""
    rag_response: str
    retrieved_nodes: List[NodeWithScore]
    query: str

class WebSearchEvent(Event):
    """Event for web search when RAG response is insufficient."""
    rag_response: str
    query: str
    search_results: Optional[str] = None

class SynthesizeEvent(Event):
    """Event for final response synthesis."""
    rag_response: str
    web_search_results: Optional[str] = None
    retrieved_nodes: List[NodeWithScore]
    query: str
    use_web_results: bool = False