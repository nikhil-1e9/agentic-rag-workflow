from typing import Optional, Generator, Any
from loguru import logger
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from src.retrieval.retriever_rerank import Retriever
from config.settings import settings

class RAG:
    """RAG (Retrieval-Augmented Generation) system with OpenAI integration."""
    
    def __init__(
        self, 
        retriever: Retriever, 
        llm_model: str = None,
        openai_api_key: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        self.retriever = retriever
        self.llm_model = llm_model or settings.llm_model
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        
        # Initialize LLM
        self.llm = self._setup_llm()
        
        # System message for consistent behavior
        self.system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that answers questions based on the provided context. "
                   "Always base your answers on the given information and clearly indicate when you don't know something."
        )
        
        # Default prompt template
        self.prompt_template = PromptTemplate(
            template=(
                "CONTEXT:\n"
                "{context}\n"
                "---------------------\n"
                "Based on the context information above, please answer the following question. "
                "If the context doesn't contain enough information to answer the question, or "
                "even if you know the answer, but it is not relevant to the provided context, "
                "clearly state that you don't know and explain what information is missing.\n\n"
                "QUESTION: {query}\n"
                "ANSWER: "
            )
        )

    def _setup_llm(self) -> OpenAI:
        """Initialize the OpenAI LLM."""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )
        
        llm = OpenAI(
            model=self.llm_model,
            api_key=self.openai_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        logger.info(f"Initialized OpenAI LLM with model: {self.llm_model}")
        return llm

    def generate_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Generate context from retrieval results."""
        return self.retriever.get_combined_context(query, top_k)

    def query(self, query: str, stream: bool = False, top_k: Optional[int] = None) -> Any:
        """
        Query the RAG system.
        
        Args:
            query: User question
            stream: Whether to stream the response
            top_k: Number of retrieval results to use
            
        Returns:
            Response text (string) or streaming response object
        """
        # Generate context from retrieval
        context = self.generate_context(query, top_k)
        
        # Create prompt from template
        prompt = self.prompt_template.format(context=context, query=query)
        
        if stream:
            # Stream response
            streaming_response = self.llm.stream_complete(prompt)
            return streaming_response
        else:
            # Complete response
            response = self.llm.complete(prompt)
            return response.text

    def chat_query(
        self, 
        query: str, 
        chat_history: Optional[list] = None,
        stream: bool = False, 
        top_k: Optional[int] = None
    ) -> Any:
        """
        Query the RAG system with chat history support.
        
        Args:
            query: User question
            chat_history: Previous chat messages
            stream: Whether to stream the response
            top_k: Number of retrieval results to use
            
        Returns:
            Response text (string) or streaming response object
        """
        # Generate context from retrieval
        context = self.generate_context(query, top_k)
        
        # Create prompt from template
        prompt = self.prompt_template.format(context=context, query=query)
        
        # Build message history
        messages = [self.system_message]
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add current user message
        user_message = ChatMessage(role=MessageRole.USER, content=prompt)
        messages.append(user_message)
        
        if stream:
            # Stream chat response
            streaming_response = self.llm.stream_chat(messages)
            return streaming_response
        else:
            # Complete chat response
            chat_response = self.llm.chat(messages)
            return chat_response.message.content

    def get_detailed_response(self, query: str, top_k: Optional[int] = None) -> dict:
        """
        Get detailed response with context and sources.
        
        Args:
            query: User question
            top_k: Number of retrieval results to use
            
        Returns:
            Dictionary with response, context, and source information
        """
        # Get retrieval results with scores
        retrieval_results = self.retriever.search_with_scores(query, top_k)
        
        # Generate context
        context = self.retriever.get_combined_context(query, top_k)
        
        # Generate response
        response = self.query(query, stream=False, top_k=top_k)
        
        return {
            "response": response,
            "context": context,
            "sources": retrieval_results,
            "query": query,
            "model": self.llm_model
        }

    def set_prompt_template(self, template: str) -> None:
        """Set a custom prompt template."""
        self.prompt_template = PromptTemplate(template=template)
        logger.info("Updated prompt template")

    def set_system_message(self, content: str) -> None:
        """Set a custom system message."""
        self.system_message = ChatMessage(role=MessageRole.SYSTEM, content=content)
        logger.info("Updated system message")

    def update_llm_params(self, **kwargs) -> None:
        """Update LLM parameters."""
        for key, value in kwargs.items():
            if hasattr(self.llm, key):
                setattr(self.llm, key, value)
                logger.info(f"Updated LLM parameter: {key} = {value}")
            else:
                logger.warning(f"Unknown LLM parameter: {key}")