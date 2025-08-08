from typing import Optional, Any, List
from loguru import logger
from firecrawl import FirecrawlApp

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    step,
    Workflow,
    Context,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore

from .events import RetrieveEvent, EvaluateEvent, WebSearchEvent, SynthesizeEvent
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from config.settings import settings

# Prompt templates for workflow steps
ROUTER_EVALUATION_TEMPLATE = PromptTemplate(
    template="""You are a quality evaluator for RAG responses. Your task is to determine if the given response adequately answers the user's question.

USER QUESTION:
{query}

RAG RESPONSE:
{rag_response}

EVALUATION CRITERIA:
- Does the response directly address the user's question?
- Is the response factually coherent and well-structured?
- Does the response contain sufficient detail to be helpful?
- If the response says "I don't know" or similar, is it because the context truly lacks the information?

Please evaluate the response quality and respond with either:
- "SATISFACTORY" - if the response adequately answers the question
- "UNSATISFACTORY" - if the response is incomplete, unclear, or doesn't answer the question

Your evaluation (SATISFACTORY or UNSATISFACTORY):"""
)

QUERY_OPTIMIZATION_TEMPLATE = PromptTemplate(
    template="""Optimize the following query for web search to get the most relevant and accurate results.

Original Query: {query}

Guidelines:
- Make the query more specific and searchable
- Add relevant keywords that would help find authoritative sources
- Keep it concise but comprehensive
- Focus on the core information need

Optimized Query:"""
)

SYNTHESIS_TEMPLATE = PromptTemplate(
    template="""You are a response synthesizer. Create a comprehensive and accurate answer based on the available information.

USER QUESTION:
{query}

RAG RESPONSE (from document knowledge):
{rag_response}

WEB SEARCH RESULTS (additional context):
{web_results}

INSTRUCTIONS:
- Synthesize information from both sources to provide the most complete answer
- Prioritize information from reliable sources
- If there are contradictions, acknowledge them
- Clearly indicate when information comes from web search vs document knowledge
- If web results are empty, refine and improve the RAG response

SYNTHESIZED RESPONSE:"""
)

class EnhancedRAGWorkflow(Workflow):
    """Enhanced RAG Workflow with router agent and web search fallback."""
    
    def __init__(
        self,
        retriever: Retriever,
        rag_system: RAG,
        firecrawl_api_key: str = None,
        openai_api_key: str = None,
        **kwargs: Any
    ) -> None:
        """Initialize the workflow."""
        super().__init__(**kwargs)
        self.retriever = retriever
        self.rag = rag_system
        
        # Initialize LLM for router and synthesis
        self.llm = OpenAI(
            api_key=openai_api_key or settings.openai_api_key,
            model=settings.llm_model,
            temperature=0.1  # Lower temperature for evaluation tasks
        )
        
        # Initialize Firecrawl for web search
        self.firecrawl_api_key = firecrawl_api_key or settings.firecrawl_api_key
        self.firecrawl = FirecrawlApp(api_key=self.firecrawl_api_key) if self.firecrawl_api_key else None
        
        if not self.firecrawl:
            logger.warning("Firecrawl API key not provided. Web search will be disabled.")

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieveEvent:
        """Retrieve relevant documents from vector database."""
        query = ev.get("query")
        top_k = ev.get("top_k", settings.top_k)
        
        if not query:
            raise ValueError("Query is required")
        
        logger.info(f"Retrieving documents for query: {query}")
        
        # Retrieve relevant documents
        retrieved_nodes = self.retriever.search(query, top_k=top_k)
        
        # Store query in context for later steps
        await ctx.set("query", query)
        
        logger.info(f"Retrieved {len(retrieved_nodes)} documents")
        return RetrieveEvent(retrieved_nodes=retrieved_nodes, query=query)

    @step
    async def generate_rag_response(self, ctx: Context, ev: RetrieveEvent) -> EvaluateEvent:
        """Generate initial RAG response."""
        query = ev.query
        retrieved_nodes = ev.retrieved_nodes
        
        logger.info("Generating RAG response")
        
        # Generate response using RAG system
        rag_response = self.rag.query(query, stream=False)
        
        logger.info("RAG response generated")
        return EvaluateEvent(
            rag_response=rag_response,
            retrieved_nodes=retrieved_nodes,
            query=query
        )

    @step
    async def evaluate_response(
        self, ctx: Context, ev: EvaluateEvent
    ) -> WebSearchEvent | SynthesizeEvent:
        """Evaluate RAG response quality and route accordingly."""
        rag_response = ev.rag_response
        query = ev.query
        retrieved_nodes = ev.retrieved_nodes
        
        logger.info("Evaluating RAG response quality")
        
        # Create evaluation prompt
        evaluation_prompt = ROUTER_EVALUATION_TEMPLATE.format(
            query=query,
            rag_response=rag_response
        )
        
        # Get evaluation from LLM
        evaluation_response = self.llm.complete(evaluation_prompt)
        evaluation = evaluation_response.text.strip().upper()
        
        logger.info(f"Evaluation result: {evaluation}")
        
        if "SATISFACTORY" in evaluation:
            # RAG response is good, proceed to synthesis for refinement
            return SynthesizeEvent(
                rag_response=rag_response,
                web_search_results=None,
                retrieved_nodes=retrieved_nodes,
                query=query,
                use_web_results=False
            )
        else:
            # RAG response is insufficient, trigger web search
            return WebSearchEvent(
                rag_response=rag_response,
                query=query
            )

    @step
    async def web_search(self, ctx: Context, ev: WebSearchEvent) -> SynthesizeEvent:
        """Perform web search for additional information."""
        query = ev.query
        rag_response = ev.rag_response
        retrieved_nodes = await ctx.get("retrieved_nodes", [])
        
        logger.info("Performing web search")
        
        search_results = ""
        
        if self.firecrawl:
            try:
                # Optimize query for web search
                optimization_prompt = QUERY_OPTIMIZATION_TEMPLATE.format(query=query)
                optimized_response = self.llm.complete(optimization_prompt)
                optimized_query = optimized_response.text.strip()
                
                logger.info(f"Optimized query: {optimized_query}")
                
                # Perform web search using Firecrawl
                search_response = self.firecrawl.search(optimized_query, limit=5)
                
                if search_response and 'data' in search_response:
                    # Extract content from search results
                    search_contents = []
                    for result in search_response['data'][:3]:  # Use top 3 results
                        if 'content' in result and result['content']:
                            # Truncate content to avoid token limits
                            content = result['content'][:1000]
                            title = result.get('title', 'No title')
                            url = result.get('url', 'No URL')
                            search_contents.append(f"Title: {title}\nURL: {url}\nContent: {content}")
                    
                    search_results = "\n\n---\n\n".join(search_contents)
                    logger.info(f"Retrieved {len(search_contents)} web search results")
                else:
                    logger.warning("No search results found")
                    search_results = "No relevant web search results found."
                    
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                search_results = "Web search unavailable due to technical issues."
        else:
            logger.warning("Web search skipped - Firecrawl not configured")
            search_results = "Web search unavailable - API not configured."
        
        return SynthesizeEvent(
            rag_response=rag_response,
            web_search_results=search_results,
            retrieved_nodes=retrieved_nodes,
            query=query,
            use_web_results=True
        )

    @step
    async def synthesize_response(self, ctx: Context, ev: SynthesizeEvent) -> StopEvent:
        """Synthesize final response from RAG and web search results."""
        rag_response = ev.rag_response
        web_results = ev.web_search_results or ""
        query = ev.query
        use_web_results = ev.use_web_results
        
        logger.info("Synthesizing final response")
        
        if use_web_results and web_results:
            # Synthesize response from both RAG and web search
            synthesis_prompt = SYNTHESIS_TEMPLATE.format(
                query=query,
                rag_response=rag_response,
                web_results=web_results
            )
            
            final_response = self.llm.complete(synthesis_prompt)
            synthesized_answer = final_response.text
            
            result = {
                "answer": synthesized_answer,
                "rag_response": rag_response,
                "web_search_used": True,
                "web_results": web_results,
                "query": query
            }
        else:
            # Just refine the RAG response
            refinement_prompt = f"""Improve and refine the following response to make it more helpful and comprehensive:

Original Response: {rag_response}

Refined Response:"""
            
            refined_response = self.llm.complete(refinement_prompt)
            
            result = {
                "answer": refined_response.text,
                "rag_response": rag_response,
                "web_search_used": False,
                "web_results": None,
                "query": query
            }
        
        logger.info("Final response synthesized")
        return StopEvent(result=result)

    async def run_workflow(self, query: str, top_k: Optional[int] = None) -> dict:
        """
        Run the complete workflow for a given query.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with final answer and metadata
        """
        try:
            result = await self.run(query=query, top_k=top_k)
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "rag_response": None,
                "web_search_used": False,
                "web_results": None,
                "query": query,
                "error": str(e)
            }