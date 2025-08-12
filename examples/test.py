import asyncio
import os
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.embed_data import EmbedData
from src.indexing.milvus_vdb import MilvusVDB
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from src.workflows.agent_workflow import EnhancedRAGWorkflow
from llama_index.core import SimpleDirectoryReader
from config.settings import settings

async def main():
    """Main function demonstrating the complete pipeline."""
    
    # Initialize logger
    logger.info("Starting Enhanced RAG Pipeline Demo")
    
    # Check required environment variables
    required_env_vars = ["OPENAI_API_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            logger.error(f"Missing required environment variable: {var}")
            return
    
    try:
        # Step 1: Load and process document
        logger.info("Step 1: Loading document...")
        
        # Replace with your PDF file path
        pdf_path = "./data/raft.pdf"
        if not pdf_path or not Path(pdf_path).exists():
            logger.error("Invalid PDF path provided")
            return
        
        # Load and split document
        text_chunks = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        text_chunks = [chunk.text for chunk in text_chunks]
        logger.info(f"Created {len(text_chunks)} text chunks")
        
        # Step 2: Create embeddings
        logger.info("Step 2: Creating embeddings...")
        
        embed_data = EmbedData(
            embed_model_name=settings.embedding_model,
            batch_size=settings.batch_size
        )
        
        # Generate embeddings with binary quantization
        embed_data.embed(text_chunks)
        logger.info("Embeddings created successfully")
        
        # Step 3: Setup vector database
        logger.info("Step 3: Setting up vector database...")
        
        vector_db = MilvusVDB(
            collection_name=settings.collection_name,
            vector_dim=settings.vector_dim,
            batch_size=settings.batch_size,
            db_file=settings.milvus_db_path
        )
        
        # Initialize database and create collection
        vector_db.initialize_client()
        vector_db.create_collection()
        
        # Ingest data
        vector_db.ingest_data(embed_data)
        logger.info("Vector database setup completed")
        
        # Step 4: Setup retrieval system
        logger.info("Step 4: Setting up retrieval system...")
        
        retriever = Retriever(
            vector_db=vector_db,
            embed_data=embed_data,
            top_k=settings.top_k
        )
        
        # Step 5: Setup RAG system
        logger.info("Step 5: Setting up RAG system...")
        
        rag_system = RAG(
            retriever=retriever,
            llm_model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        # Step 6: Setup workflow
        logger.info("Step 6: Setting up enhanced workflow...")
        
        workflow = EnhancedRAGWorkflow(
            retriever=retriever,
            rag_system=rag_system,
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        logger.info("Setup completed! Ready for queries.")
        
        # Interactive query loop
        print("\n" + "="*60)
        print("Enhanced RAG Pipeline Ready!")
        print("Type 'quit' to exit, 'help' for commands")
        print("="*60)
        
        while True:
            try:
                query = input("\nEnter your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif not query:
                    continue
                
                logger.info(f"Processing query: {query}")
                
                # Run the complete workflow
                result = await workflow.run_workflow(query)
                
                # Display results
                print("\n" + "-"*60)
                print("ANSWER:")
                print(result["answer"])
                
                if result.get("web_search_used", False):
                    print(f"\n🌐 Web search was used to enhance the response")
                else:
                    print(f"\n📚 Response based on document knowledge")
                
                print("-"*60)
                
                # Option to show detailed results
                show_details = input("\nShow detailed results? (y/n): ").strip().lower()
                if show_details == 'y':
                    print_detailed_results(result)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")
        
        # Cleanup
        logger.info("Cleaning up...")
        vector_db.close()
        logger.info("Demo completed")
        
    except Exception as e:
        logger.error(f"Pipeline setup failed: {e}")
        print(f"Setup failed: {e}")

def print_detailed_results(result):
    """Print detailed workflow results."""
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    print(f"\nOriginal Query: {result['query']}")
    
    if result.get('rag_response'):
        print(f"\nRAG Response:")
        print(result['rag_response'])
    
    if result.get('web_search_used') and result.get('web_results'):
        print(f"\nWeb Search Results:")
        print(result['web_results'][:500] + "..." if len(result['web_results']) > 500 else result['web_results'])
    
    if result.get('error'):
        print(f"\nError: {result['error']}")
    
    print("="*60)

async def test_retrieval():
    """Test retrieval without user input."""
    logger.info("Running retrieval test...")
    
    # Create a simple test document
    test_text = [
        "This is a test document about artificial intelligence.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers."
    ]
    
    # Test embedding
    embed_data = EmbedData()
    embed_data.embed(test_text)
    
    # Test vector database
    vector_db = MilvusVDB(collection_name="test_collection")
    vector_db.initialize_client()
    vector_db.create_collection()
    vector_db.ingest_data(embed_data)
    
    # Test retrieval
    retriever = Retriever(vector_db, embed_data)
    results = retriever.search("What is machine learning?")
    
    logger.info(f"Test completed. Retrieved {len(results)} results")
    
    # Cleanup
    vector_db.close()
    
    return True

if __name__ == "__main__":
    asyncio.run(main())