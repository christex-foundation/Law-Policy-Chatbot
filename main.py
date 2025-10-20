from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import json
import os
import asyncio

# Import your existing RAG system
from rag import rag_chain, llm, create_rag_chain
from dataLoading import initialize_vectorstore


# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The legal question to ask")
    stream: bool = Field(default=False, description="Enable streaming response")


class SourceDocument(BaseModel):
    source: str
    page: Optional[int]
    content_preview: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    status: str = "success"


class ErrorResponse(BaseModel):
    error: str
    status: str = "error"


class RebuildResponse(BaseModel):
    message: str
    status: str
    metadata: dict


# Global variable to hold the RAG chain
current_rag_chain = None


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_rag_chain

    # Startup
    print("=" * 80)
    print("🇸🇱 Sierra Leone Legal Assistant API Starting...")
    print("=" * 80)

    # Load cached vectorstore (will be fast after first build)
    current_rag_chain = rag_chain
    print("✅ RAG system loaded successfully!")

    yield

    # Shutdown
    print("👋 Shutting down Sierra Leone Legal Assistant API...")


# Initialize FastAPI app
app = FastAPI(
    title="Sierra Leone Legal Assistant API",
    description="API for querying Sierra Leone laws and policies in simple, everyday language",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_source_path(source: str) -> str:
    """Clean up source path for better readability"""
    if 'data/pdfs/' in source:
        return source.replace('data/pdfs/', '').replace('.pdf', '')
    return source


def is_relevant_source(doc_content: str, question: str, min_relevance: float = 0.3) -> bool:
    """Check if source is relevant to the question"""
    question_keywords = set(question.lower().split())
    content_words = set(doc_content.lower().split())

    overlap = len(question_keywords & content_words)
    if len(question_keywords) == 0:
        return False

    relevance_score = overlap / len(question_keywords)
    return relevance_score >= min_relevance


def process_sources(source_documents, question):
    """Process and format source documents"""
    sources = []
    if source_documents:
        # Filter for relevant sources
        relevant_sources = [
            doc for doc in source_documents
            if is_relevant_source(doc.page_content, question)
        ]

        # If no relevant sources, use top 3
        docs_to_process = relevant_sources if relevant_sources else source_documents[:3]

        for doc in docs_to_process:
            metadata = doc.metadata
            source_name = clean_source_path(metadata.get('source', 'Unknown source'))
            page_num = metadata.get('page')

            # Create preview (first 250 characters)
            content_preview = doc.page_content[:250]
            if len(doc.page_content) > 250:
                content_preview += "..."

            sources.append({
                "source": source_name,
                "page": page_num + 1 if page_num is not None else None,
                "content_preview": content_preview
            })

    return sources


# async def stream_response(question: str):
#     """Stream the response word by word"""
#     try:
#         # Get the full response first (we need sources)
#         result = current_rag_chain({"query": question})
#         answer = result['result']
#         sources = process_sources(result.get('source_documents', []), question)
#
#         # Stream the answer word by word
#         words = answer.split()
#         for i, word in enumerate(words):
#             chunk_data = {
#                 "type": "answer_chunk",
#                 "content": word + " ",
#                 "done": False
#             }
#             yield f"data: {json.dumps(chunk_data)}\n\n"
#             await asyncio.sleep(0.05)  # Small delay for streaming effect
#
#         # Send sources at the end
#         final_data = {
#             "type": "complete",
#             "sources": sources,
#             "done": True
#         }
#         yield f"data: {json.dumps(final_data)}\n\n"
#
#     except Exception as e:
#         error_data = {
#             "type": "error",
#             "error": str(e),
#             "done": True
#         }
#         yield f"data: {json.dumps(error_data)}\n\n"

async def stream_response(question: str):
    """Stream the response token by token"""
    try:
        # Get relevant documents first
        retriever = current_rag_chain.retriever
        docs = retriever.get_relevant_documents(question)

        # Build context from documents
        context = "\n\n".join([doc.page_content for doc in docs[:3]])

        # Create the prompt
        prompt = f"""Based on the following context about Sierra Leone laws and policies, answer the question in simple, everyday language.

Context:
{context}

Question: {question}

Answer:"""

        # Stream from the LLM
        if hasattr(llm, 'stream'):
            # If your LLM supports streaming natively
            for chunk in llm.stream(prompt):
                if hasattr(chunk, 'content'):
                    token = chunk.content
                else:
                    token = str(chunk)

                chunk_data = {
                    "type": "answer_chunk",
                    "content": token,
                    "done": False
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0)
        else:
            # Fallback: generate full response then stream word by word
            result = current_rag_chain({"query": question})
            answer = result['result']

            # Stream word by word
            words = answer.split()
            for word in words:
                chunk_data = {
                    "type": "answer_chunk",
                    "content": word + " ",
                    "done": False
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.05)  # Small delay for visual effect

            docs = result.get('source_documents', [])

        # Send sources at the end
        sources = process_sources(docs, question)
        final_data = {
            "type": "complete",
            "sources": sources,
            "done": True
        }
        yield f"data: {json.dumps(final_data)}\n\n"

    except Exception as e:
        error_data = {
            "type": "error",
            "error": str(e),
            "done": True
        }
        yield f"data: {json.dumps(error_data)}\n\n"
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🇸🇱 Sierra Leone Legal Assistant API",
        "status": "running",
        "version": "1.0.0",
        "description": "Ask questions about Sierra Leone laws and policies in simple language",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/ask": "Ask a legal question (POST)",
            "/ask-stream": "Ask with streaming response (POST)",
            "/rebuild": "Rebuild vectorstore (POST)",
            "/status": "Get vectorstore status (GET)",
            "/docs": "Interactive API documentation",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sierra Leone Legal Assistant",
        "rag_system": "operational"
    }


@app.get("/status")
async def get_status():
    """Get vectorstore status and metadata"""
    try:
        metadata_file = "vectorstore_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return {
                "status": "ready",
                "vectorstore_exists": True,
                "metadata": metadata
            }
        else:
            return {
                "status": "not_initialized",
                "vectorstore_exists": False,
                "message": "Vectorstore not yet built"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild_vectorstore():
    """
    Force rebuild of the vectorstore from scratch.

    Use this endpoint when you:
    - Add new PDF files to data/pdfs/
    - Update the data/urls.txt file
    - Want to refresh all data

    ⚠️ Warning: This operation takes 1-2 minutes and will temporarily disrupt service.
    """
    global current_rag_chain

    try:
        print("\n" + "=" * 80)
        print("🔄 MANUAL REBUILD TRIGGERED")
        print("=" * 80)

        # Force rebuild
        new_vectorstore, new_embeddings = initialize_vectorstore(force_rebuild=True)

        # Update the RAG chain with new vectorstore
        current_rag_chain = create_rag_chain(new_vectorstore, llm)

        # Load metadata
        metadata_file = "vectorstore_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print("✅ Rebuild complete! RAG chain updated.")
        print("=" * 80 + "\n")

        return RebuildResponse(
            message="Vectorstore rebuilt successfully",
            status="success",
            metadata=metadata
        )

    except Exception as e:
        print(f"❌ Rebuild failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild vectorstore: {str(e)}"
        )


@app.post("/ask-stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a legal question with streaming response.

    The response will be streamed word-by-word for a better user experience.
    """
    print(f"\n🔍 Processing streaming question: {request.question}")

    return StreamingResponse(
        stream_response(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/ask", response_model=AnswerResponse, responses={500: {"model": ErrorResponse}})
async def ask_question(request: QuestionRequest):
    """
    Ask a legal question about Sierra Leone laws and policies.

    The AI will explain laws in simple, everyday language that anyone can understand,
    breaking down legal terms and using everyday examples when helpful.

    - **question**: Your legal question about Sierra Leone laws or policies
    """
    global current_rag_chain

    try:
        print(f"\n🔍 Processing question: {request.question}")

        # Query the RAG system
        result = current_rag_chain({"query": request.question})

        # Process source documents
        sources = process_sources(result.get('source_documents', []), request.question)

        print(f"✅ Answer generated with {len(sources)} source(s)")

        return AnswerResponse(
            question=request.question,
            answer=result['result'],
            sources=[SourceDocument(**s) for s in sources]
        )

    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Failed to process question: {str(e)}"
            ).dict()
        )


@app.get("/examples")
async def get_examples():
    """Get example questions you can ask"""
    return {
        "examples": [
            "What are my rights if I'm arrested?",
            "How do I register a business in Sierra Leone?",
            "What does the Constitution say about freedom of speech?",
            "What are the requirements for voting?",
            "What are the laws about property ownership?",
            "How does the court system work in Sierra Leone?",
            "What are my rights as a worker?",
            "How can I file a complaint against the police?"
        ],
        "tips": [
            "Ask questions in plain English",
            "Be specific about what you want to know",
            "You can ask follow-up questions for clarification"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 80)
    print("🚀 Starting Sierra Leone Legal Assistant API...")
    print("=" * 80)
    print("📖 Interactive docs: http://localhost:8000/docs")
    print("💡 Example questions: http://localhost:8000/examples")
    print("🏥 Health check: http://localhost:8000/health")
    print("🔄 Rebuild vectorstore: http://localhost:8000/rebuild (POST)")
    print("📊 Check status: http://localhost:8000/status")
    print("⚡ Streaming endpoint: http://localhost:8000/ask-stream (POST)")
    print("=" * 80 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )