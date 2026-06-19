import logging # This is for logging - helps in tracking events that happen when some software runs.
from fastapi import FastAPI # FastAPI is a modern web framework for building APIs with Python 3.6+ based on standard Python type hints.
import inngest # Inngest is a service for building event-driven applications.
import inngest.fast_api # This module integrates Inngest with FastAPI.
from inngest.experimental import ai # This module provides experimental AI features in Inngest.
from dotenv import load_dotenv # This module loads environment variables from a .env file into the system's environment variables. 
import uuid # This module provides immutable UUID objects (universally unique identifiers) and the functions to generate them.
import os # This module provides a way of using operating system dependent functionality like reading or writing to the file system.
import datetime # This module supplies classes for manipulating dates and times.
from google import genai # This is the Google Gemini API client library for Python, used to interact with Google's Gemini AI models.
from data_loader import load_and_chunk_pdf, embed_texts # Importing functions from the data_loader module to load and chunk PDF files and to embed texts.
from vector_db import QdrantStorage # Importing the QdrantStorage class from the vector_db module to interact with the Qdrant vector search engine.
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult # Importing custom data models from the custom_types module for structured data handling in the application.

load_dotenv() # Load environment variables from a .env file into the system's environment variables.


inngest_client = inngest.Inngest(
    app_id = "rag_app",
    logger = logging.getLogger("uvicorn"),
    is_production = False,
    event_key = os.getenv("INNGEST_API_KEY"),
    event_api_base_url = "https://api.inngest.com",
    serializer = inngest.PydanticSerializer(),

)


@inngest_client.create_function(
    fn_id="Rag: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
    
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}-{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)

    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Search PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)

async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        results = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=results["Contexts"], sources=results["Sources"])
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"\nContext: {context_block}\n\n"
        f"Question: {question}\n"
        "answer consisely using the context above"
    )
    gemeni_api = os.getenv("GEMENI_API")
    gemeni_client = genai.Client(api_key=gemeni_api)
    res = await ctx.step.run(
        "generate-answer",
        lambda: gemeni_client.models.generate_content(
            model="gemini-flash-latest",
            contents=user_content,
    
        ).text
    )
    return {"answer": res, "sources":found.sources, "num_contexts": len(found.contexts)}



app = FastAPI() # Create an instance of the FastAPI class. This instance will be our WSGI application.

inngest.fast_api.serve(app, inngest_client, [ingest_pdf, rag_query_pdf_ai], serve_path="/api/inngest") # Serve the FastAPI application with Inngest integration.