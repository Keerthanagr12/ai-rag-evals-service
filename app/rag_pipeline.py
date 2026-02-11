"""RAG Pipeline - Core retrieval-augmented generation orchestration.

This module implements the end-to-end RAG workflow:
- Query processing and embedding
- Semantic retrieval from vector store
- Context assembly and filtering
- LLM generation with retrieved context
- Response logging and tracing

Design philosophy:
- Type safety: Full type hints for all functions
- Observability: Structured logging at each stage
- Testability: Dependency injection for all external services
- Performance: Async operations where beneficial
- Trade-offs: Balanced between latency, cost, and accuracy
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


# Type definitions for external dependencies (enables mocking)
class VectorStore(Protocol):
    """Interface for vector database operations."""

    def query(
        self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        ...


class EmbeddingModel(Protocol):
    """Interface for embedding generation."""

    def embed(self, text: str) -> List[float]:
        ...


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline behavior.

    Trade-offs documented for each parameter:
    - chunk_overlap: Higher = better context continuity, more storage
    - retrieval_k: Higher = more context, higher latency & cost
    - temperature: Lower = more deterministic, less creative
    - max_tokens: Higher = more complete answers, higher cost
    """

    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.3  # Low temp for factual accuracy
    max_tokens: int = 500
    retrieval_k: int = 5
    chunk_overlap: int = 100
    system_prompt: str = (
        "You are a helpful AI assistant. Answer based ONLY on the provided context. "
        "If the answer is not in the context, say so explicitly."
    )


@dataclass
class RetrievedDocument:
    """Represents a document retrieved from vector store."""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Structured response from RAG pipeline."""

    query: str
    answer: str
    sources: List[RetrievedDocument]
    latency_ms: float
    model: str
    tokens_used: Optional[int] = None


class RAGPipeline:
    """End-to-end RAG pipeline with observability and error handling.

    Example usage:
        >>> pipeline = RAGPipeline(vector_store, embedding_model, config)
        >>> response = await pipeline.query("What is RAG?")
        >>> print(response.answer)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        config: Optional[RAGConfig] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config or RAGConfig()
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None

        logger.info(
            "RAG pipeline initialized",
            extra={
                "model": self.config.model_name,
                "retrieval_k": self.config.retrieval_k,
                "temperature": self.config.temperature,
            },
        )

    def query(
        self, query: str, metadata_filters: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """Execute full RAG pipeline for a user query.

        Args:
            query: User's natural language question
            metadata_filters: Optional filters for vector search (e.g., {"source": "docs"})

        Returns:
            RAGResponse with answer, sources, and metadata

        Raises:
            ValueError: If query is empty
            RuntimeError: If retrieval or generation fails
        """
        start_time = time.time()

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(
            "RAG query started",
            extra={"query": query, "filters": metadata_filters},
        )

        try:
            # Step 1: Embed the query
            query_embedding = self._embed_query(query)

            # Step 2: Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(
                query_embedding, metadata_filters
            )

            if not retrieved_docs:
                logger.warning("No documents retrieved for query", extra={"query": query})
                return RAGResponse(
                    query=query,
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    model=self.config.model_name,
                )

            # Step 3: Assemble context from retrieved docs
            context = self._build_context(retrieved_docs)

            # Step 4: Generate answer using LLM
            answer, tokens_used = self._generate_answer(query, context)

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "RAG query completed",
                extra={
                    "query": query,
                    "num_sources": len(retrieved_docs),
                    "latency_ms": latency_ms,
                    "tokens_used": tokens_used,
                },
            )

            return RAGResponse(
                query=query,
                answer=answer,
                sources=retrieved_docs,
                latency_ms=latency_ms,
                model=self.config.model_name,
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.error(
                "RAG query failed",
                extra={"query": query, "error": str(e)},
                exc_info=True,
            )
            raise RuntimeError(f"RAG pipeline failed: {e}") from e

    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for user query.

        Trade-off: Using same embedding model as indexing ensures
        semantic consistency. Different models may yield poor retrieval.
        """
        try:
            embedding = self.embedding_model.embed(query)
            logger.debug("Query embedded", extra={"embedding_dim": len(embedding)})
            return embedding
        except Exception as e:
            logger.error("Embedding generation failed", extra={"error": str(e)})
            raise RuntimeError(f"Failed to embed query: {e}") from e

    def _retrieve_documents(
        self,
        query_embedding: List[float],
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """Retrieve top-k most relevant documents from vector store.

        Trade-off: Higher k = more context for LLM but higher cost and
        potential noise. Lower k = faster but may miss relevant info.
        """
        try:
            raw_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=self.config.retrieval_k,
                filters=metadata_filters,
            )

            retrieved_docs = [
                RetrievedDocument(
                    doc_id=doc["id"],
                    content=doc["content"],
                    score=doc["score"],
                    metadata=doc.get("metadata", {}),
                )
                for doc in raw_results
            ]

            logger.debug(
                "Documents retrieved",
                extra={
                    "count": len(retrieved_docs),
                    "top_score": retrieved_docs[0].score if retrieved_docs else None,
                },
            )

            return retrieved_docs

        except Exception as e:
            logger.error("Document retrieval failed", extra={"error": str(e)})
            raise RuntimeError(f"Failed to retrieve documents: {e}") from e

    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        """Assemble context string from retrieved documents.

        Trade-off: Including doc IDs helps with source attribution
        but uses extra tokens. Format optimized for token efficiency.
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Document {i}] (ID: {doc.doc_id}, Score: {doc.score:.3f})\n{doc.content}"
            )

        context = "\n\n".join(context_parts)
        logger.debug(
            "Context assembled",
            extra={
                "num_docs": len(documents),
                "context_length": len(context),
            },
        )
        return context

    def _generate_answer(self, query: str, context: str) -> tuple[str, Optional[int]]:
        """Generate answer using LLM with retrieved context.

        Trade-off: Lower temperature = more deterministic, better for
        factual Q&A. Higher temperature = more creative, better for
        open-ended generation.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        try:
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ]

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            answer = response.choices[0].message.content or "(No answer generated)"
            tokens_used = response.usage.total_tokens if response.usage else None

            logger.debug(
                "Answer generated",
                extra={
                    "answer_length": len(answer),
                    "tokens_used": tokens_used,
                },
            )

            return answer, tokens_used

        except Exception as e:
            logger.error("Answer generation failed", extra={"error": str(e)})
            raise RuntimeError(f"Failed to generate answer: {e}") from e


def create_rag_pipeline(
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    openai_api_key: str,
    config: Optional[RAGConfig] = None,
) -> RAGPipeline:
    """Factory function for creating RAG pipeline instances.

    Useful for dependency injection in tests and applications.

    Args:
        vector_store: Initialized vector database client
        embedding_model: Initialized embedding model
        openai_api_key: OpenAI API key for generation
        config: Optional custom configuration

    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(
        vector_store=vector_store,
        embedding_model=embedding_model,
        config=config,
        openai_api_key=openai_api_key,
    )
