# AI RAG Evals Service

> Production-grade RAG service with FastAPI, typed Python, automated evals, and CI/CD. Ships real AI applications with reliability focus: semantic search, context engineering, regression gates, and observability.

## Features

- **RAG Pipeline**: End-to-end retrieval-augmented generation with semantic search, context engineering, and metadata filtering
- **Automated Evals**: YAML-based test cases with LLM judge and keyword metrics, regression gates prevent quality degradation
- **Production-Ready**: Typed Python code, FastAPI, Docker, GitHub Actions CI, structured logging, OpenTelemetry stubs
- **Developer Experience**: Clear separation of concerns, testable modules, meaningful error messages, comprehensive docs
- **Observability**: Query tracking, retrieved documents logging, latency measurements, response quality metrics

## Architecture

```
┌─────────────────┐
│  FastAPI Server │
└────────┬────────┘
         │
    ┌────▼────────────────┐
    │ RAG Pipeline        │
    │ - Query Processor   │
    │ - Retriever        │
    │ - Ranker (optional) │
    │ - Generator        │
    └────┬───────────────┘
         │
    ┌────▼────────────────┐     ┌──────────────┐
    │ Vector Store        │◄───┤ LLM (OpenAI) │
    │ (Chroma/Qdrant)     │     └──────────────┘
    └────┬───────────────┘
         │
    ┌────▼──────────────────┐
    │ Evals Framework       │
    │ - Test Runner        │
    │ - Metrics Calculator │
    │ - Regression Gates   │
    └──────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (optional)
- OpenAI API key

### Local Setup

```bash
# Clone repository
git clone https://github.com/Keerthanagr12/ai-rag-evals-service.git
cd ai-rag-evals-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your OpenAI API key

# Run server
uvicorn app.main:app --reload
```

### Docker Setup

```bash
# Build and run with docker-compose
docker-compose up --build

# Server will be at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic data models
│   ├── rag_pipeline.py         # Core RAG orchestration
│   ├── store.py                # Vector database client
│   ├── evaluator.py            # Evaluation utilities
│   └── logging_config.py        # Structured logging
├── tests/
│   ├── test_rag_pipeline.py    # RAG pipeline tests
│   ├── test_api.py             # API endpoint tests
│   └── conftest.py             # Pytest fixtures
├── evals/
│   ├── cases.yaml              # Evaluation test cases
│   ├── run_evals.py            # Evaluation runner CLI
│   └── metrics.py              # Metric calculation
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## API Endpoints

### Query RAG Pipeline

```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "What is retrieval-augmented generation?",
  "top_k": 5,
  "metadata_filter": {"source": "docs"}
}

Response:
{
  "query": "What is retrieval-augmented generation?",
  "answer": "RAG combines...",
  "sources": [
    {"doc_id": "doc_1", "content": "...", "score": 0.95}
  ],
  "latency_ms": 245,
  "model": "gpt-3.5-turbo"
}
```

### Ingest Documents

```http
POST /api/v1/ingest
Content-Type: application/json

{
  "documents": [
    {"id": "doc_1", "content": "...", "metadata": {"source": "docs"}}
  ]
}

Response:
{
  "ingested": 1,
  "embeddings_created": 1
}
```

### Run Evaluations

```http
GET /api/v1/evals/run
Query params: mode=full|quick

Response:
{
  "pass_rate": 0.85,
  "total_tests": 20,
  "passed": 17,
  "failed": 3,
  "timestamp": "2025-02-11T..."
}
```

## Evaluation Setup

### Test Cases Format (YAML)

```yaml
tests:
  - id: test_001
    query: "What is RAG?"
    expected_answer: "Retrieval-augmented generation combines..."
    category: "fundamentals"
    metadata:
      difficulty: "easy"

  - id: test_002
    query: "How does RAG improve accuracy?"
    expected_answer: "RAG grounds LLM responses in retrieved documents..."
    category: "advanced"
    metadata:
      difficulty: "medium"
```

### Running Evals

```bash
# Run full evaluation suite
python -m evals.run_evals --mode=full

# Run quick evaluation (subset)
python -m evals.run_evals --mode=quick

# Export results
python -m evals.run_evals --mode=full --output=json > results.json
```

### Eval Metrics

- **Exact Match**: String comparison against expected answer
- **F1 Score**: Token-level similarity (retrieval-focused)
- **LLM Judge**: Separate LLM-graded semantic correctness with deterministic config
- **Latency**: End-to-end response time tracking
- **Regression Gate**: CI fails if pass-rate drops below configured threshold (default: 80%)

## CI/CD Pipeline

### GitHub Actions Workflow

Automatically triggered on:
- `push` to main branch
- `pull_request` to main branch

**Jobs:**
1. **Lint & Format**: Python type checking, code style
2. **Unit Tests**: `pytest` with coverage reporting
3. **Integration Tests**: API endpoint testing
4. **Evals**: Run evaluation suite with regression gates
5. **Build**: Docker image build and push (on main)

Failure triggers:
- Test failures
- Eval pass-rate < 80%
- Coverage drops below threshold

## Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=6333
MODEL_NAME=gpt-3.5-turbo
LOG_LEVEL=INFO
EVAL_PASS_RATE_THRESHOLD=0.80
```

### RAG Pipeline Configuration

```python
# app/config.py
from dataclasses import dataclass

@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 100
    retrieval_k: int = 5
    temperature: float = 0.3
    system_prompt: str = "You are a helpful AI..."
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_rag_pipeline.py

# With coverage
pytest --cov=app --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy app/

# Linting
flake8 app/ tests/

# Formatting
black app/ tests/
```

### Adding a New RAG Component

1. Create new module in `app/`
2. Add type hints and docstrings
3. Write unit tests in `tests/`
4. Update evaluation cases if needed
5. Add to CI workflow if necessary

## Performance Considerations

### Latency Tradeoffs

| Component | Latency | Optimization |
|-----------|---------|---------------|
| Vector Search | 50-100ms | Index tuning, batch queries |
| LLM Generation | 500-2000ms | Model choice, prompt optimization |
| Reranking | 100-300ms | Optional, threshold-based |
| Total E2E | 650-2400ms | Caching, async processing |

### Cost Optimization

- **Batch ingestion**: Process documents in bulk
- **Semantic caching**: Cache frequent queries
- **Model selection**: Use gpt-3.5-turbo over gpt-4 for most use cases
- **Context length**: Optimize token usage in prompts

## Deployment

### Cloud Deployment (AWS Example)

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build -t ai-rag-evals-service .
docker tag ai-rag-evals-service:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ai-rag-evals-service:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ai-rag-evals-service:latest

# Deploy to ECS/EKS
kubectl apply -f k8s/deployment.yaml
```

### Environment-Specific Configs

- **Dev**: SQLite vector DB, mock LLM, verbose logging
- **Staging**: Postgres pgvector, OpenAI API, moderate logging
- **Production**: Managed vector DB (Pinecone/Qdrant), OpenAI API, monitored logging

## Observability

### Logging

```python
# Structured logging example
logger.info("rag_query_started", extra={
    "query": query,
    "user_id": user_id,
    "timestamp": datetime.now().isoformat()
})
```

### Metrics

- Query latency (p50, p95, p99)
- Eval pass-rate per category
- API error rates
- Vector DB query time
- LLM token usage

### Tracing (OpenTelemetry)

```python
# Minimal OpenTelemetry integration
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("rag_query") as span:
    span.set_attribute("query", query)
    # ... RAG pipeline ...
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with type hints and tests
4. Push: `git push origin feature/my-feature`
5. Open pull request with clear description

### Code Standards

- Python 3.9+ with full type hints
- Tests for all new features (>85% coverage)
- Meaningful commit messages
- No commits larger than 500 lines
- CI must pass before merge

## Hardest Bug Solved

**Issue**: Inconsistent eval results due to LLM non-determinism

**Root Cause**: LLM-judge calls without seed/temperature control produced different outputs on identical inputs

**Solution**:
1. Isolated LLM-judge to separate component with fixed config (temperature=0.0, seed)
2. Added "mock mode" for deterministic runs in CI
3. Tightened judge prompt to reduce variance
4. Logged all judge calls for reproducibility
5. Created regression detection: alert if eval changes >5% between runs

**Result**: Deterministic evals, reproducible CI failures, <1% variance

## Stack

- **Language**: Python 3.9+ (fully typed)
- **Framework**: FastAPI
- **Vector DB**: Chroma (local) / Qdrant (production)
- **LLM**: OpenAI API (gpt-3.5-turbo, gpt-4)
- **Testing**: pytest, pytest-asyncio
- **CI/CD**: GitHub Actions
- **Containerization**: Docker, Docker Compose
- **Observability**: Python logging, OpenTelemetry stubs
- **Data Validation**: Pydantic

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration: [Your Email]
