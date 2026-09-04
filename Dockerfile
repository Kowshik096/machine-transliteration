# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the transliteration package itself (pyproject.toml is tracked).
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed site-packages and binaries from builder (system install).
COPY --from=builder /usr/local /usr/local

# Copy application source (data/checkpoints are NOT baked in; mount at runtime).
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py .
COPY convert_aksharantar.py .

# Create non-root user (no PATH hack needed: packages are system-wide).
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check without curl (python:3.10-slim has no curl by default).
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Default command.
# Mount data and checkpoints at runtime, e.g.:
#   docker run -p 8501:8501 -v ${PWD}/data:/app/data -v ${PWD}/attention_best.pt:/app/attention_best.pt:ro image
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]