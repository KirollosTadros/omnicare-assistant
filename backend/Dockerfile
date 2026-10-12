FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache

COPY requirements.txt .
RUN --mount=type=bind,from=pipcache,target=/root/.cache/pip,rw \
    pip install -r requirements.txt torch==2.13.0

# bake the embedding model into the image so the first request is fast and works offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
