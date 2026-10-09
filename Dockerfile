FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY runbooks ./runbooks
COPY data ./data
RUN pip install --no-cache-dir .

ENV AIOPS_RUNBOOKS=/app/runbooks
EXPOSE 8000
CMD ["uvicorn", "aiops.api:app", "--host", "0.0.0.0", "--port", "8000"]
