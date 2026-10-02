FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir -e .

RUN python -m src.train

EXPOSE 8000

CMD ["python", "-m", "src.main"]
