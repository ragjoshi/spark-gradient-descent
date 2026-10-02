# Image for the Streamlit app. Spark runs in local[N] mode inside the
# container, so the instance's vCPUs are the cores being benchmarked.
FROM python:3.12-slim-trixie

# Spark 4 needs Java 17 or 21; procps provides `ps`, which Spark's launch
# scripts call.
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-21-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
# Run from /app so Streamlit picks up .streamlit/config.toml (1 GB uploads).
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
