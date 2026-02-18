FROM python:3.11-slim

WORKDIR /app

# Install certificates (fix SSL issues)
RUN apt-get update && apt-get install -y ca-certificates

# Copy only requirements first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy only needed files
COPY main.py .
COPY test.py .
COPY config.yaml .

# Copy model folder (after dvc pull)
COPY output/ output/

# Default command
CMD ["python", "test.py"]
