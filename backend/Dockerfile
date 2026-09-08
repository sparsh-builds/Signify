FROM python:3.11-slim

WORKDIR /app

# Install native OpenCV and shared system libraries compatible with Debian Bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files directly into /app
COPY . .

# Create persistent data directory
RUN mkdir -p /app/data

EXPOSE 8000

# 2 workers to fit safely within the 512MB RAM budget
CMD ["uvicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", "app:app"]