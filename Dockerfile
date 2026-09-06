FROM python:3.11-slim

WORKDIR /app

# Install native OpenCV and image processing shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY backend/ ./backend/

# Create persistent data directory
RUN mkdir -p /app/data

EXPOSE 8000

# Run with Gunicorn process manager (4 Uvicorn workers)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "--chdir", "backend", "app:app"]