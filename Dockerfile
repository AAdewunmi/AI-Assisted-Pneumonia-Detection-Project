FROM python:3.11-slim

# Install deps required by OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Ensure the model directory is included
COPY models/ models/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose for Render
ENV PORT=5000
EXPOSE 5000

CMD ["python", "app/app.py"]
