# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and models
COPY app.py .
COPY models/ models/
COPY templates/ templates/
COPY static/ static/

# Expose Flask port
EXPOSE 8000

# Run the app
CMD ["python", "app.py"]
