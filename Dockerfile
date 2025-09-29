# Use official Python image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (if your app.py runs a web server, e.g., Flask on 5000)
EXPOSE 5000

# Default command to run the app
CMD ["python", "app.py"]


