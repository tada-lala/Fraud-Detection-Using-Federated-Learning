FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV MPLBACKEND Agg

WORKDIR /app

# Cache dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
