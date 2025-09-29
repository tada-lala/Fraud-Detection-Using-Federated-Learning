# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

# system deps for compiling some python packages (keeps image small-ish)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential libfreetype6-dev libpng-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# copy app code and data
COPY . .

EXPOSE 8000

# run gunicorn (app:app must match your Flask file)
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8000", "app:app"]


