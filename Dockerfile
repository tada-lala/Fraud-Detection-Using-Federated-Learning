
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential libfreetype6-dev libpng-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


COPY . /opt/

EXPOSE 8000

WORKDIR /opt

ENTRYPOINT ["python", "app.py"]


