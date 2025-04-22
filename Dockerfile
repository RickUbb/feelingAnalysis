# Tests
FROM python:3.12.3-slim

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

RUN pytest --disable-pytest-warnings

# Imagen
FROM python:3.12.3-slim 

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["/bin/bash", "-c", "cd /app && uvicorn api:app --host 0.0.0.0 --port 8000"]

