# Wybór lekkiego obrazu bazowego Pythona
FROM python:3.10-slim

# Ustawienie katalogu roboczego wewnątrz kontenera
WORKDIR /app

# Instalacja zależności systemowych niezbędnych do przetwarzania obrazów
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie pliku wymagań i instalacja bibliotek (w tym onnx i onnxruntime)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    multipart \
    python-multipart \
    onnxruntime \
    onnx \
    pillow \
    numpy

# Kopiowanie kodu źródłowego aplikacji oraz modelu ONNX do kontenera
COPY app.py .
COPY food_classifier_20.onnx .
COPY food_classifier_20.onnx.data .

# Otwarcie portu serwera FastAPI
EXPOSE 8000

# Komenda uruchamiająca serwer uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]