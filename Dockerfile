FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --root-user-action=ignore -r requirements.txt

COPY handler.py .

EXPOSE 8000
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000"]
