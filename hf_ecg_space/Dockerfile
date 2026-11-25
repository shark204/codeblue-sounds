FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV / Pillow / git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Open-ECG-Digitizer (Ahus-AIM, CC-BY-4.0 / permissive)
RUN pip install --no-cache-dir git+https://github.com/Ahus-AIM/Open-ECG-Digitizer.git || true

COPY app.py .

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
