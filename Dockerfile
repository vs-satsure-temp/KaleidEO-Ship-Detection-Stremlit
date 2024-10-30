FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY . /app

COPY models/yolov8.pt /app/yolov8.pt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir streamlit pillow omegaconf

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
