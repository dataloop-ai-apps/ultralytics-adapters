FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

USER root

# Create directory and set ownership in one step
RUN mkdir -p /tmp/app && chown 1000:1000 /tmp/app
RUN mkdir -p /tmp/app/weights && chown 1000:1000 /tmp/app/weights

USER 1000

# Download weights
RUN wget -O /tmp/app/weights/yolov11m-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt
RUN wget -O /tmp/app/weights/yolov11m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
RUN wget -O /tmp/app/weights/yolov10m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt
RUN wget -O /tmp/app/weights/yolov9c-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c-seg.pt
RUN wget -O /tmp/app/weights/yolov9c.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt
RUN wget -O /tmp/app/weights/yolov8m-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt
RUN wget -O /tmp/app/weights/yolov8m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt

# Install packages
RUN pip install --user \
    ultralytics \
    pyyaml \
    git+https://github.com/dataloop-ai-apps/dtlpy-converters.git \

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/yolov9:0.0.20 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/yolov9:0.0.20 bash