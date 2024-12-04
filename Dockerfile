FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

USER 1000

# Create directory and set ownership in one step
RUN mkdir -p /tmp/app && chown 1000:1000 /tmp/app
RUN mkdir -p /tmp/app/weights && chown 1000:1000 /tmp/app/weights

RUN wget -O /tmp/app/weights/yolov9c-seg.pt https://github.com/ultralytics/assets/releases/tag/v8.3.0/yolov11m-seg.pt
RUN wget -O /tmp/app/weights/yolov9c.pt https://github.com/ultralytics/assets/releases/tag/v8.3.0/yolov11m.pt
RUN wget -O /tmp/app/weights/yolov9c-seg.pt https://github.com/ultralytics/assets/releases/tag/v8.3.0/yolov9c-seg.pt
RUN wget -O /tmp/app/weights/yolov9c.pt https://github.com/ultralytics/assets/releases/tag/v8.3.0/yolov9c.pt
RUN wget -O /tmp/app/weights/yolov9c-seg.pt https://github.com/ultralytics/assets/releases/tag/v8.3.0/yolov8m-seg.pt
RUN wget -O /tmp/app/weights/yolov9c.pt https://github.com/ultralytics/assets/releases/tag/v8.3.0/yolov8m.pt

RUN pip install --user \
    ultralytics \
    pyyaml \
    git+https://github.com/dataloop-ai-apps/dtlpy-converters.git

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/yolov9:0.0.20 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/yolov9:0.0.20 bash