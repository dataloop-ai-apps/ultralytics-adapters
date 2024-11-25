FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

RUN pip install --user \
    ultralytics==8.2.22 \
    pyyaml