FROM python:3.11-slim

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple cython "numpy<1.24" setuptools wheel && \
    sed -i '/madmom/d' requirements.txt && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation madmom==0.16.* && \
    find /usr/local/lib/python*/site-packages/madmom -name '*.py' \
        -exec sed -i 's/from collections import Mutable/from collections.abc import Mutable/g' {} + && \
    python -c "from madmom.features.key import CNNKeyRecognitionProcessor; print('madmom OK')"

COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
