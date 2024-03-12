FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        gcc \
        curl \
        wget \
        sudo \
        pciutils \
        python3-all-dev \
        python-is-python3 \
        python3-pip \
        ffmpeg \
        libsdl2-dev \
        pulseaudio \
        alsa-utils \
        portaudio19-dev \
    && pip install pip -U

RUN pip install faster-whisper \
                streamlit \
                audio-recorder-streamlit \
                fastapi uvicorn python-multipart \
                google-generativeai 
# streamlit run streamlit/main.py --server.port 8502  --server.address 0.0.0.0
# streamlit run streamlit/test.py --server.port 8502  --server.address 0.0.0.0