
<h1>
<img src="https://raw.githubusercontent.com/Sunwood-ai-labs/faster-whisper-docker/master/docs/fast_icon.png" height=200px align="left"/>
faster-whisper-docker <br>
</h1>

**Faster Whisper** はOpenAIのWhisperモデルを再実装したもので、CTranslate2を使用して高速に音声認識を行います。このガイドでは、Dockerを使用してFaster Whisperを簡単に設定し、実行する方法を紹介します。

CTranslate2を使用したFaster Whisperについては[こちら](https://github.com/Sunwood-ai-labs/faster-whisper-docker/blob/master/README_fast_JP.md)




## 前提条件
- Dockerがインストールされていること
- NVIDIA GPUを搭載したマシン（GPUを使用する場合）
## セットアップ

### 1. `docker-compose.yml`の準備

Docker Composeを使用して、Faster Whisperを含む環境を構築します。以下の内容で`docker-compose.yml`ファイルを作成してください。

```yaml
services:
  app:
    build: .
    volumes:
      - ./:/app
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
              
    tty: true
```


### 2. `Dockerfile`の作成

以下の内容で`Dockerfile`を作成します。このファイルは、Faster Whisperとその依存関係をインストールするために使用されます。

```Dockerfile
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04
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
        portaudio19-dev \
    && pip install pip -U

RUN pip install faster-whisper
```


### 3. コンテナのビルドと起動

`docker-compose.yml`と`Dockerfile`を準備したら、以下のコマンドでコンテナをビルドし、起動します。

```bash
docker-compose up
```


## デモの実行

Faster Whisperを使って音声ファイルをテキストに変換するデモを実行してみましょう。以下のPythonスクリプトを`demo01_whisper_audio_transcription.py`として保存します。

```python
from faster_whisper import WhisperModel

model_size = "large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe("audio/Word2Motion/001/02_nodding.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```



コンテナ内でこのスクリプトを実行するには、以下のコマンドでコンテナのシェルにアクセスし、スクリプトを実行します。

```bash
docker-compose exec app /bin/bash
python demo/demo01_whisper_audio_transcription.py
```











