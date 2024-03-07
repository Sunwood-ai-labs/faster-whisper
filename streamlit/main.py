import streamlit as st
from audio_recorder_streamlit import audio_recorder
from datetime import datetime
from pathlib import Path
import os

from faster_whisper import WhisperModel
import time
import numpy as np

import base64
import requests
import pprint
import json

import google.generativeai as genai

model_size = "large-v3"  # モデルのサイズを指定
model = WhisperModel(model_size, device="cuda", compute_type="float16")  # ここでは例としてFP16を使用

GOOGLE_API_KEY= "AIzaSyCbRltHDKdUCZZ3p7rNkFoifmlT6aIdouM"
genai.configure(api_key=GOOGLE_API_KEY)
model_genai = genai.GenerativeModel('gemini-pro')

def save_audio_with_date(audio_bytes, folder_path="streamlit/saved_audios"):
    # 現在の日時をファイル名に使用するためのフォーマット
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"audio_{date_str}.wav"
    
    # 指定したフォルダパスをPathオブジェクトに変換
    folder_path = Path(folder_path)
    
    # フォルダが存在しない場合に作成
    if not folder_path.exists():
        os.makedirs(folder_path)
    
    # ファイルのフルパスを生成
    full_path = folder_path / file_name
    
    # 音声ファイルを保存
    with open(full_path, "wb") as file:
        file.write(audio_bytes)

    # 計測開始時刻
    start_time = time.time()


    with st.chat_message("user", avatar="🦖"):
        st.write(f"full_path: {full_path}")
        # 録音した音声ファイルを書き起こし
        segments, info = model.transcribe(str(full_path), beam_size=5)

        # 計測終了時刻と実行時間の計算、出力
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"文字お越し時間: {elapsed_time}秒")

        # 検出された言語とその確率、書き起こされたテキストを表示
        st.write("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            st.write("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


    with st.chat_message("assistant", avatar="🤖"):
        start_time = time.time()
        response = model_genai.generate_content(f"{segment.text}")
        st.write(response.text)
        # 計測終了時刻と実行時間の計算、出力
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"回答時間: {elapsed_time}秒")


    return full_path  # 保存したファイルのパスを返す

# Example usage with Streamlit:
def main():
    st.title("Voice to Text Transcription 2")
    
    # Record audio using Streamlit widget
    audio_bytes = audio_recorder(pause_threshold=30)
    
    # 音声を指定のフォルダに日付情報を付与して保存
    if audio_bytes:
        saved_file_path = save_audio_with_date(audio_bytes, "streamlit/saved_audios")
        st.write(f"Audio saved to: {saved_file_path}")

if __name__ == "__main__":
    main()
