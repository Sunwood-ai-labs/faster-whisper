import requests
import streamlit as st
import base64
import time

def generate_voice(text):
    """Style-BERT APIを使用して音声を生成する関数"""
    voice_url = "http://style-bert:5000/voice"
    params = {
        "text": text,
        "encoding": "utf-8",
        "model_id": 0,
        "speaker_id": 0,
        "sdp_ratio": 0.2,
        "noise": 0.6,
        "noisew": 0.8,
        "length": 1,
        "language": "JP",
        "auto_split": True,
        "split_interval": 0.5,
        "assist_text": "",
        "assist_text_weight": 1,
        "style": "Neutral",
        "style_weight": 5
    }
    response = requests.get(voice_url, params=params)
    if response.status_code == 200:
        return response.content
    else:
        print("音声生成エラー:", response.status_code)
        return None

def generate_voice_and_play(text):
    """テキストから音声を生成し、Streamlitアプリで再生する"""
    voice_data = generate_voice(text)
    if voice_data:
        # 一時ファイルに音声を保存
        voice_file = "temp_voice_output.wav"
        with open(voice_file, "wb") as f:
            f.write(voice_data)

        # Streamlitアプリで音声を再生
        audio_str = "data:audio/ogg;base64,%s" % (base64.b64encode(voice_data).decode())
        audio_html = f"""
                        <audio autoplay=True>
                        <source src="{audio_str}" type="audio/ogg">
                        Your browser does not support the audio element.
                        </audio>
                      """
        time.sleep(0.5)  # 音声が上手く再生されるように少し待機
        st.markdown(audio_html, unsafe_allow_html=True)

        return voice_file
