import streamlit as st
import os
import requests
import google.generativeai as genai

import time
import base64
# Streamlitアプリケーションのタイトル設定
st.title("Chat TTS DEMO")

# 環境変数からAPIキーを読み込む
YOUR_GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
# or
# YOUR_GOOGLE_AI_STUDIO_API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# APIリクエストのURL
url = "http://gemini-openai-proxy:8080/v1/chat/completions"

genai.configure(api_key=YOUR_GOOGLE_AI_STUDIO_API_KEY)
model_genai = genai.GenerativeModel('gemini-1.0-pro-latest')

SYSTEM_PROMPT = """
あなたは猫の国の猫男子「ねこた」です。
「ねこた」になりきって応答して

好奇心旺盛です
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

# 既存のメッセージをチャットウィンドウに表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Style-BERT APIを使用して音声を生成する関数
def generate_voice(text):
    # Style-BERT APIのベースURL
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

# ユーザーからの新しい入力を受け取る
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # レスポンスからアシスタントのメッセージを取得
    msg = f"""
    SYSTEM PROMPT: {SYSTEM_PROMPT}

    USER PROMPT: {prompt}
    
    """
    assistant_message = model_genai.generate_content(msg)
    st.session_state.messages.append({"role": "assistant", "content": assistant_message.text})
    with st.chat_message("assistant"):
        st.markdown(assistant_message.text)

        # 音声を生成し、Streamlitアプリで再生
        voice_data = generate_voice(assistant_message.text)
        if voice_data:
            # 一時ファイルに音声を保存
            voice_file = "temp_voice_output.wav"
            with open(voice_file, "wb") as f:
                f.write(voice_data)

            audio_placeholder = st.empty()
            with open(voice_file, "rb") as f:
                contents = f.read()

            audio_str = "data:audio/ogg;base64,%s"%(base64.b64encode(contents).decode())
            audio_html = """
                            <audio autoplay=True>
                            <source src="%s" type="audio/ogg" autoplay=True>
                            Your browser does not support the audio element.
                            </audio>
                        """ %audio_str
            time.sleep(0.5) #これがないと上手く再生されません
            audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

            # 音声ファイルをアプリに埋め込み、再生
            st.audio(voice_file)

            # while mixer.music.get_busy():  # ファイルの再生が完了するのを待つ
            #     await asyncio.sleep(1)

# 今日の気分はどう？