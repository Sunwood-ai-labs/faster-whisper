import streamlit as st
import os
import requests
import google.generativeai as genai

# Streamlitアプリケーションのタイトル設定
st.title("ChatGPT-like clone")

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
