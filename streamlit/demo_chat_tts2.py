import streamlit as st
import sys
import os

# 現在のファイルのディレクトリパスを取得
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
import pprint

pprint.pprint(sys.path)

from modules.chat_utils import display_messages, add_user_input
from modules.voice_generator import generate_voice_and_play
from modules.genai_model import get_assistant_message
from modules.config import YOUR_GOOGLE_AI_STUDIO_API_KEY

# アプリケーションのタイトル設定
st.title("Chat TTS DEMO")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 既存のメッセージをチャットウィンドウに表示
display_messages(st.session_state.messages)

# ユーザーからの新しい入力を受け取る
if prompt := st.chat_input("What is up?"):
    add_user_input(st.session_state.messages, prompt)
    assistant_message = get_assistant_message(prompt)

    with st.chat_message("assistant"):
        st.markdown(assistant_message)

    voice_file = generate_voice_and_play(assistant_message)
    
    # 音声ファイルをアプリに埋め込み、再生
    st.audio(voice_file)
# 今日の気分はどう？