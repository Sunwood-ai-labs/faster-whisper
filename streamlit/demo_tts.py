import streamlit as st
from datetime import datetime
from pathlib import Path
import os

import time
import numpy as np

import base64
import requests
import pprint
import json

import google.generativeai as genai


# Example usage with Streamlit:
def main():
    st.title("TTS DEMO")

    prompt = st.chat_input("Say something")

    with st.chat_message("assistant", avatar="🤖"):
        
        if prompt:
            st.write(f"User has sent the following prompt: {prompt}")

if __name__ == "__main__":
    main()