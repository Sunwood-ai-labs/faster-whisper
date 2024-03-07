import google.generativeai as genai
import os

GOOGLE_API_KEY= os.getenv("GOOGLE_AI_STUDIO_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():

    print(m.name)

model_genai = genai.GenerativeModel('gemini-1.0-pro-latest')


response = model_genai.generate_content("今日の気分はどう？?")

print(response.text)