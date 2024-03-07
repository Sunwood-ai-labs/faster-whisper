import google.generativeai as genai
from config import YOUR_GOOGLE_AI_STUDIO_API_KEY

# Google Generative AIのAPIキーとモデル設定
genai.configure(api_key=YOUR_GOOGLE_AI_STUDIO_API_KEY)
model_genai = genai.GenerativeModel('gemini-1.0-pro-latest')

def get_assistant_message(prompt):
    """ユーザープロンプトに基づいてアシスタントのメッセージを生成する"""
    SYSTEM_PROMPT = """
    あなたは猫の国の猫男子「ねこた」です。
    「ねこた」になりきって応答して

    好奇心旺盛です
    """
    
    # プロンプトのフォーマットを設定
    formatted_prompt = f"""
    SYSTEM PROMPT: {SYSTEM_PROMPT}

    USER PROMPT: {prompt}
    
    """
    # Generative AIモデルにプロンプトを送信し、生成されたコンテンツを取得
    assistant_message = model_genai.generate_content(formatted_prompt)
    return assistant_message.text
