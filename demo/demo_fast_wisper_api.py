import requests
import time

# APIのURLを指定
api_url = "http://localhost:8181/transcribe"

# アップロードする音声ファイルのパスを指定
audio_file_path = r"output\output.wav"

# リクエスト開始時間を記録
start_time = time.time()

# 音声ファイルをアップロードするためのファイルオブジェクトを作成
with open(audio_file_path, "rb") as file:
    # APIにPOSTリクエストを送信
    response = requests.post(api_url, files={"audio_file": file})

# リクエスト終了時間を記録
end_time = time.time()

# リクエストの処理時間を計算
request_time = end_time - start_time

# レスポンスのステータスコードを確認
if response.status_code == 200:
    # レスポンスのJSONデータを取得
    data = response.json()
    
    # 検出された言語と確率を表示
    language_info = data["language_info"]
    print("検出された言語:", language_info["language"])
    print("確率:", language_info["probability"])
    
    # 書き起こされたテキストを表示
    transcribed_text = data["transcribed_text"]
    for segment in transcribed_text:
        print(f"[{segment['start']} -> {segment['end']}] {segment['text']}")
    
    # リクエストの往復時間を表示
    print(f"リクエストの往復時間: {request_time} 秒")
else:
    print("エラーが発生しました。ステータスコード:", response.status_code)