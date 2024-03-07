import requests
import os

# style-bert APIのベースURL
url = "http://style-bert:5000/voice"

# クエリパラメータ
params = {
    "text": "こんにちは、今日の気分はどうですか？",
    "encoding": "utf-8",  # 必要に応じて指定
    "model_id": 0,
    "speaker_id": 0,
    "sdp_ratio": 0.2,
    "noise": 0.6,
    "noisew": 0.8,
    "length": 1,
    "language": "JP",
    "auto_split": True,
    "split_interval": 0.5,
    "assist_text": "",  # この例では空ですが、必要に応じて指定
    "assist_text_weight": 1,
    "style": "Neutral",
    "style_weight": 5
}


# 出力フォルダのパス
output_folder = "audio/output"

# フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 音声ファイルの保存パス
output_path = os.path.join(output_folder, "output.wav")

# リクエストの送信とレスポンスの確認
response = requests.get(url, params=params)  # `url` と `params` は前の手順と同じ

if response.status_code == 200:
    # レスポンスから音声データを取得し、ファイルに保存
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"音声ファイルを '{output_path}' に保存しました。")
else:
    print("エラーが発生しました。ステータスコード:", response.status_code)