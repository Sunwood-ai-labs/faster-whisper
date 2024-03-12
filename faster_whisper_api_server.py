from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    # 一時ファイルに音声ファイルを保存
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await audio_file.read())
        temp_file_path = temp_file.name

    # 音声ファイルを書き起こし
    segments, info = model.transcribe(temp_file_path, beam_size=5)

    # 一時ファイルを削除
    os.unlink(temp_file_path)

    # 検出された言語とその確率を返す
    language_info = {
        "language": info.language,
        "probability": info.language_probability
    }

    # 書き起こされたテキストを返す
    transcribed_text = []
    for segment in segments:
        transcribed_text.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })

    return {"language_info": language_info, "transcribed_text": transcribed_text}