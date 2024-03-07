from faster_whisper import WhisperModel
import sounddevice as sd
from scipy.io.wavfile import write
import time
import numpy as np

def record_from_mic(duration, fs=44100, filename='temp.wav'):
    print("録音を開始します...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # 録音が終了するまで待機
    write(filename, fs, recording)  # 録音をファイルに保存
    print("録音が完了しました。")

model_size = "large-v3"  # モデルのサイズを指定
model = WhisperModel(model_size, device="cuda", compute_type="float16")  # ここでは例としてFP16を使用

# 録音する音声の長さを秒単位で設定（例：5秒）
record_duration = 5
# マイクから音声を録音
record_from_mic(record_duration)

# 計測開始時刻
start_time = time.time()

# 録音した音声ファイルを書き起こし
segments, info = model.transcribe("temp.wav", beam_size=5)

# 計測終了時刻と実行時間の計算、出力
end_time = time.time()
elapsed_time = end_time - start_time
print(f"実行時間: {elapsed_time}秒")

# 検出された言語とその確率、書き起こされたテキストを表示
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
