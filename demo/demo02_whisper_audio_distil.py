# 使用法
# Faster-whisperを使用する
from faster_whisper import WhisperModel
import time

# model_size = "large-v3"  # モデルのサイズを指定
model_size = "distil-large-v2"

# GPUでFP16を使用して実行
model = WhisperModel(model_size, device="cuda", compute_type="float16")
# またはGPUでINT8を使用して実行
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# またはCPUでINT8を使用して実行
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# 計測開始時刻
start_time = time.time()
# 音声ファイルを書き起こし、beam_sizeを5に設定して精度を向上させる
segments, info = model.transcribe("audio/Word2Motion/001/02_nodding.wav", beam_size=5, language='ja')
# 計測終了時刻
end_time = time.time()
# 実行時間の計算
elapsed_time = end_time - start_time
# 実行時間の出力
print(f"実行時間: {elapsed_time}秒")
# 検出された言語とその確率を表示
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# 書き起こされた各セグメント（テキスト）とその時間範囲を表示
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    
# 注意: segmentsはジェネレータなので、イテレート開始時に書き起こしが始まります。
# 以下のようにセグメントをリストに集約するか、forループでイテレートすることで、書き起こしを完了させることができます：

# segments, _ = model.transcribe("audio/Word2Motion/001/02_nodding.wav")
# segments = list(segments)  # 実際の書き起こしがここで実行されます。
# print(segments)
