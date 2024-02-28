**faster-whisper** は、OpenAIのWhisperモデルを[CTranslate2](https://github.com/OpenNMT/CTranslate2/) を使って再実装したものです。CTranslate2は、Transformerモデルのための高速な推論エンジンです。

https://github.com/Sunwood-ai-labs/faster-whisper



この実装は、同じ精度で[openai/whisper](https://github.com/openai/whisper) よりも最大4倍高速で、メモリ使用量も少なくなっています。CPUとGPUの両方で8ビット量子化を使用することで、さらに効率を向上させることができます。
## ベンチマーク

### Whisper

参考までに、異なる実装を使用して[13分間](https://www.youtube.com/watch?v=0u7tTptBo9I) のオーディオを書き起こすのに必要な時間とメモリ使用量を示します： 
- [openai/whisper](https://github.com/openai/whisper) @[6dea21fd](https://github.com/openai/whisper/commit/6dea21fd7f7253bfe450f1e2512a0fe47ee2d258) 
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) @[3b010f9](https://github.com/ggerganov/whisper.cpp/commit/3b010f9bed9a6068609e9faf52383aea792b0362) 
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) @[cce6b53e](https://github.com/guillaumekln/faster-whisper/commit/cce6b53e4554f71172dad188c45f10fb100f6e3e)

### GPU上のLarge-v2モデル

| Implementation | Precision | Beam size | Time | Max. GPU memory | Max. CPU memory |
| --- | --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 4m30s | 11325MB | 9439MB |
| faster-whisper | fp16 | 5 | 54s | 4755MB | 3244MB |
| faster-whisper | int8 | 5 | 59s | 3091MB | 3117MB |

*Executed with CUDA 11.7.1 on a NVIDIA Tesla V100S.*


### CPU上のSmallモデル

| Implementation | Precision | Beam size | Time | Max. memory |
| --- | --- | --- | --- | --- |
| openai/whisper | fp32 | 5 | 10m31s | 3101MB |
| whisper.cpp | fp32 | 5 | 17m42s | 1581MB |
| whisper.cpp | fp16 | 5 | 12m39s | 873MB |
| faster-whisper | fp32 | 5 | 2m44s | 1675MB |
| faster-whisper | int8 | 5 | 2m04s | 995MB |

*Executed with 8 threads on a Intel(R) Xeon(R) Gold 6226R.*


### Distil-whisper

| Implementation | Precision | Beam size | Time | Gigaspeech WER |
| --- | --- | --- | --- | --- |
| distil-whisper/distil-large-v2 | fp16 | 4 |- | 10.36 |
| [faster-distil-large-v2](https://huggingface.co/Systran/faster-distil-whisper-large-v2) | fp16 | 5 | - | 10.28 |
| distil-whisper/distil-medium.en | fp16 | 4 | - | 11.21 |
| [faster-distil-medium.en](https://huggingface.co/Systran/faster-distil-whisper-medium.en) | fp16 | 5 | - | 11.21 |

*Executed with CUDA 11.4 on a NVIDIA 3090.*

## 要件
- Python 3.8以上

openai-whisperとは異なり、システムにFFmpegをインストールする必要は**ありません** 。オーディオは、FFmpegライブラリをパッケージに含むPythonライブラリ[PyAV](https://github.com/PyAV-Org/PyAV) を使用してデコードされます。

### GPU

GPU実行には、以下のNVIDIAライブラリがインストールされている必要があります： 
- [CUDA 11用cuBLAS](https://developer.nvidia.com/cublas) 
- [CUDA 11用cuDNN 8](https://developer.nvidia.com/cudnn)

これらのライブラリをインストールする方法は複数あります。公式のNVIDIAドキュメントで説明されている方法が推奨されていますが、以下に他のインストール方法も示します。

## インストール

モジュールは[PyPI](https://pypi.org/project/faster-whisper/) からインストールできます：

```bash
pip install faster-whisper
```

## 使用法

### Faster-whisperの使用

Faster-whisperを使用して、音声ファイルの書き起こしを行う例を示します。

```python

from faster_whisper import WhisperModel

model_size = "large-v3"

# GPUでFP16を使用して実行
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# GPUでINT8を使用して実行
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# CPUでINT8を使用して実行
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("検出された言語 '%s' の確率 %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```



**警告:**  `segments`は*ジェネレータ*なので、イテレートすることで書き起こしが開始されます。セグメントをリストに集約するか、`for`ループで実行することで、書き起こしを完了させることができます。

```python

segments, _ = model.transcribe("audio.mp3")
segments = list(segments)  # ここで書き起こしが実際に実行されます。
```


### Faster-distil-whisper

Faster-distil-whisperの使用については、[こちら](https://github.com/guillaumekln/faster-whisper/issues/533) を参照してください。

```python

model_size = "distil-large-v2"
# model_size = "distil-medium.en"
model = WhisperModel(model_size, device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5, 
    language="en", max_new_tokens=128, condition_on_previous_text=False)
```



経験則によると、`condition_on_previous_text=True`は長いオーディオのパフォーマンスを低下させます。`initial_prompt`を使用した場合にも、最初のチャンクでの劣化が観察されました。

### 単語レベルのタイムスタンプ

```python

segments, _ = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
```


### VADフィルター

ライブラリは[Silero VAD](https://github.com/snakers4/silero-vad) モデルを統合して、発話のないオーディオ部分をフィルタリングします。

```python

segments, _ = model.transcribe("audio.mp3", vad_filter=True)
```



デフォルトの挙動は保守的で、2秒以上の無音を除去します。利用可能なVADパラメーターとデフォルト値は[ソースコード](https://github.com/guillaumekln/faster-whisper/blob/master/faster_whisper/vad.py) で確認できます。これらは`vad_parameters`辞書引数でカスタマイズ可能です。

### ログ設定

ライブラリのログレベルは以下のように設定できます。

```python

import logging

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
```











