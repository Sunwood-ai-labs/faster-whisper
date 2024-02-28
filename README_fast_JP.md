# **CTranslate2を使用したFaster Whisperの書き起こし** 

**faster-whisper** は、高速な推論エンジンである[CTranslate2](https://github.com/OpenNMT/CTranslate2/) を使用して、OpenAIのWhisperモデルを再実装したものです。

この実装は、[openai/whisper](https://github.com/openai/whisper) と比較して同じ精度で最大4倍高速であり、メモリ使用量も少なくなっています。CPUとGPUの両方で8ビット量子化を使用することで、さらに効率を向上させることができます。

## ベンチマーク
### Whisper

参考のために、異なる実装を使用して[13分間](https://www.youtube.com/watch?v=0u7tTptBo9I) のオーディオを書き起こすのに必要な時間とメモリ使用量をこちらに示します：
### Large-v2モデル（GPU上）実装精度ビームサイズ時間最大GPUメモリ最大CPUメモリopenai/whisperfp1654m30s11325MB9439MBfaster-whisperfp16554s4755MB3244MBfaster-whisperint8559s3091MB3117MB

*NVIDIA Tesla V100S上でCUDA 11.7.1を使用して実行されました。*
### 小型モデル（CPU上）実装精度ビームサイズ時間最大メモリopenai/whisperfp32510m31s3101MBwhisper.cppfp32517m42s1581MBwhisper.cppfp16512m39s873MBfaster-whisperfp3252m44s1675MBfaster-whisperint852m04s995MB

*Intel(R) Xeon(R) Gold 6226R上で8スレッドを使用して実行されました。*
### Distil-whisper実装精度ビームサイズ時間Gigaspeech WERdistil-whisper/distil-large-v2fp164-10.36[faster-distil-large-v2](https://huggingface.co/Systran/faster-distil-whisper-large-v2) fp165-10.28distil-whisper/distil-medium.enfp164-11.21[faster-distil-medium.en](https://huggingface.co/Systran/faster-distil-whisper-medium.en) fp165-11.21

*NVIDIA 3090上でCUDA 11.4を使用して実行されました。*
## 必要条件
- Python 3.8以上

openai-whisperとは異なり、システムにFFmpegをインストールする必要は**ありません** 。オーディオはFFmpegライブラリをパッケージに含むPythonライブラリ[PyAV](https://github.com/PyAV-Org/PyAV) でデコードされます。
### GPU

GPU実行には、以下のNVIDIAライブラリがインストールされている必要があります： 
- [CUDA 11用のcuBLAS](https://developer.nvidia.com/cublas) 
- [CUDA 11用のcuDNN 8](https://developer.nvidia.com/cudnn)

これらのライブラリをインストールする方法は複数あります。推奨される方法は公式のNVIDIAドキュメントに記載されていますが、以下に他のインストール方法も示します。
## インストール

モジュールは[PyPI](https://pypi.org/project/faster-whisper/) からインストールできます：

```bash
pip install faster-whisper
```


## 使用法
### Faster-whisper

```python
from faster_whisper import WhisperModel

model_size = "large-v3"

# GPUでFP16を使用して実行
model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```



**警告:**  `segments`は*ジェネレータ*なので、イテレーションを開始したときにのみ書き起こしが開始されます。書き起こしは、セグメントをリストに集めたり、`for`ループで回したりすることで完了します。
### Faster-distil-whisper

`faster-ditil-whisper`の使用法については、[https://github.com/guillaumekln/faster-whisper/issues/533](https://github.com/guillaumekln/faster-whisper/issues/533)  を参照してください。
### モデル変換

モデルのサイズ、例えば`WhisperModel("large-v3")`からモデルをロードする場合、対応するCTranslate2モデルが自動的に[Hugging Face Hub](https://huggingface.co/Systran) からダウンロードされます。

また、Transformersライブラリと互換性のある任意のWhisperモデルを変換するためのスクリプトも提供しています。これらは、元のOpenAIモデルやユーザーがファインチューニングしたモデルである可能性があります。

たとえば、以下のコマンドは[元の"large-v3" Whisperモデル](https://huggingface.co/openai/whisper-large-v3) を変換し、FP16で重みを保存します：

```bash
pip install transformers[torch]>=4.23

ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2
--copy_files tokenizer.json preprocessor_config.json --quantization float16
```


## 他の実装との性能比較

他のWhisper実装と性能を比較する場合は、類似の設定で比較を実行していることを確認する必要があります。特に： 
- 特に同じビームサイズを使用しているかどうか、同じ書き起こしオプションが使用されていることを確認してください。たとえばopenai/whisperでは、`model.transcribe`はデフォルトでビームサイズ1を使用しますが、ここではデフォルトでビームサイズ5を使用します。 
- CPU上で実行する場合は、同じスレッド数を設定していることを確認してください。多くのフレームワークは環境変数`OMP_NUM_THREADS`を読み取りますが、スクリプトを実行するときに設定することができます：

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```