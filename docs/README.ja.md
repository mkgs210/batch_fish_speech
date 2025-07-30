[English](../README.md) | [Russian](README.ru.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | **日本語** | [한국어](README.ko.md)<br>

# Fish Speech バッチ推論

**Fish Speech** のフォークで、効率的な音声生成のためにバッチ推論機能を強化しました。

## 🚀 特長

- **バッチ処理**: 複数のテキストを同時に処理し推論速度を向上  
- **安定かつ効率的**: 空の結果なし、冗長な計算を排除し、正しいアテンションマスクを適用

## 🛠️ 使い方

1. **コーデックモデルをダウンロード**  
2. **参照音声ファイルとチェックポイントパスを指定して `fake.npy` ファイルを作成**:

    ```bash
    python fish_speech/models/dac/inference.py \
        -i "ref_audio_name.wav" \
        --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
    ```

    このコマンドで `fake.npy` が生成されます（出力パスは指定可能）。

3. **`fish_batch_inference.py` 内で `fake.npy` のパスを設定**  
4. **バッチ推論を実行**:

    ```bash
    python fish_batch_inference.py
    ```

## 🔄 今後の予定

- **VQ-GAN 並列化** によるさらなる推論高速化  
- **Gradio ウェブUI** による使いやすいバッチ処理の実装

## 📊 パフォーマンス

- **速度**: 順次処理より3〜4倍高速  
- **品質**: より多様で安定した音声生成

**リポジトリ**: https://github.com/mkgs210/batch_fish_speech

*Fish Speech のフォーク版で完全なバッチ推論を実装。VQ-GAN 並列化と Gradio 対応は近日公開予定！*
