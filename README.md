**English** | [Russian](docs/README.ru.md) | [简体中文](docs/README.zh.md) | [Portuguese](docs/README.pt-BR.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) <br>

# Fish Speech Batch Inference

A **Fish Speech fork** with enhanced batch inference for efficient speech generation.

Batch inference allows you generate multiple audio at once instead of one by one. This makes the process much faster and saves time. To use batch inference, set up your reference audio file and texts, and configure generation settings as shown in the usage section below.


## 🚀 Features

- **Batch Processing**: Handles multiple texts at once for faster inference
- **Stable & Efficient**: No empty results, no redundant calculations, correct attention masking

## 🛠️ Usage

1. **Download the codec model**.
2. **Create the `fake.npy` file** with your reference audio and the checkpoint path:

    ```bash
    python fish_speech/models/dac/inference.py \
        -i "ref_audio_name.wav" \
        --checkpoint-path "checkpoints/fish-speech-1.5/"
    ```

    This command will generate `fake.npy` (specify the output path if needed).

3. **Set the path to `fake.npy`** in `fish_batch_inference.py`.

4. **Run batch inference**:

    ```bash
    python fish_batch_inference.py
    ```

## 🔄 Roadmap

- **VQ-GAN Parallelization** for even faster inference
- **Gradio Web UI** for easy batch processing

## 📊 Performance

- **Speed**: Up to 3-4x faster than sequential processing
- **Quality**: More diverse and robust audio results

**Repository**: https://github.com/mkgs210/batch_fish_speech

*Fish Speech fork with true batch inference. VQ-GAN and Gradio support coming soon!*
