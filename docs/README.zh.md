[English](../README.md) | [Russian](README.ru.md) | **简体中文** | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | [한국어](README.ko.md)<br>

# Fish Speech 批量推理

这是一个基于 **Fish Speech** 的分支，增强了批量推理功能，实现高效语音生成。

## 🚀 特性

- **批量处理**：一次处理多个文本，加速推理过程  
- **稳定高效**：杜绝空结果，无冗余计算，注意力掩码正确处理

## 🛠️ 使用指南

1. **下载编码器模型**。  
2. **使用参考音频和模型检查点路径生成 `fake.npy` 文件**：

    ```bash
    python fish_speech/models/dac/inference.py \
        -i "ref_audio_name.wav" \
        --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
    ```

    以上命令会生成 `fake.npy`（可指定保存路径）。

3. **在 `fish_batch_inference.py` 中设置 `fake.npy` 路径**。

4. **运行批量推理**：

    ```bash
    python fish_batch_inference.py
    ```

## 🔄 未来规划

- **VQ-GAN 并行加速**，实现更快推理  
- **基于 Gradio 的网页界面**，简化批量处理

## 📊 性能表现

- **速度**：比顺序处理快 3-4 倍  
- **质量**：生成更丰富、更稳定的语音结果

**仓库地址**: https://github.com/mkgs210/batch_fish_speech

*Fish Speech 的批量推理分支，VQ-GAN 并行和 Gradio 支持即将推出！*
