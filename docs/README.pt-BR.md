[English](../README.md) | [Russian](README.ru.md) | [简体中文](README.zh.md) | **Portuguese** | [日本語](README.ja.md) | [한국어](README.ko.md)<br>

# Fish Speech Batch Inference

Este é um fork do **Fish Speech** com suporte aprimorado para inferência em batch, para geração eficiente de voz.

## 🚀 Recursos

- **Processamento em lote**: Processa múltiplos textos simultaneamente para acelerar a inferência  
- **Estável e eficiente**: Sem resultados vazios, com cálculos otimizados e máscaras de atenção corretas

## 🛠️ Uso

1. **Baixe o modelo do codec**.  
2. **Crie o arquivo `fake.npy` com seu áudio de referência e o caminho para o checkpoint**:

    ```bash
    python fish_speech/models/dac/inference.py \
        -i "ref_audio_name.wav" \
        --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
    ```

    Esse comando irá gerar o arquivo `fake.npy` (é possível especificar caminho de saída).

3. **Configure o caminho para o `fake.npy` no arquivo `fish_batch_inference.py`.**

4. **Execute a inferência em batch**:

    ```bash
    python fish_batch_inference.py
    ```

## 🔄 Roteiro

- **Paralelização do VQ-GAN** para inferência ainda mais rápida  
- **Interface Web Gradio** para facilitar o processamento em lote

## 📊 Desempenho

- **Velocidade**: Até 3-4 vezes mais rápido que o processamento sequencial  
- **Qualidade**: Resultados em áudio mais diversos e robustos

**Repositório**: https://github.com/mkgs210/batch_fish_speech

*Fork do Fish Speech com inferência em batch completa. Suporte a VQ-GAN paralelo e Gradio está a caminho!*

[1] https://github.com/mkgs210/batch_fish_speech/blob/main/docs/docs/README.pt-BR.md
