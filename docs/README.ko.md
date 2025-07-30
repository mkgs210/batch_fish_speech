[English](../README.md) | [Russian](docs/README.ru.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | **한국어** <br>

# Fish Speech 배치 추론

**Fish Speech**의 포크 버전으로, 효율적인 음성 생성을 위한 배치 추론 기능이 향상되었습니다.

## 🚀 주요 기능

- **배치 처리**: 여러 텍스트를 동시에 처리하여 추론 속도 향상  
- **안정적이고 효율적임**: 빈 결과 없음, 중복 계산 제거, 올바른 어텐션 마스크 처리

## 🛠️ 사용법

1. **코덱 모델 다운로드**  
2. **참조 오디오 및 체크포인트 경로를 사용하여 `fake.npy` 파일 생성**:

    ```bash
    python fish_speech/models/dac/inference.py \
        -i "ref_audio_name.wav" \
        --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
    ```

    이 명령어는 `fake.npy` 파일을 생성합니다 (출력 경로 지정 가능).

3. **`fish_batch_inference.py` 에서 `fake.npy` 경로 설정**  
4. **배치 추론 실행**:

    ```bash
    python fish_batch_inference.py
    ```

## 🔄 향후 계획

- **VQ-GAN 병렬 처리**로 더욱 빠른 추론 구현  
- **Gradio 웹 UI**를 통한 간편한 배치 처리 지원

## 📊 성능

- **속도**: 순차 처리 대비 3~4배 빠름  
- **품질**: 더욱 다양하고 안정적인 음성 출력

**레포지토리**: https://github.com/mkgs210/batch_fish_speech

*Fish Speech 포크 버전의 완전한 배치 추론. VQ-GAN 병렬 처리 및 Gradio 지원 곧 제공 예정!*
