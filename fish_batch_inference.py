from fish_speech.models.text2semantic.inference import batch_inference, load_model
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
import torch
import numpy as np
import os
import time
from loguru import logger
from pathlib import Path
import soundfile as sf

text = [
    "Что? Он никогда не работал?",
    "Никогда бы о таком не подумал раньше"
]
prompt_text = [
    "Как это не надо? Ты первый вор! Ты первый вор, гнида, паршивая! А вы все депутаты, там, нахлебники! Их... Один Столыпин придумал программу какую! А они сидят."# там миллионы, жируют! А этого вообще клоуна надо убрать! Вот клоуна этого! Он позорит наше вот это всё! И вообще там их всех надо, понимаешь? Вот эти, показывать старческие, вот эти вот, молодых надо!"
] * 2
prompt_tokens = [Path("fake.npy")] * 2
num_samples = 1
max_new_tokens = 256
top_p = 0.7
repetition_penalty = 1.2
temperature = 0.7
checkpoint_path = Path("/home/mkgs/fish_speech/new_fish/fish/checkpoints/fish-speech-1.5")
device = "cuda"
compile = False
seed = 1
half = False
output_dir = Path("temp")
batch_size = 2

# Параметры для vqgan
vqgan_config = "firefly_gan_vq"
vqgan_checkpoint = "/home/mkgs/fish_speech/new_fish/fish/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
vqgan_device = device

os.makedirs(output_dir, exist_ok=True)
precision = torch.half if half else torch.bfloat16

if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
    raise ValueError(
        f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
    )

logger.info("Loading model ...")
t0 = time.time()
model, decode_one_token = load_model(
    checkpoint_path, device, precision, compile=compile
)
with torch.device(device):
    model.setup_caches(
        max_batch_size=batch_size,
        max_seq_len=model.config.max_seq_len,
        dtype=next(model.parameters()).dtype,
    )
if torch.cuda.is_available():
    torch.cuda.synchronize()

logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

if prompt_tokens is not None:
    prompt_tokens_ = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]
else:
    prompt_tokens_ = None

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

responses = batch_inference(
    model=model,
    device=device,
    decode_one_token=decode_one_token,
    text=text,
    num_samples=num_samples,
    max_new_tokens=max_new_tokens,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
    temperature=temperature,
    compile=compile,
    prompt_text=prompt_text,
    prompt_tokens=prompt_tokens_,
)

# Функция для декодирования кодов в аудио через vqgan
def codes_to_wav(codes, output_path, vqgan_model):
    import torch
    import numpy as np
    import soundfile as sf
    device = next(vqgan_model.parameters()).device
    codes = torch.cat(codes, dim=1).long().to(device)  # [num_codebooks, T]
    if codes.ndim == 3:
        codes = codes[0]  # [num_codebooks, T]
    feature_lengths = torch.tensor([codes.shape[1]], device=device)
    fake_audios, _ = vqgan_model.decode(indices=codes[None], feature_lengths=feature_lengths)
    fake_audio = fake_audios[0, 0].float().detach().cpu().numpy()
    sf.write(output_path, fake_audio, vqgan_model.spec_transform.sample_rate)

# Загрузка vqgan модели один раз
vqgan_model = load_vqgan_model(vqgan_config, vqgan_checkpoint, device=vqgan_device)


idx = 0
for response in responses:
    if hasattr(response, 'action') and response.action == "sample":
        # ---------- NEW ----------
        # теперь response.codes – это ОДИН запрос, поэтому сохраняем сразу
        wav_path = os.path.join(output_dir, f"gen_{idx}.wav")
        codes_to_wav([response.codes], wav_path, vqgan_model)
        logger.info(f"[{idx}] {response.text} → {wav_path}")
        idx += 1
        # ---------- /NEW ----------
    # блок "next" больше не нужен, так как каждый сэмпл
    # записывается прямо в ветке "sample"
    elif hasattr(response, "action") and response.action == "next":
        logger.info("Finished current batch‑sample")
