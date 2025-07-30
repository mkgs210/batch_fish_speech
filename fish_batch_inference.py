from fish_speech.models.text2semantic.inference import batch_generate_long, load_model
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
import torch
import numpy as np
import os
import time
from loguru import logger
from pathlib import Path
import soundfile as sf

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

text = [
    "Text 1",
    "Text 2",
]
prompt_text = [
    "Your prompt text"
] * len(text)
prompt_tokens = [Path("fake.npy")] * len(text)
num_samples = 1
max_new_tokens = 256
chunk_length = 100
top_p = 0.75
repetition_penalty = 1.9
temperature = 0.4
seed = 42
checkpoint_path = Path("/home/mkgs/fish_speech/new_fish/fish/checkpoints/fish-speech-1.5")
device = "cuda"
compile = True
half = False
output_dir = Path("temp")
batch_size = len(text)
iterative_prompt = True

# Параметры для vqgan
vqgan_config = "firefly_gan_vq"
vqgan_checkpoint = "/home/mkgs/fish_speech/new_fish/fish/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
vqgan_device = device

os.makedirs(output_dir, exist_ok=True)
precision = torch.half if half else torch.bfloat16

logger.info("Starting batch inference script...")
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

prompt_tokens_ = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# --- COLD START (compile) ---
if compile:
    logger.info("Cold start: compiling model for fast batch inference...")
    t_compile = time.time()
    _ = list(batch_generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=["Привет!"]*len(text),
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        chunk_length=chunk_length,
        iterative_prompt=iterative_prompt,
        prompt_text=prompt_text, #[""]*len(text),
        prompt_tokens=prompt_tokens_, #[None]*len(text),
        target_batch_size=len(text),
    ))
    logger.info(f"Cold start complete. Model is ready for fast batch inference. Compile time: {time.time() - t_compile:.2f} seconds")

start_time = time.time()

responses = batch_generate_long(
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
    chunk_length=chunk_length,
    iterative_prompt=iterative_prompt,
    target_batch_size=len(text),
)

# Функция для декодирования кодов в аудио через vqgan

def codes_to_wav(codes, output_path, vqgan_model):
    device = next(vqgan_model.parameters()).device
    codes = torch.cat(codes, dim=1).long().to(device) # [num_codebooks, T]
    
    # ПРОВЕРКА НА ПУСТЫЕ КОДЫ
    if codes.shape[-1] == 0:
        logger.warning(f"Empty codes passed to codes_to_wav, skipping {output_path}")
        return
        
    if codes.ndim == 3:
        codes = codes[0] # [num_codebooks, T]
    
    feature_lengths = torch.tensor([codes.shape[1]], device=device)
    t0 = time.time()
    fake_audios, _ = vqgan_model.decode(indices=codes[None], feature_lengths=feature_lengths)
    fake_audio = fake_audios[0, 0].float().detach().cpu().numpy()
    t_decode = time.time() - t0
    sf.write(output_path, fake_audio, vqgan_model.spec_transform.sample_rate)
    logger.info(f"Decoded and saved audio to {output_path} in {t_decode:.2f} seconds")

# Загрузка vqgan модели один раз
t_vqgan = time.time()
vqgan_model = load_vqgan_model(vqgan_config, vqgan_checkpoint, device=vqgan_device)
logger.info(f"Loaded VQGAN model in {time.time() - t_vqgan:.2f} seconds")

for response in responses:
    t_ch = time.time()
    if hasattr(response, 'action') and response.action == "sample" and response.chunk_idx != -1:
        wav_path = os.path.join(output_dir, f"gen_{response.batch_idx}_{response.chunk_idx}.wav")
        logger.info(f"[BATCH] Generating chunk: batch_idx={getattr(response, 'batch_idx', None)}, chunk_idx={getattr(response, 'chunk_idx', None)}, text_len={len(response.text) if response.text else 0}")
        codes_to_wav([response.codes], wav_path, vqgan_model)
        logger.info(f"[BATCH] [b{response.batch_idx}:c{response.chunk_idx}] {response.text} → {wav_path} | Time: {time.time() - t_ch:.2f}s")
    elif hasattr(response, "action") and response.action == "next":
        logger.info("Finished current batch‑sample")

logger.info(f"Total batch inference time after: {time.time() - start_time:.2f} seconds")
