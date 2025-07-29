import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.conversation import (
    CODEBOOK_PAD_TOKEN_ID,
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from fish_speech.models.text2semantic.llama import BaseModelArgs
from fish_speech.models.text2semantic.utils import collate
from fish_speech.text import clean_text, split_text
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer

from torch.compiler import cudagraph_mark_step_begin

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync_agent(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def sample_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs_agent(
        logits=logits[:, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync_agent(probs)
    return idx_next, probs


def decode_one_token_ar_agent(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # print(x, input_pos)
    x = model.forward_generate(x, input_pos)
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    sampling_kwargs_main = sampling_kwargs.copy()
    sampling_kwargs_main["temperature"] = 0.1
    sampling_kwargs_main["top_p"] = 0.1
    sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample_agent(
            logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample_agent(
            logits,
            previous_tokens=(
                previous_tokens[:, codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~torch.isin(codebooks[:, :1, :], semantic_ids_tensor),
        CODEBOOK_PAD_TOKEN_ID,
    )

    return codebooks

def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # Теперь можем подавать маску, если она есть
    result = model.forward_generate(
        x, input_pos, key_padding_mask=key_padding_mask
    )
    
    sampling_kwargs_main = sampling_kwargs.copy()
    
    # Автоопределение batch режима по размеру тензора
    is_batched = result.logits.size(0) > 1
    
    if is_batched:
        _sample = sample_agent
        prev_tokens_slice = (
            previous_tokens[:, 0] if previous_tokens is not None else None
        )
    else:
        _sample = sample
        prev_tokens_slice = (
            previous_tokens[0] if previous_tokens is not None else None
        )
    
    codebooks = [
        _sample(
            result.logits,  # ИСПОЛЬЗУЕМ result.logits вместо model.lm_head
            previous_tokens=prev_tokens_slice,
            **sampling_kwargs_main,
        )[0]
    ]
    
    hidden_states = result.hidden_states
    
    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)
    
    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)
    
    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        
        if is_batched:
            prev_tokens = (
                previous_tokens[:, codebook_idx + 1]
                if previous_tokens is not None else None
            )
        else:
            prev_tokens = (
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None else None
            )
            
        a = _sample(
            logits,
            previous_tokens=prev_tokens,
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)
    
    # Автоопределение размерности для stack
    if is_batched:
        codebooks = torch.stack(codebooks, dim=1)
    else:
        codebooks = torch.stack(codebooks, dim=0)
    
    return codebooks




aboba = []

def decode_n_tokens(
    model,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    decode_one_token=decode_one_token_ar,
    **sampling_kwargs,
):
    logger.info("decode_n_tokens")
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]
        
        global aboba
        if not aboba:
            aboba = [cur_token.shape, previous_tokens.shape, window.shape, input_pos.shape, len(semantic_ids)]
            logger.info(f"cur_token: {cur_token.shape}")
            logger.info(f"previous_tokens: {previous_tokens.shape}")
            logger.info(f"window: {window.shape}")
            logger.info(f"input_pos: {input_pos.shape}")
            logger.info(f"semantic_ids: {len(semantic_ids)}")
        elif aboba != [cur_token.shape, previous_tokens.shape, window.shape, input_pos.shape, len(semantic_ids)]:
            logger.info(f"[ABOBA] cur_token: {cur_token.shape}, previous_tokens: {previous_tokens.shape}, window: {window.shape}, input_pos: {input_pos.shape}, semantic_ids: {len(semantic_ids)}")

        with (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
            if torch.cuda.is_available()
            else nullcontext()
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break

    return previous_tokens[:, : i + 1]


def decode_n_tokens_batched(
    model,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    key_padding_mask: torch.Tensor | None = None,   # NEW
    decode_one_token=decode_one_token_ar,
    **sampling_kwargs,
):
    logger.info("decode_n_tokens_batched")
    
    batch_size = cur_token.size(0)
    previous_tokens = torch.zeros(
        (batch_size, model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )
    
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=cur_token.device)
    finished = finished | (cur_token[:, 0, -1] == im_end_id)
    
    final_tokens = cur_token.clone()
    
    for i in tqdm(range(num_new_tokens)):
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :, :win_size]
        else:
            window = previous_tokens[:, :, i - win_size : i]
            
        with (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
            if torch.cuda.is_available()
            else nullcontext()
        ):
            # УБИРАЕМ ВСЕ ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            ).clone()
        
        input_pos += 1
        
        # Обработка завершенных элементов
        for j in range(batch_size):
            if finished[j]:
                next_token[j] = final_tokens[j]
            else:
                final_tokens[j] = next_token[j]
                
        cur_token = next_token.view(batch_size, model.config.num_codebooks + 1, -1)
        previous_tokens[:, :, i : i + 1] = next_token.view(
            batch_size, model.config.num_codebooks + 1, -1
        )
        
        new_finished = (cur_token[:, 0, -1] == im_end_id)
        finished = finished | new_finished
        
        if finished.all():
            break
    
    return previous_tokens[:, :, : i + 1]



@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    decode_one_token=decode_one_token_ar,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    logger.info("generate")
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    # semantic_id = model.tokenizer.convert_tokens_to_ids("<|semantic|>")
    semantic_ids = [
        model.tokenizer.get_token_id(f"<|semantic:{i}|>") for i in range(1024)
    ]

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype

    codebook_dim = 1 + model.config.num_codebooks
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (decode_one_token_ar)

    next_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),  # assume bs = 1
        input_pos,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )
    seq[:, T : T + 1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),  # assume bs = 1
        input_pos,
        max_new_tokens - 1,
        decode_one_token=decode_one_token,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )
    # x = torch.cat(generated_tokens, dim=1)
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    return seq

def prefill_kv_cache_only(
    model: DualARTransformer,
    prompt: torch.Tensor,
    input_pos: torch.Tensor,
    key_padding_mask=None,
) -> torch.Tensor:
    """
    Заполняет KV-кэш модели без генерации токенов
    Возвращает hidden_states для первого токена генерации
    """
    # Простой проход через модель только для заполнения кэша
    with torch.inference_mode():
        result = model.forward_generate(prompt, input_pos, key_padding_mask=key_padding_mask)
    
    # Возвращаем hidden states последней позиции для инициализации генерации
    return result.hidden_states[:, :, -1:].clone()

@torch.no_grad()
@torch.inference_mode()
def generate_batched(
    *,
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    decode_one_token=decode_one_token_ar,
    prompt_mask=None,  # Получаем, но не используем для устранения перекомпиляции
    **sampling_kwargs,
) -> torch.Tensor:
    logger.info("generate_batched")
    
    bs, C, T = prompt.shape
    device, dtype = prompt.device, prompt.dtype
    
    semantic_ids = [
        model.tokenizer.get_token_id(f"<|semantic:{i}|>") for i in range(1024)
    ]
    
    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

    codebook_dim = 1 + model.config.num_codebooks
    seq = torch.empty(
        (bs, codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    seq[:, :, :T] = prompt
    
    input_pos = torch.arange(0, T, device=device)
    
    # --- PREFILL ---
    prefill_start_time = time.time()
    next_token = decode_one_token_ar(        # EAGER ⇒ без доп. графа
        model,
        prompt.view(bs, codebook_dim, -1),
        input_pos,
        semantic_ids=semantic_ids,
        key_padding_mask=prompt_mask,        # ← маска работает!
        **sampling_kwargs,
    ).clone()
    
    prefill_end_time = time.time()
    prefill_time = prefill_end_time - prefill_start_time
    logger.info(f"Prefill decode_one_token time: {prefill_time:.3f} seconds")
    
    seq[:, :, T:T+1] = next_token.view(bs, codebook_dim, -1)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    
    x = decode_n_tokens_batched(
        model,
        next_token.view(bs, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        decode_one_token=decode_one_token,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )
    
    seq = seq[:, :, : T + 1 + x.size(2)]
    seq[:, :, T + 1 :] = x.view(bs, model.config.num_codebooks + 1, -1)
    return seq


def decode_n_tokens_agent(
    model,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_ar_agent,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    batch_size = cur_token.size(0)
    previous_tokens = torch.zeros(
        (batch_size, model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=cur_token.device)
    finished = finished | (cur_token[:, 0, -1] == im_end_id)
    start_time = time.time()

    for i in tqdm(range(num_new_tokens), desc="Decoding: ", total=num_new_tokens):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :, :win_size]
        else:
            window = previous_tokens[:, :, i - win_size : i]

        with sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(batch_size, model.config.num_codebooks + 1, -1)
        previous_tokens[:, :, i : i + 1] = next_token.view(
            batch_size, model.config.num_codebooks + 1, -1
        )

        yield cur_token.cpu()

        finished = finished | (cur_token[:, 0, -1] == im_end_id)
        if finished.all() or (
            0 < early_stop_threshold < 1
            and finished.sum() >= round(batch_size * early_stop_threshold)
        ):
            break

    total_time = time.time() - start_time
    generated_tokens = i + 1
    tokens_per_second = (generated_tokens / total_time) * batch_size
    logger.info(
        f"Decoded {generated_tokens} x {batch_size} tokens in {total_time:.2f}s ({tokens_per_second:.2f} tokens/s)"
    )


@torch.no_grad()
@torch.inference_mode()
def generate_agent(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_ar_agent,
    num_samples: int = 1,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (decode_one_token_ar_agent)
    next_token = prefill_decode(
        model,
        prompt,
        input_pos,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    ).view(num_samples, codebook_dim, -1)
    yield next_token.cpu()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    yield from decode_n_tokens_agent(
        model,
        next_token,
        input_pos,
        max_new_tokens - 1,
        im_end_id=im_end_id,
        semantic_ids=semantic_ids,
        decode_one_token=decode_one_token,
        early_stop_threshold=early_stop_threshold,
        **sampling_kwargs,
    )


def encode_tokens(
    tokenizer,
    string,
    device="cuda",
    prompt_tokens=None,
    num_codebooks=4,
):
    string = clean_text(string)

    messages = []
    messages.append(
        Message(
            role="user",
            parts=[TextPart(text=string)],
            cal_loss=False,
        )
    )

    if prompt_tokens is not None:
        if prompt_tokens.ndim == 3:
            assert (
                prompt_tokens.shape[0] == 1
            ), f"3D prompt tokens should have shape (1, num_codebooks, seq_len), found {prompt_tokens.shape[0]}"
            prompt_tokens = prompt_tokens[0]

        assert prompt_tokens.ndim == 2, "Prompt tokens should be 2D tensor"

        if prompt_tokens.shape[0] > num_codebooks:
            logger.warning(
                f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
            )
            prompt_tokens = prompt_tokens[:num_codebooks]

        vq_part = VQPart(codes=prompt_tokens.to(device))

        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>"), vq_part],
                cal_loss=False,
            )
        )
    else:
        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>")],
                cal_loss=False,
                add_im_end=False,
            )
        )

    conversation = Conversation(messages=messages)
    # conversation.visualize(tokenizer)
    encoded = conversation.encode_for_inference(
        tokenizer=tokenizer,
        num_codebooks=num_codebooks,
    )

    return encoded.to(device)


def load_model(checkpoint_path, device, precision, compile=False, is_agent=False):
    model: DualARTransformer = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True, is_agent=is_agent
    )

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    decode_one_token = (
        decode_one_token_ar_agent if is_agent else decode_one_token_ar
    )
    logger.info("Using DualARTransformer")

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            fullgraph=True,
            dynamic=False,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None
    batch_idx: Optional[int] = None
    chunk_idx: Optional[int] = None


def generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: int = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    compile: bool = False,
    iterative_prompt: bool = True,
    max_length: int = 2048,
    chunk_length: int = 150,
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    logger.info("generate_long")
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    im_end_id = tokenizer.get_token_id("<|im_end|>")

    encoded = []
    texts = split_text(text, chunk_length) if iterative_prompt else [text]
    encoded_prompts = [
        Conversation(
            messages=[
                Message(
                    role="system",
                    parts=[TextPart(text="Speak out the provided text.")],
                    cal_loss=False,
                )
            ]
        )
        .encode_for_inference(
            tokenizer=tokenizer,
            num_codebooks=model.config.num_codebooks,
        )
        .to(device)
    ]

    if use_prompt:
        for idx, (t, c) in enumerate(zip(prompt_text, prompt_tokens)):
            encoded_prompts.append(
                encode_tokens(
                    tokenizer,
                    string=t,
                    device=device,
                    prompt_tokens=c,
                    num_codebooks=model.config.num_codebooks,
                )
            )

    for idx, text in enumerate(texts):
        encoded.append(
            encode_tokens(
                tokenizer,
                string=text,
                device=device,
                num_codebooks=model.config.num_codebooks,
            )
        )
        logger.info(f"Encoded text: {text}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        global_encoded = []
        seg_idx = 0

        while seg_idx < len(encoded):
            logger.info(
                f"Generating sentence {seg_idx + 1}/{len(encoded)} of sample {sample_idx + 1}/{num_samples}"
            )

            seg = encoded[seg_idx]
            global_encoded.append(seg)

            lengths = reversed([seg.size(1) for seg in global_encoded])

            # Pick last 2000 tokens
            count = 0
            for i, length in enumerate(lengths):
                count += length
                if count + length > max_length - 1024 - sum(
                    t.shape[1] for t in encoded_prompts
                ):
                    break

            if i != 0 and i % 2 == 0:
                i -= 1

            # Rotate the list, always make sure first segment is included to avoid drift
            if i < len(global_encoded) - 2:
                partial_encoded = global_encoded[:2] + global_encoded[-i:]
            else:
                partial_encoded = global_encoded

            if use_prompt:
                partial_encoded = encoded_prompts + partial_encoded

            cat_encoded = torch.cat(partial_encoded, dim=1)
            prompt_length = cat_encoded.size(1)

            logger.info(f"[SINGLE] cat_encoded shape: {list(cat_encoded.shape)}")
            logger.info(f"[SINGLE] prompt_length: {prompt_length}")
            
            t0 = time.perf_counter()
            y = generate(
                model=model,
                prompt=cat_encoded,
                max_new_tokens=max_new_tokens,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if sample_idx == 0 and seg_idx == 0 and compile:
                logger.info(f"generate: {time.perf_counter() - t0:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t = time.perf_counter() - t0

            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t
            logger.info(
                f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

            if torch.cuda.is_available():
                logger.info(
                    f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
                )

            # Put the generated tokens
            # since there is <im_end>, we remove last token
            codes = y[1:, prompt_length + 1 :].clone()
            assert (codes >= 0).all(), f"Negative code found"

            decoded = y[:, prompt_length:].clone()
            # But for global encoding, we should keep the <im_end> token

            global_encoded.append(decoded)
            assert (codes >= 0).all(), f"Negative code found: {codes}"
            yield GenerateResponse(action="sample", codes=codes, text=texts[seg_idx])
            seg_idx += 1

        # This indicates the end of the current sample
        yield GenerateResponse(action="next")


def batch_inference(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: str | list[str],
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: int = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    compile: bool = False,
    max_length: int = 2048,
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    logger.info("batch_inference")
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"
    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    im_end_id = tokenizer.get_token_id("<|im_end|>")

    segs = []
    encodeds = []
    encoded_prompts_ = []
    for idx, t in enumerate(text):
        prompt_stack = [
            Conversation(
                messages=[
                    Message(
                        role="system",
                        parts=[TextPart(text="Speak out the provided text.")],
                        cal_loss=False,
                    )
                ]
            )
            .encode_for_inference(
                tokenizer=tokenizer,
                num_codebooks=model.config.num_codebooks,
            )
            .to(device)
        ]

        if use_prompt:
            prompt_stack.append(
                encode_tokens(
                    tokenizer,
                    string=prompt_text[idx],
                    device=device,
                    prompt_tokens=prompt_tokens[idx],
                    num_codebooks=model.config.num_codebooks,
                )
            )
        # ---------- NEW ----------
        # объединяем все system / voice‑prompt сегменты в ОДИН тензор
        encoded_prompt_tensor = torch.cat(prompt_stack, dim=1)
        encoded_prompts_.append(encoded_prompt_tensor)
        # ---------- /NEW ----------
        tokens = encode_tokens(
            tokenizer,
            string=t,
            device=device,
            num_codebooks=model.config.num_codebooks,
        )
        logger.info(f"Encoded text: {t}")
        encodeds.extend([tokens])

        # segs формируем из prompt + самого текста
        segs.append(torch.cat([encoded_prompt_tensor, tokens], dim=1))  # CHANGED

    original_text_lengths = []
    for encoded in encodeds:
        original_text_lengths.append(encoded.shape[1])  # длина текста без padding
    
    segs, segs_mask = collate(
        segs, end_of_text=tokenizer.get_token_id("<|end_of_text|>"), pad_right=False
    )
    logger.info(
        f"segs shape {list(segs.shape)}, mask {segs_mask.shape}"
    )
    segs_mask = segs_mask.to(device=device)
    
    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    # We rely on compiled decode_one_token only (same as single inference); leave generate_batched in eager mode to keep tqdm working.
    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(f"Generating batch of {len(text)} sentences, sample "
                    f"{sample_idx + 1}/{num_samples}")

        prompt_length = segs.size(2)

        t0 = time.perf_counter()
        y = generate_batched(
            model=model,
            prompt=segs,
            prompt_mask=segs_mask,
            max_new_tokens=max_new_tokens,
            decode_one_token=decode_one_token,  # decode_one_token_ar
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if sample_idx == 0 and compile:
            logger.info(f"generate_batched: {time.perf_counter() - t0:.2f} seconds")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t = time.perf_counter() - t0

        tokens_generated = y.size(2) - prompt_length
        tokens_sec = tokens_generated / t
        logger.info(
            f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(
            f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
        )

        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        # Put the generated tokens
        # since there is <im_end>, we remove last token
        codes = y[:, 1:, prompt_length + 1 :].clone()
        im_end_id = tokenizer.get_token_id("<|im_end|>")

        for i, single_text in enumerate(text):
            # Ищем позицию токена окончания в сгенерированной части
            generated_tokens = y[i, 0, prompt_length + 1:]  # токены без промпта
            
            # Находим первое вхождение im_end_id
            end_positions = (generated_tokens == im_end_id).nonzero(as_tuple=True)
            
            if len(end_positions[0]) > 0:
                # Обрезаем до первого токена окончания (не включая его)
                end_pos = end_positions[0][0].item()
                trimmed_codes = codes[i, :, :end_pos]
            else:
                # Если токен окончания не найден, используем всю длину
                trimmed_codes = codes[i]
            
            yield GenerateResponse(
                action="sample",
                codes=trimmed_codes,
                text=single_text,
            )


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = load_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


def launch_thread_safe_queue_agent(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    config = BaseModelArgs.from_pretrained(checkpoint_path)

    def worker():
        model, decode_one_token = load_model(
            checkpoint_path, device, precision, compile=compile, is_agent=True
        )

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for token in generate_agent(
                    model=model,
                    decode_one_token=decode_one_token,
                    **kwargs,
                ):
                    response_queue.put(token)

                response_queue.put("stop")
            except Exception as e:
                import traceback

                logger.exception(f"Error in worker: {traceback.format_exc()}")
                response_queue.put("error")

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue, tokenizer, config


@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.7)
@click.option("--repetition-penalty", type=float, default=1.2)
@click.option("--temperature", type=float, default=0.7)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/fish-speech-1.5",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=100)
@click.option("--output-dir", type=Path, default="temp")
def main(
    text: str,
    prompt_text: Optional[list[str]],
    prompt_tokens: Optional[list[Path]],
    num_samples: int,
    max_new_tokens: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
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
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    if prompt_tokens is not None:
        prompt_tokens = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
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
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")
            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


def batch_generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: list[str],
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    compile: bool = False,
    chunk_length: int = 150,
    iterative_prompt: bool = True,
    prompt_text: Optional[list[str]] = None,
    prompt_tokens: Optional[list[torch.Tensor]] = None,
    **kwargs,
):
    logger.info("batch_generate_long")
    """
    Batch streaming inference с поддержкой чанков и стриминга по чанкам.
    Для каждого текста из батча разбивает на чанки и стримит коды по мере генерации.
    Возвращает: GenerateResponse(action, codes, text, batch_idx, chunk_idx)
    """
    assert isinstance(text, list), "text должен быть списком строк"
    if prompt_text is not None and isinstance(prompt_text, str):
        prompt_text = [prompt_text] * len(text)
    if prompt_tokens is not None and isinstance(prompt_tokens, torch.Tensor):
        prompt_tokens = [prompt_tokens] * len(text)

    # 1. Разбиваем все тексты на чанки
    all_chunks = []  # list of list[str]
    for t in text:
        if iterative_prompt:
            chunks = split_text(t, chunk_length)
        else:
            chunks = [t]
        all_chunks.append(chunks)
    num_chunks = [len(chunks) for chunks in all_chunks]
    max_chunks = max(num_chunks)

    # 2. Для каждого текста — подготовить промпты
    prompt_texts = prompt_text if prompt_text is not None else [None] * len(text)
    prompt_tokens_list = prompt_tokens if prompt_tokens is not None else [None] * len(text)

    # 3. Для каждого чанка каждого текста — поддерживать прогресс
    chunk_ptrs = [0 for _ in text]  # указатель на текущий чанк для каждого текста
    finished = [False for _ in text]

    for sample_idx in range(num_samples):
        # пока есть непросмотренные чанки — генерируем и стримим их
        while True:
            # Собираем батч из непросмотренных чанков
            batch_texts = []
            batch_prompt_texts = []
            batch_prompt_tokens = []
            batch_indices = []
            batch_chunk_indices = []
            
            for i, (chunks, ptr, done) in enumerate(zip(all_chunks, chunk_ptrs, finished)):
                if not done and ptr < len(chunks):
                    batch_texts.append(chunks[ptr])
                    batch_prompt_texts.append(prompt_texts[i])
                    batch_prompt_tokens.append(prompt_tokens_list[i])
                    batch_indices.append(i)
                    batch_chunk_indices.append(ptr)
            
            if not batch_texts:
                break  # Все чанки обработаны
            
            # ИСПРАВЛЕНИЕ: Дополняем батч до фиксированного размера
            target_batch_size = len(text)  # Оригинальный размер
            while len(batch_texts) < target_batch_size:
                # Дублируем последний элемент
                batch_texts.append(batch_texts[-1])
                batch_prompt_texts.append(batch_prompt_texts[-1])
                batch_prompt_tokens.append(batch_prompt_tokens[-1])
                batch_indices.append(-1)  # Маркер для игнорирования
                batch_chunk_indices.append(-1)
            
            # Генерируем текущий батч
            responses = batch_inference(
                model=model,
                device=device,
                decode_one_token=decode_one_token,
                text=batch_texts,
                num_samples=1,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                compile=compile,
                prompt_text=batch_prompt_texts,
                prompt_tokens=batch_prompt_tokens,
            )
            
            # Обрабатываем ответы
            response_counter = 0
            for resp in responses:
                if resp.action == "sample":
                    idx = batch_indices[response_counter]
                    chk = batch_chunk_indices[response_counter]
                    
                    yield GenerateResponse(
                        action="sample",
                        codes=resp.codes,
                        text=resp.text,
                        batch_idx=idx,
                        chunk_idx=chk,
                    )
                    
                    # Обновляем указатели
                    chunk_ptrs[idx] += 1
                    if chunk_ptrs[idx] >= num_chunks[idx]:
                        finished[idx] = True
                    
                    response_counter += 1
        
        # Сигнал завершения sample
        for i in range(len(text)):
            yield GenerateResponse(action="next", batch_idx=i, chunk_idx=None)


if __name__ == "__main__":
    main()
