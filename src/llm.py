import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _is_vllm(model) -> bool:
    """True if the model is a vllm.LLM, checked without importing vLLM."""
    return type(model).__module__.startswith("vllm")


def get_model_name(model) -> str:
    """Return the model name for both HF and vLLM backends."""
    if _is_vllm(model):
        return model.model_config.model
    return getattr(model.config, "_name_or_path", "unknown")


def _load_hf(model_name: str, quantization: str):
    """Load the model with Hugging Face transformers."""
    if quantization == "none":
        # No quantization: load weights in FP16.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    elif quantization == "bf16":
        # No quantization: load weights in BF16.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    elif quantization == "int8":
        # Actual 8-bit quantization with bitsandbytes.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            ),
            device_map="auto",
        )

    elif quantization == "int4":
        # Actual 4-bit NF4 quantization with bitsandbytes.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
        )

    else:
        raise ValueError(
            f"Unsupported quantization '{quantization}'. "
            "Choose 'none', 'bf16', 'int8', or 'int4'."
        )

    model.eval()
    return model


def _load_vllm(model_name: str, quantization: str, generation_config: dict):
    """Load the model with vLLM."""
    # Imported lazily so the HF backend (and the tests) do not need vLLM.
    from vllm import LLM

    llm_kwargs = {
        "model": model_name,
        "gpu_memory_utilization": generation_config.get(
            "gpu_memory_utilization", 0.9
        ),
    }

    max_model_len = generation_config.get("max_model_len")
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    if quantization == "none":
        llm_kwargs["dtype"] = "float16"

    elif quantization == "bf16":
        llm_kwargs["dtype"] = "bfloat16"

    elif quantization == "int4":
        # In-flight 4-bit quantization with bitsandbytes.
        llm_kwargs["dtype"] = "float16"
        llm_kwargs["quantization"] = "bitsandbytes"

    elif quantization == "int8":
        raise ValueError(
            "In-flight int8 quantization is not supported with the vLLM "
            "backend. Use 'int4', 'none', 'bf16', or backend 'hf'."
        )

    else:
        raise ValueError(
            f"Unsupported quantization '{quantization}'. "
            "Choose 'none', 'bf16', 'int8', or 'int4'."
        )

    return LLM(**llm_kwargs)


def load_llm(config: dict):
    """Load the language model with the configured backend and quantization."""
    generation_config = config["generation"]
    model_name = generation_config["model_name"]
    quantization = generation_config.get("quantization", "none").lower()
    backend = generation_config.get("backend", "hf").lower()

    if backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = _load_hf(model_name, quantization)

    elif backend == "vllm":
        model = _load_vllm(model_name, quantization, generation_config)
        tokenizer = model.get_tokenizer()

    else:
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose 'hf' or 'vllm'."
        )

    print(f"Model: {model_name}")
    print(f"Backend: {backend}")
    print(f"Quantization: {quantization}")

    return model, tokenizer


def _generate_hf(model, tokenizer, chat: str, max_new_tokens: int):
    inputs = tokenizer(chat, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer, input_length, len(new_tokens)


def _generate_vllm(model, tokenizer, chat: str, max_new_tokens: int):
    from vllm import SamplingParams

    # temperature=0 means greedy decoding, same as do_sample=False in HF.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    request_output = model.generate(
        [chat],
        sampling_params,
        use_tqdm=False,
    )[0]
    completion = request_output.outputs[0]

    # prompt_token_ids can be None in some cases; count with the tokenizer.
    prompt_token_ids = request_output.prompt_token_ids
    if prompt_token_ids is None:
        prompt_token_ids = tokenizer(chat)["input_ids"]

    return (
        completion.text.strip(),
        len(prompt_token_ids),
        len(completion.token_ids),
    )


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    """Apply the chat template, run greedy generation and decode new tokens."""
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    if _is_vllm(model):
        answer, input_tokens, output_tokens = _generate_vllm(
            model, tokenizer, chat, max_new_tokens
        )
    else:
        answer, input_tokens, output_tokens = _generate_hf(
            model, tokenizer, chat, max_new_tokens
        )

    return {
        "chat": chat,
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
