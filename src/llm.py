
def get_model_name(model, model_name: str = None) -> str:
    """Return the model name for the configured backend."""
    if model_name is not None:
        return model_name

    return getattr(model.config, "_name_or_path", "unknown")


def _load_hf(model_name: str, quantization: str):
    """Load a model using Hugging Face Transformers."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_kwargs = {"device_map": "auto"}

    if quantization == "none":
        model_kwargs["torch_dtype"] = torch.float16

    elif quantization == "bf16":
        model_kwargs["torch_dtype"] = torch.bfloat16

    elif quantization == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    elif quantization == "int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    else:
        raise ValueError(
            f"Unsupported quantization '{quantization}'. "
            "Choose 'none', 'bf16', 'int8', or 'int4'."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.eval()

    return model


def _load_http(generation_config: dict):
    """Create an HTTP client for a vLLM inference server."""
    from openai import OpenAI

    return OpenAI(
        base_url=generation_config["base_url"],
        api_key=generation_config.get("api_key", "EMPTY"),
        timeout=generation_config.get("timeout", 120),
        max_retries=generation_config.get("max_retries", 2),
    )


def load_llm(config: dict):
    """Load the configured inference backend."""
    generation_config = config["generation"]

    model_name = generation_config["model_name"]
    backend = generation_config.get("backend", "hf").lower()
    quantization = generation_config.get("quantization", "none").lower()

    if backend == "hf":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = _load_hf(model_name, quantization)

    elif backend == "http":
        model = _load_http(generation_config)
        tokenizer = None

    else:
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose 'hf' or 'http'."
        )

    print(f"Model: {model_name}")
    print(f"Backend: {backend}")

    if backend == "hf":
        print(f"Quantization: {quantization}")

    return model, tokenizer


def _generate_hf(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """Generate text using Hugging Face Transformers."""
    import torch

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

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

    return {
        "chat": chat,
        "answer": answer,
        "input_tokens": input_length,
        "output_tokens": len(new_tokens),
    }


def _generate_http(
    client,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """Generate text using an HTTP inference server."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_new_tokens,
        temperature=0,
    )

    usage = response.usage

    return {
        "chat": prompt,
        "answer": response.choices[0].message.content or "",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    backend: str = "hf",
    model_name: str = None,
) -> dict:
    """Generate text using the configured inference backend."""
    if backend == "http":
        return _generate_http(
            client=model,
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    if backend == "hf":
        return _generate_hf(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    raise ValueError(
        f"Unsupported backend '{backend}'. Choose 'hf' or 'http'."
    )
