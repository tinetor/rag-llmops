
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_llm(config: dict):
    """Load the language model with optional bitsandbytes quantization."""
    generation_config = config["generation"]
    model_name = generation_config["model_name"]
    quantization = generation_config.get("quantization", "none").lower()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantization == "none":
        # No quantization: load weights in FP16.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

    elif quantization == "bf16":
        # No quantization: load weights in BF16.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
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

    print(f"Model: {model_name}")
    print(f"Quantization: {quantization}")

    return model, tokenizer
