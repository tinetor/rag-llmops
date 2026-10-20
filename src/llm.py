from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm(config: dict):
    """Load the language model and tokenizer from configuration."""
    model_name = config["generation"]["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    return model, tokenizer