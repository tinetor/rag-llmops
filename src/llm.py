from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 300):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(prompt: str) -> str:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate


if __name__ == "__main__":
    llm = load_llm()
    print(llm("Say hello in one short sentence."))