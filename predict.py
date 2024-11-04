# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cpu"


def create_model() -> AutoModelForCausalLM:
    """create the code completion model"""
    checkpoint = "bigcode/tiny_starcoder_py"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id if not set

    model = AutoModelForCausalLM.from_pretrained(checkpoint, max_length=300).to(device)
    return tokenizer, model


def get_prediction(
    input_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: 80,
) -> str:
    """get the code completion for a given input text"""
    inputs = tokenizer.encode(input_text, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0])
