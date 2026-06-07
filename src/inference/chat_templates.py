"""Single-turn chat template formatting for instruct GGUF models.

llama.cpp completion tokenizes the prompt with special=True, so the literal
special-token strings below map to their registered special token ids.
BOS is omitted: llama.cpp adds it automatically (add_bos), so emitting it here
would double it.
"""

from __future__ import annotations


def format_chat(prompt: str, template: str | None) -> str:
    """Wrap a single user prompt in the model's chat template.

    Unknown/None templates pass the prompt through unchanged.
    """
    if template == "chatml":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if template == "llama3":
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if template == "phi3":
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    return prompt
