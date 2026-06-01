from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import logging
logging.basicConfig(level=logging.INFO)

class QwenLLM:
    def __init__(self, model_name="Qwen/Qwen2-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, max_tokens=512) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length or 2048
            )

            inputs = inputs.to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.15,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            return generated_text

        except Exception as e:
            raise RuntimeError(f"Ошибка генерации: {e}")
