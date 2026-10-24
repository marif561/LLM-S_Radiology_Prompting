"""Inference-only generation with 4-bit NF4 quantisation and greedy decoding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    load_in_4bit: bool = True
    batch_size: int = 8


class HFGenerator:
    def __init__(self, model_id: str, revision: str | None = None, cfg: GenConfig = GenConfig()):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        quant = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if cfg.load_in_4bit
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, quantization_config=quant, device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.supports_system_role = self._check_system_role()

    def _check_system_role(self) -> bool:
        try:
            self.tok.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False,
            )
            return True
        except Exception:  # e.g. Mistral-Instruct templates reject the system role
            return False

    def generate(self, batch_messages: list[list[dict]]) -> list[str]:
        import torch

        prompts = [
            self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,  # greedy -> deterministic
                pad_token_id=self.tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(new, skip_special_tokens=True)
