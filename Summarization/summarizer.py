from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

@dataclass
class SummarizerConfig:
    """
    config LLM summarizer params
    """
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"  
    device: Optional[str] = None          
    max_new_tokens: int = 512
    temperature: float = 0.4
    max_input_chars: int = 12000          
    use_4bit: bool = False           

    # system prompt 可以根据效果微调
    system_prompt: str = (
        "You are a helpful assistant that summarizes Steam user reviews for video games. "
        "You write concise, neutral summaries with clear pros and cons."
    )
    

class ReviewSummarizer:
    def __init__(self, config: SummarizerConfig):
        self.config = config
        self._load_model()
        
    def _load_model(self) -> None:
        cfg = self.config

        if cfg.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
            
        
        logger.info("Loading model %s on device=%s (4bit=%s)",
                    cfg.model_name, device, cfg.use_4bit)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_auth_token=True,
        )
        
        # model
        if cfg.use_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
        )

        logger.info("Model loaded.")
    
    def build_user_prompt(self, app_name: str, reviews_joined: str) -> str:
        return f"""
You are summarizing Steam user reviews for a video game.

Game: {app_name}

Based on the following user reviews, write a structured summary with:

1. A short overview paragraph about how players generally feel.
2. A bullet list of Pros (each item starting with "- ").
3. A bullet list of Cons (each item starting with "- ").
4. One sentence about what kind of players would enjoy this game.

Keep the summary concise and neutral. Do not copy long sentences verbatim from the reviews.

User reviews:
{reviews_joined}
""".strip()

    def summarize_reviews(self, app_name: str, texts: List[str]) -> str:
        if not texts:
            return "No reviews available."
        
        joined = "\n\n".join(texts)
        if len(joined) > self.config.max_input_chars:
            joined = joined[: self.config.max_input_chars]

        user_content = self.build_user_prompt(app_name, joined)
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
           
            prompt_text = (
                f"[SYSTEM] {self.config.system_prompt}\n"
                f"[USER] {user_content}\n\n[ASSISTANT]"
            )
            
        
        outputs = self.pipe(
            prompt_text,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            temperature=self.config.temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = outputs[0]["generated_text"]
        
        if full_text.startswith(prompt_text):
            summary = full_text[len(prompt_text):].strip()
        else:
            summary = full_text.strip()

        return summary
