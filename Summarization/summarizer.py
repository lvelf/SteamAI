from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)


@dataclass
class SummarizerConfig:
    """
    Config for local LLM summarizer.
    """
   
    #model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    device: Optional[str] = None          
    max_new_tokens: int = 512
    temperature: float = 0.4
    max_input_chars: int = 12000          # 
    use_4bit: bool = False                # 4-bit

   
    hf_token: Optional[str] = None

    # system prompt
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

    
        token = cfg.hf_token or os.environ.get("HF_TOKEN")
        token_kwargs = {}
        if token is not None:
            token_kwargs["token"] = token

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            **token_kwargs,
        )

        # model
        if cfg.use_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                device_map="auto",
                **token_kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                **token_kwargs,
            )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            #device=0 if device == "cuda" else -1,
        )

        logger.info("Model loaded.")

    # ---------- prompt ----------
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

        # apply_chat_template
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



if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    cfg = SummarizerConfig(
        model_name = "mistralai/Mistral-7B-Instruct-v0.2",
        use_4bit=False,       
        max_new_tokens=256,  
        max_input_chars=4000,
    )
    summarizer = ReviewSummarizer(cfg)

    dummy_reviews = [
        "I absolutely love this game. The puzzles are smart and the humor is great.",
        "The game is fun but sometimes the difficulty spikes feel a bit unfair.",
        "Performance is good on my laptop and co-op mode is really enjoyable.",
        "I didn't like the story that much, but the gameplay loop is addictive.",
    ]

    app_name = "Dummy Game"
    print("Running a test summary for", app_name)
    summary = summarizer.summarize_reviews(app_name, dummy_reviews)
    print("==== SUMMARY ====")
    print(summary)
