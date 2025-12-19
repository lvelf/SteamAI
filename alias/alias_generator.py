from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse
logger = logging.getLogger(__name__)

@dataclass
class AliasConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    device: Optional[str] = None
    max_new_tokens: int = 192
    temperature: float = 0.2
    max_input_chars: int = 6000
    use_4bit: bool = False
    hf_token: Optional[str] = None

    system_prompt: str = (
        "You are a helpful assistant for Steam game metadata enrichment. "
        "Given a game's name and description, generate short user-friendly aliases and search keywords. "
        "Return STRICT JSON only."
    )


class AliasGenerator:
    def __init__(self, config: AliasConfig):
        self.config = config
        self._load_model()

    def _load_model(self) -> None:
        cfg = self.config
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        token = cfg.hf_token or os.environ.get("HF_TOKEN")
        token_kwargs = {}
        if token is not None:
            token_kwargs["token"] = token

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **token_kwargs)

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
        )

        logger.info("Alias model loaded: %s", cfg.model_name)


    def build_user_prompt(
        self,
        name: str,
        short_description: str = "",
        genres: str = "",
        extra: str = "",
        k: int = 6,
    ) -> str:
        return f"""
Generate up to {k} aliases for a Steam game to improve search and discoverability.

Rules (follow strictly):
- Do NOT repeat the original name exactly.
- Aliases must look like real search queries typed by players.
- Use lowercase only.
- Prefer 1-4 words separated by spaces (NOT CamelCase / PascalCase / synthetic tags).
  Good examples: "portal coop", "co op puzzle", "aperture science", "p2"
  Bad examples: "SciFiPuzzle", "AwardWinningPortal", "InnovativePuzzle"
- Avoid generic marketing adjectives (e.g., "innovative", "award-winning", "best", "ultimate").
- Allowed alias types:
  1) common abbreviations / nicknames (e.g., "p2")
  2) franchise / universe terms (e.g., "aperture science")
  3) core gameplay keywords players search (e.g., "co op puzzle", "first person puzzle")
- Output STRICT JSON ONLY with this schema (no extra text, no markdown):
  {{"aliases": ["alias1", "alias2", "..."]}}

Game name: {name}
Short description: {short_description}
Genres: {genres}
Extra context: {extra}
""".strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return None

    @staticmethod
    def _looks_camel_case(s: str) -> bool:
        return bool(re.search(r"[a-z][A-Z]", s))

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = s.strip().strip('"').strip("'")
        s = s.lower()
        s = re.sub(r"[-_/]+", " ", s)              
        s = re.sub(r"[^a-z0-9\s+]", "", s)        
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _is_unrealistic_alias(s_raw: str, s_norm: str) -> bool:
        if not s_norm:
            return True

        
        if len(s_norm) > 32:
            return True
        if len(s_norm.split()) > 5:
            return True

       
        if AliasGenerator._looks_camel_case(s_raw):
            return True
        if (" " not in s_norm) and len(s_norm) >= 10:
        
            return True

       
        banned = {
            "award", "awardwinning", "award-winning",
            "innovative", "masterpiece", "ultimate", "best",
        }
        for w in banned:
            if w in s_norm:
                return True

        return False

    @staticmethod
    def _normalize_aliases(aliases: List[str], original_name: str, max_k: int = 8) -> List[str]:
        seen = set()
        out = []
        orig = (original_name or "").strip().lower()

        for a in aliases or []:
            if a is None:
                continue
            s_raw = str(a).strip()
            if not s_raw:
                continue

           
            if orig and s_raw.strip().lower() == orig:
                continue

            s_norm = AliasGenerator._normalize_text(s_raw)

            if orig and AliasGenerator._normalize_text(orig) == s_norm:
                continue

            if AliasGenerator._is_unrealistic_alias(s_raw, s_norm):
                continue

            if s_norm in seen:
                continue
            seen.add(s_norm)

            out.append(s_norm)
            if len(out) >= max_k:
                break

        return out

    def generate_aliases(
        self,
        name: str,
        short_description: str = "",
        genres: str = "",
        extra: str = "",
        k: int = 6,
    ) -> List[str]:
        cfg = self.config
        user_prompt = self.build_user_prompt(
            name=name,
            short_description=(short_description or "")[: cfg.max_input_chars],
            genres=(genres or "")[: 500],
            extra=(extra or "")[: 1000],
            k=k,
        )

        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"[SYSTEM] {cfg.system_prompt}\n[USER] {user_prompt}\n\n[ASSISTANT]"

        outputs = self.pipe(
            prompt_text,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = outputs[0]["generated_text"]
        gen = full_text[len(prompt_text):].strip() if full_text.startswith(prompt_text) else full_text.strip()

        obj = self._extract_json(gen)
        aliases = []
        if isinstance(obj, dict) and "aliases" in obj and isinstance(obj["aliases"], list):
            aliases = obj["aliases"]

        return self._normalize_aliases(aliases, original_name=name, max_k=max(6, k))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser("Local LLM alias generator test")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    cfg = AliasConfig(
        model_name=args.model,
        use_4bit=args.use_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    gen = AliasGenerator(cfg)
    
    name = "Portal 2"
    short_description = (
        "Portal 2 draws from the award-winning formula of innovative gameplay, story, and music "
        "that earned the original Portal over 70 industry accolades."
    )
    genres = "Puzzle, Co-op, Sci-fi"

    # build prompt
    user_prompt = gen.build_user_prompt(
        name=name,
        short_description=short_description,
        genres=genres,
        extra="",
        k=args.k,
    )
    print("\n========== USER PROMPT ==========\n")
    print(user_prompt)

    # run generation
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(gen.tokenizer, "apply_chat_template"):
        prompt_text = gen.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = f"[SYSTEM] {cfg.system_prompt}\n[USER] {user_prompt}\n\n[ASSISTANT]"

    outputs = gen.pipe(
        prompt_text,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        temperature=cfg.temperature,
        eos_token_id=gen.tokenizer.eos_token_id,
        pad_token_id=gen.tokenizer.eos_token_id,
    )
    full_text = outputs[0]["generated_text"]
    gen_text = full_text[len(prompt_text):].strip() if full_text.startswith(prompt_text) else full_text.strip()

    
    print(gen_text)

    obj = gen._extract_json(gen_text)
    
    print(obj)

    aliases = []
    if isinstance(obj, dict) and isinstance(obj.get("aliases"), list):
        aliases = gen._normalize_aliases(obj["aliases"], original_name=name, max_k=max(args.k, 6))

    print("\n========== FINAL ALIASES ==========\n")
    for i, a in enumerate(aliases, 1):
        print(f"{i}. {a}")

    # show the normal API output
    aliases2 = gen.generate_aliases(
        name=name,
        short_description=short_description,
        genres=genres,
        extra="",
        k=args.k,
    )
    print("\n========== generate_aliases() RESULT ==========\n")
    print(aliases2)