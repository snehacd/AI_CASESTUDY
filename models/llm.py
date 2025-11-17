"""
models/llm.py
Robust LLM wrapper: handles OpenAI SDK differences, uses supported Groq model,
and calls Google GenAI defensively.
"""

import os
from typing import Optional, Iterator

# config import (assumes config.py loads .env)
from config.config import OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY

# Lazy imports are used inside methods to avoid import-time crashes.

class LLMModel:
    def __init__(self, provider="openai", model=None, temperature=0.7):
        self.provider = (provider or "openai").lower()
        self.temperature = temperature
        # sensible defaults — use a currently supported Groq model
        self.default_models = {
            "openai": "gpt-3.5-turbo",
            "groq": "llama3-8b-4096",       # <- smaller & usually available; change if you have a specific model
            "gemini": "models/text-bison-001",
        }
        # only use provided model for the primary provider
        self.primary_model = model or self.default_models.get(self.provider)

    # --- lazy import helpers ---
    def _get_openai_client(self):
        # support both new and older openai packages
        try:
            # new style
            from openai import OpenAI as OpenAIClient
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY missing")
            return OpenAIClient(api_key=OPENAI_API_KEY)
        except Exception as e_new:
            # try legacy import fallback
            try:
                import openai
                if not getattr(openai, "api_key", None):
                    # set key for legacy SDK
                    openai.api_key = OPENAI_API_KEY
                return openai  # legacy module with ChatCompletion
            except Exception as e_legacy:
                raise ImportError(f"OpenAI import failed (new: {e_new}; legacy: {e_legacy})")

    def _get_groq_client(self):
        try:
            from groq import Groq as GroqClient
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY missing")
            return GroqClient(api_key=GROQ_API_KEY)
        except Exception as e:
            raise

    def _get_genai(self):
        try:
            import google.generativeai as genai
        except Exception as e:
            raise ImportError(f"google.generativeai import failed: {e}")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing")
        genai.configure(api_key=GEMINI_API_KEY)
        return genai

    # --- utility ---
    def _is_quota_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        indicators = ["quota", "exceeded", "billing", "payment", "rate limit", "quota_exceeded", "insufficient_quota"]
        return any(ind in msg for ind in indicators)

    # --- synchronous generate ---
    def generate_response(self, prompt, context=None, max_tokens=1000):
        enhanced_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt

        # providers sequence: primary then others
        seq = [self.provider] + [p for p in ("openai","groq","gemini") if p != self.provider]
        errors = []

        for prov in seq:
            try:
                if prov == "openai":
                    client = self._get_openai_client()
                    model = self.primary_model if prov == self.provider else self.default_models["openai"]

                    # New SDK: client.chat.completions.create(...)
                    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role":"system","content":"You are a helpful AI assistant."},
                                {"role":"user","content":enhanced_prompt}
                            ],
                            temperature=self.temperature,
                            max_tokens=max_tokens
                        )
                        return resp.choices[0].message.content
                    # Legacy SDK: openai.ChatCompletion.create(...)
                    elif hasattr(client, "ChatCompletion"):
                        resp = client.ChatCompletion.create(
                            model=model,
                            messages=[
                                {"role":"system","content":"You are a helpful AI assistant."},
                                {"role":"user","content":enhanced_prompt}
                            ],
                            temperature=self.temperature,
                            max_tokens=max_tokens
                        )
                        return resp.choices[0].message["content"]
                    else:
                        raise RuntimeError("Unrecognized OpenAI client shape")

                elif prov == "groq":
                    client = self._get_groq_client()
                    model = self.primary_model if prov == self.provider else self.default_models["groq"]
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role":"system","content":"You are a helpful AI assistant."},
                            {"role":"user","content":enhanced_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=max_tokens
                    )
                    return resp.choices[0].message.content

                elif prov == "gemini":
                    genai = self._get_genai()
                    model = self.primary_model if prov == self.provider else self.default_models["gemini"]

                    # try common shapes:
                    # 1) generate_text
                    if hasattr(genai, "generate_text"):
                        r = genai.generate_text(model=model, prompt=enhanced_prompt)
                        return getattr(r, "text", str(r))

                    # 2) genai.models.generate
                    if hasattr(genai, "models") and hasattr(genai.models, "generate"):
                        r = genai.models.generate(model=model, input=enhanced_prompt)
                        if hasattr(r, "output"):
                            return "".join(item.get("content", str(item)) if isinstance(item, dict) else str(item) for item in r.output)
                        return str(r)

                    # 3) genai.responses.create
                    if hasattr(genai, "responses") and hasattr(genai.responses, "create"):
                        r = genai.responses.create(model=model, input=enhanced_prompt)
                        if hasattr(r, "output"):
                            return "".join(item.get("content", str(item)) if isinstance(item, dict) else str(item) for item in r.output)
                        return str(r)

                    raise RuntimeError("No supported call method found in installed google.generativeai SDK.")

            except Exception as e:
                errors.append((prov, e))
                # if likely quota/billing, go to next provider
                if self._is_quota_error(e):
                    print(f"[LLMModel] {prov} quota/billing error -> trying next provider: {e}")
                    continue
                print(f"[LLMModel] {prov} error -> trying next provider: {e}")
                continue

        # if we reach here, all failed
        details = " | ".join([f"{p}:{str(e)}" for p,e in errors])
        raise RuntimeError("All LLM providers failed. Details: " + details)

    # --- streaming generator (best-effort) ---
    def stream_response(self, prompt, context=None, max_tokens=1000) -> Iterator[str]:
        enhanced_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
        seq = [self.provider] + [p for p in ("openai","groq","gemini") if p != self.provider]

        for prov in seq:
            try:
                if prov == "openai":
                    client = self._get_openai_client()
                    model = self.primary_model if prov == self.provider else self.default_models["openai"]

                    # new SDK streaming
                    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                        iter_resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role":"system","content":"You are a helpful AI assistant."},
                                      {"role":"user","content":enhanced_prompt}],
                            temperature=self.temperature,
                            max_tokens=max_tokens,
                            stream=True
                        )
                        for ev in iter_resp:
                            try:
                                delta = ev.choices[0].delta
                                if hasattr(delta, "get") and delta.get("content"):
                                    yield delta.get("content")
                                elif hasattr(delta, "content"):
                                    yield delta.content
                            except Exception:
                                yield str(ev)
                        return

                    # legacy streaming not typical - skip to fallback
                    continue

                if prov == "groq":
                    client = self._get_groq_client()
                    model = self.primary_model if prov == self.provider else self.default_models["groq"]
                    iter_resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role":"system","content":"You are a helpful AI assistant."},
                                  {"role":"user","content":enhanced_prompt}],
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    for ev in iter_resp:
                        try:
                            delta = ev.choices[0].delta
                            if hasattr(delta, "content"):
                                yield delta.content
                            elif hasattr(delta, "get") and delta.get("content"):
                                yield delta.get("content")
                        except Exception:
                            yield str(ev)
                    return

                if prov == "gemini":
                    # Gemini streaming is not consistent; yield one final chunk
                    genai = self._get_genai()
                    model = self.primary_model if prov == self.provider else self.default_models["gemini"]
                    text = None
                    if hasattr(genai, "generate_text"):
                        text = getattr(genai.generate_text(model=model, prompt=enhanced_prompt), "text", None)
                    elif hasattr(genai, "models") and hasattr(genai.models, "generate"):
                        resp = genai.models.generate(model=model, input=enhanced_prompt)
                        if hasattr(resp, "output"):
                            text = "".join(item.get("content", str(item)) if isinstance(item, dict) else str(item) for item in resp.output)
                        else:
                            text = str(resp)
                    elif hasattr(genai, "responses") and hasattr(genai.responses, "create"):
                        resp = genai.responses.create(model=model, input=enhanced_prompt)
                        if hasattr(resp, "output"):
                            text = "".join(item.get("content", str(item)) if isinstance(item, dict) else str(item) for item in resp.output)
                        else:
                            text = str(resp)
                    if text is not None:
                        yield text
                        return
                    continue

            except Exception as e:
                yield f"[LLMModel] {prov} failed, trying next: {e}"
                continue

        yield "[LLMModel] All providers failed. Check API keys, billing, or network."
