"""Minimal OpenAI-compatible chat client (stdlib only).

Works with the Meta Llama API (/compat/v1), Gemini's OpenAI endpoint, SubQ, or
any other OpenAI-compatible server. Honors HTTPS_PROXY via urllib.
"""

import json
import random
import threading
import time
import urllib.error
import urllib.request


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage):
        with self.lock:
            self.calls += 1
            self.prompt_tokens += (usage or {}).get("prompt_tokens", 0) or 0
            self.completion_tokens += (usage or {}).get("completion_tokens", 0) or 0

    def take(self):
        with self.lock:
            out = {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                   "completion_tokens": self.completion_tokens}
            self.calls = self.prompt_tokens = self.completion_tokens = 0
            return out


USAGE = Usage()


class LLMError(RuntimeError):
    pass


def chat(ep, messages, temperature=0.0, max_tokens=1500, retries=5, timeout=120):
    url = ep.base_url + "/chat/completions"
    body = json.dumps({"model": ep.model, "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ep.api_key}",
               "User-Agent": "scar-tissue/0.1"}
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            USAGE.add(data.get("usage"))
            msg = data["choices"][0]["message"]
            content = msg.get("content")
            if isinstance(content, list):  # some servers return content parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content or ""
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            last = LLMError(f"{ep.label} HTTP {e.code}: {detail}")
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last = LLMError(f"{ep.label}: {e}")
        time.sleep(min(30, 2 ** i + random.random()))
    raise last


def list_models(ep):
    req = urllib.request.Request(ep.base_url + "/models", headers={"Authorization": f"Bearer {ep.api_key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    return [m.get("id") for m in data.get("data", [])]
