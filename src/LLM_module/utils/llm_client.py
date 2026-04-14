from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from io import BytesIO
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
for p in map(str, (SRC, ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
from LLM_module import eval_config as ecfg


_DNS_FALLBACK_CACHE: Dict[str, str] = {}


def _get_dns_fallback_nameservers() -> list[str]:
    """Nameservers for user-space DNS fallback.

    Override with env var LLM_DNS_FALLBACK_NAMESERVERS="ip1,ip2".
    """
    import os

    raw = str(os.environ.get("LLM_DNS_FALLBACK_NAMESERVERS", "") or "").strip()
    if raw:
        items = [x.strip() for x in raw.split(",") if x.strip()]
        if items:
            return items
    return ["1.1.1.1", "8.8.8.8"]


def _resolve_host_via_public_dns(hostname: str, *, timeout_s: float = 2.0) -> Optional[str]:
    """Best-effort resolve without relying on system resolver.

    Uses dnspython (if installed) to query public resolvers. Returns one IPv4.
    """
    cached = _DNS_FALLBACK_CACHE.get(hostname)
    if cached:
        return cached

    try:
        import dns.resolver  # type: ignore
    except Exception:
        return None

    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = _get_dns_fallback_nameservers()
    resolver.lifetime = timeout_s
    resolver.timeout = min(timeout_s, 2.0)
    try:
        ans = resolver.resolve(hostname, "A")
        ip = str(ans[0])
        _DNS_FALLBACK_CACHE[hostname] = ip
        return ip
    except Exception:
        return None


def _https_post_via_ip(
    *,
    url: str,
    body: bytes,
    headers: Dict[str, str],
    timeout_s: float,
    ip: str,
) -> bytes:
    """HTTPS POST to an IP while preserving Host + SNI for TLS verification."""
    import http.client
    import socket
    import ssl

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"DNS fallback only supports https URLs, got: {url!r}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid URL (no hostname): {url!r}")
    port = parsed.port or 443
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED

    class _SNIHTTPSConnection(http.client.HTTPSConnection):
        def __init__(self, *, ip_addr: str, sni_host: str):
            super().__init__(host=ip_addr, port=port, timeout=timeout_s, context=context)
            self._sni_host = sni_host

        def connect(self):  # type: ignore[override]
            sock = socket.create_connection((self.host, self.port), self.timeout, self.source_address)
            self.sock = self._context.wrap_socket(sock, server_hostname=self._sni_host)

    conn = _SNIHTTPSConnection(ip_addr=ip, sni_host=hostname)
    try:
        # Preserve original hostname for HTTP routing on the server side.
        req_headers = dict(headers)
        req_headers["Host"] = hostname
        conn.request("POST", path, body=body, headers=req_headers)
        resp = conn.getresponse()
        raw = resp.read()
        status = int(getattr(resp, "status", 0) or 0)
        if 200 <= status < 300:
            return raw
        # Raise an urllib-compatible HTTPError so existing retry logic works.
        from urllib.error import HTTPError

        raise HTTPError(url, status, getattr(resp, "reason", ""), dict(getattr(resp, "headers", {}) or {}), BytesIO(raw))
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage
    raw: Any = None


class LLMClientProtocol(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        ...


class AigcBestChatClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.base_url = (base_url or getattr(ecfg, "AIGC_BEST_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or getattr(ecfg, "AIGC_BEST_API_KEY", "")
        self.model = model or ecfg.MODEL
        self.system_prompt = system_prompt or ecfg.SYSTEM_PROMPT

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        import json, time, random, socket, ssl
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError

        if not str(self.api_key or "").strip():
            raise RuntimeError(
                "Missing API key for AigcBestChatClient. "
                "Set env var AIGC_BEST_API_KEY (or OPENAI_API_KEY) before running evaluation."
            )
        
        model_id = (model or self.model).split("/")[-1]
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": str(prompt or "")}
            ],
            "temperature": temperature if temperature is not None else ecfg.TEMPERATURE,
            "top_p": top_p if top_p is not None else ecfg.TOP_P,
        }
        if (max_tokens if max_tokens is not None else ecfg.MAX_TOKENS) is not None:
            payload["max_tokens"] = max_tokens if max_tokens is not None else ecfg.MAX_TOKENS
        
        headers = {**ecfg.LLM_HTTP_HEADERS_BASE, "Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        for attempt in range(1, ecfg.LLM_MAX_RETRY + 1):
            try:
                body_bytes = json.dumps(payload).encode("utf-8")
                with urlopen(
                    Request(url, data=body_bytes, headers=headers, method="POST"),
                    timeout=ecfg.LLM_REQUEST_TIMEOUT_S,
                ) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                content = str(obj["choices"][0]["message"]["content"])
                # Treat empty content as a retriable failure (it breaks downstream scoring/CSV).
                if not content.strip():
                    raise RuntimeError("LLM returned empty content")

                return LLMResponse(
                    text=content,
                    model=str(payload["model"]),
                    usage=LLMUsage(
                        prompt_tokens=(usage_obj := obj.get("usage", {})).get("prompt_tokens"),
                        completion_tokens=usage_obj.get("completion_tokens"),
                        total_tokens=usage_obj.get("total_tokens"),
                    ),
                    raw=obj,
                )
            except HTTPError as e:
                # urllib.error.HTTPError is also a file-like object that may contain
                # a JSON error payload from the provider. Surfacing it makes 400s
                # (e.g., wrong base_url, invalid model, token limits) much easier to debug.
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""

                if e.code in ecfg.LLM_RETRIABLE_HTTP_STATUS and attempt < ecfg.LLM_MAX_RETRY:
                    sleep_time = 2.0 + random.uniform(0, 1)
                    extra = f" Body: {err_body[:500]}" if err_body else ""
                    print(
                        f"[LLM Retry {attempt}/{ecfg.LLM_MAX_RETRY}] HTTP {e.code}."
                        f" Retrying in {sleep_time:.2f}s...{extra}"
                    )
                    time.sleep(sleep_time)
                    continue

                msg = f"LLM HTTP {e.code}: {getattr(e, 'reason', '')}".strip()
                if err_body:
                    msg = f"{msg}\nProvider response body: {err_body}"
                raise RuntimeError(msg) from e
            except URLError as e:
                # Special-case DNS failure: attempt user-space resolution without admin access.
                reason = getattr(e, "reason", None)
                if isinstance(reason, socket.gaierror):
                    host = urlparse(url).hostname or ""
                    ip = _resolve_host_via_public_dns(host, timeout_s=2.0) if host else None
                    if ip:
                        try:
                            raw = _https_post_via_ip(
                                url=url,
                                body=json.dumps(payload).encode("utf-8"),
                                headers=headers,
                                timeout_s=float(ecfg.LLM_REQUEST_TIMEOUT_S),
                                ip=ip,
                            )
                            obj = json.loads(raw.decode("utf-8"))
                            content = str(obj["choices"][0]["message"]["content"])
                            if not content.strip():
                                raise RuntimeError("LLM returned empty content")
                            return LLMResponse(
                                text=content,
                                model=str(payload["model"]),
                                usage=LLMUsage(
                                    prompt_tokens=(usage_obj := obj.get("usage", {})).get("prompt_tokens"),
                                    completion_tokens=usage_obj.get("completion_tokens"),
                                    total_tokens=usage_obj.get("total_tokens"),
                                ),
                                raw=obj,
                            )
                        except HTTPError:
                            # Let the HTTPError branch handle retries and surfacing body.
                            raise
                        except Exception:
                            # Fall back to normal retry behavior below.
                            pass

                if attempt < ecfg.LLM_MAX_RETRY:
                    sleep_time = 2.0 + random.uniform(0, 1)
                    print(
                        f"[LLM Retry {attempt}/{ecfg.LLM_MAX_RETRY}] Network error. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    continue
                raise
            except Exception as e:
                if attempt < ecfg.LLM_MAX_RETRY:
                    sleep_time = 3.0 + random.uniform(0, 1)
                    err_type = "SSL/timeout" if any(x in str(e).lower() for x in ["handshake", "timed out"]) or isinstance(e, (TimeoutError, socket.timeout)) else "Network"
                    print(f"[LLM Retry {attempt}/{ecfg.LLM_MAX_RETRY}] {err_type} error. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue
                raise


def get_default_client() -> LLMClientProtocol:
    provider = str(getattr(ecfg, "LLM_PROVIDER", "aigcbest") or "aigcbest").strip().lower()
    if provider == "local":
        return LocalHFChatClient()
    if provider in {"aigc", "aigcbest", "api2"}:
        return AigcBestChatClient()
    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}. Supported: 'local', 'aigcbest' (aka 'aigc'/'api2')."
    )


class LocalHFChatClient:
    """Run inference against a local HuggingFace Transformers checkpoint."""

    _shared_model = None
    _shared_tokenizer = None
    _shared_path: Optional[str] = None

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model_path = model_path or getattr(ecfg, "LOCAL_MODEL_PATH", None)
        if not self.model_path:
            raise ValueError(
                "LocalHFChatClient requires a model_path. "
                "Set LOCAL_MODEL_PATH in eval_config or pass model_path= explicitly."
            )
        self.model_name = model or ecfg.MODEL
        self.system_prompt = system_prompt or ecfg.SYSTEM_PROMPT
        self._ensure_loaded()

    # ---- lazy singleton: load weights once across all instances ----
    def _ensure_loaded(self) -> None:
        if (
            LocalHFChatClient._shared_model is not None
            and LocalHFChatClient._shared_path == self.model_path
        ):
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import os

        print(f"[LocalHF] Loading model from {self.model_path} …")
        # Allow overriding device placement for local inference.
        # - "auto" (default): let HF/accelerate decide (usually GPU if available)
        # - "cpu": force CPU (avoids GPU watchdog/launch-timeout issues)
        device_map = str(getattr(ecfg, "LOCAL_MODEL_DEVICE", "auto") or "auto").strip().lower()
        if device_map not in {"auto", "cpu", "cuda"}:
            device_map = "auto"

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        def _load(*, device_map_arg: str):
            use_cuda = device_map_arg in {"auto", "cuda"} and torch.cuda.is_available()
            dtype = torch.bfloat16 if use_cuda else torch.float32
            return AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
                device_map=("auto" if device_map_arg == "auto" else device_map_arg),
            )

        try:
            model = _load(device_map_arg=device_map)
        except Exception as e:
            msg = str(e).lower()
            cuda_timeout = "launch timed out" in msg or "cudaerrorlaunchtimeout" in msg
            cuda_related = cuda_timeout or "cuda" in msg or "torch.acceleratorerror" in msg
            if device_map != "cpu" and cuda_related:
                print(f"[LocalHF] WARNING: CUDA load failed. Falling back to CPU.")
                # Best-effort cleanup before retry.
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
                model = _load(device_map_arg="cpu")
            else:
                raise
        model.eval()
        LocalHFChatClient._shared_model = model
        LocalHFChatClient._shared_tokenizer = tokenizer
        LocalHFChatClient._shared_path = self.model_path
        print(f"[LocalHF] Model loaded on {model.device if hasattr(model, 'device') else 'auto'}")

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        import torch

        tokenizer = LocalHFChatClient._shared_tokenizer
        hf_model = LocalHFChatClient._shared_model

        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": str(prompt or "")},
        ]

        # Use chat template if available, otherwise plain concatenation
        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            input_text = (
                f"<|system|>\n{messages[0]['content']}\n"
                f"<|user|>\n{messages[1]['content']}\n<|assistant|>\n"
            )

        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(hf_model.device)
        prompt_len = input_ids.shape[-1]

        temp = temperature if temperature is not None else ecfg.TEMPERATURE
        tp = top_p if top_p is not None else ecfg.TOP_P
        gen_max = max_tokens if max_tokens is not None else (ecfg.MAX_TOKENS or 4096)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": gen_max,
            "do_sample": temp > 0,
        }
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = tp

        with torch.no_grad():
            output_ids = hf_model.generate(input_ids, **gen_kwargs)

        new_ids = output_ids[0][prompt_len:]
        content = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Strip <think>...</think> reasoning blocks from DeepSeek-R1 output
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        if not content:
            raise RuntimeError("Local model returned empty content")

        return LLMResponse(
            text=content,
            model=model or self.model_name,
            usage=LLMUsage(
                prompt_tokens=prompt_len,
                completion_tokens=len(new_ids),
                total_tokens=prompt_len + len(new_ids),
            ),
        )


