"""OpenAI-compatible endpoints on the existing FastAPI app (loopback only, like the rest of the API).

    GET  /v1/models
    POST /v1/completions        prompt (text, token ids, or a batch of either), echo, logprobs (0-5),
                                stop, seed, temperature, top_p, top_k, max_tokens (0-400), SSE streaming
    POST /v1/chat/completions   a plain-text script template (the models are NOT instruction-tuned)
    POST /tokenize, POST /detokenize, GET /tokenizer_info    (vLLM-style; EleutherAI lm-evaluation-harness's
                                ``local-completions`` model uses these as its remote tokenizer)

``echo=True`` + ``logprobs`` + ``max_tokens=1`` is what the harness sends for loglikelihood tasks:
``token_logprobs`` then holds log p(token_i | tokens_<i) for every prompt token after the first (the first
is ``null``), followed by the generated token. All logprobs are natural-log probabilities of the
untempered model distribution. They are model scores, not calibrated confidence.

Requests are deterministic: the sampler is a CPU ``torch.Generator`` seeded with ``seed`` (0 when absent,
which differs from OpenAI, where omitting it means "random").
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any, Literal

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, Field

from morrowglass.inference.generate import MAX_NEW_TOKENS, _filtered_probs
from morrowglass.serving.backends import Backend

MAX_PROMPT_CHARS = 20_000
MAX_BATCH = 32
NOT_INSTRUCTION_TUNED = (
    "This model is not instruction-tuned. It continues text in the style it was trained on; the chat endpoint "
    "only wraps the conversation in a plain-text script template, so replies are pastiche, not answers."
)
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] | upper }}.\n{{ m['content'] }}\n\n{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT.\n{% endif %}"
)
CHAT_DEFAULT_STOP = ["\n\n"]
_jinja = ImmutableSandboxedEnvironment(keep_trailing_newline=True)


def render_chat(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    return _jinja.from_string(CHAT_TEMPLATE).render(messages=messages, add_generation_prompt=add_generation_prompt)


class APIError(Exception):
    def __init__(self, status: int, message: str, param: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status, self.message, self.param, self.code = status, message, param, code

    def response(self) -> JSONResponse:
        kind = "invalid_request_error" if self.status < 500 else "server_error"
        return JSONResponse({"error": {"message": self.message, "type": kind, "param": self.param,
                                       "code": self.code}}, status_code=self.status)


# ------------------------------------------------------------------------------------------------ schemas


class _Sampling(BaseModel):
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=0, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=50_000)  # extension (not in OpenAI's schema)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class CompletionRequest(_Sampling):
    prompt: str | list[str] | list[int] | list[list[int]]
    logprobs: int | None = Field(default=None, ge=0, le=5)
    echo: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=MAX_PROMPT_CHARS)


class ChatRequest(_Sampling):
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)


class TokenizeRequest(BaseModel):
    prompt: str = Field(max_length=2_000_000)
    model: str | None = None
    add_special_tokens: bool = False


class DetokenizeRequest(BaseModel):
    tokens: list[int]
    model: str | None = None


# ------------------------------------------------------------------------------------------------ core


def _stops(stop: str | list[str] | None) -> list[str]:
    s = [stop] if isinstance(stop, str) else list(stop or [])
    if len(s) > 4 or any(not x for x in s):
        raise APIError(400, "stop: up to 4 non-empty strings", "stop")
    return s


def _check_common(req: _Sampling) -> None:
    if req.presence_penalty or req.frequency_penalty:
        raise APIError(400, "presence/frequency penalties are not implemented", "presence_penalty")


def _sample_probs(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float) -> torch.Tensor:
    p = _filtered_probs(logits, temperature, top_k)
    if temperature > 0 and top_p < 1.0:
        sp, si = torch.sort(p, descending=True)
        keep = torch.cumsum(sp, 0) - sp < top_p  # smallest prefix with mass >= top_p (always keeps the top token)
        mask = torch.zeros_like(p, dtype=torch.bool)
        mask[si[keep]] = True
        p = torch.where(mask, p, torch.zeros_like(p))
        p = p / p.sum()
    return p


def _top_dict(backend: Backend, row: torch.Tensor, k: int) -> dict[str, float]:
    out: dict[str, float] = {}
    if k <= 0:
        return out
    vals, idx = torch.topk(row, k)
    for v, i in zip(vals.tolist(), idx.tolist(), strict=True):
        out.setdefault(backend.decode([int(i)]), float(v))  # never let a colliding string overwrite a higher value
    return out


def _prompt_ids(backend: Backend, prompt: str | list[int]) -> list[int]:
    if isinstance(prompt, str):
        if len(prompt) > MAX_PROMPT_CHARS:
            raise APIError(400, f"prompt longer than {MAX_PROMPT_CHARS} characters", "prompt")
        ids = backend.encode(prompt)
    else:
        ids = [int(t) for t in prompt]
        if any(t < 0 or t >= backend.vocab_size for t in ids):
            raise APIError(400, f"token ids must be in [0, {backend.vocab_size})", "prompt")
    return ids


def _split_prompts(prompt: str | list[str] | list[int] | list[list[int]]) -> list[str | list[int]]:
    if isinstance(prompt, str):
        return [prompt]
    if not prompt:
        raise APIError(400, "empty prompt list", "prompt")
    if all(isinstance(p, int) for p in prompt):
        return [list(prompt)]  # type: ignore[arg-type]
    items: list[str | list[int]] = [p for p in prompt if isinstance(p, str | list)]
    if len(items) != len(prompt) or len(items) > MAX_BATCH:
        raise APIError(400, f"prompt: a string, token ids, or a list of up to {MAX_BATCH} of either", "prompt")
    return items


class _Gen:
    """One sampled continuation: token ids, raw logprob rows, text, finish reason."""

    def __init__(self, backend: Backend, ids: list[int], max_tokens: int, req: _Sampling, stops: list[str]) -> None:
        self.backend, self.ids, self.max_tokens, self.req, self.stops = backend, ids, max_tokens, req, stops
        self.out: list[int] = []
        self.rows: list[torch.Tensor] = []
        self.finish = "length"

    def tokens(self) -> Iterator[int]:
        """Yield new token ids. Holds the backend lock for the whole generation."""
        b, req = self.backend, self.req
        gen = torch.Generator().manual_seed(req.seed if req.seed is not None else 0)
        stepper = b.stepper()
        seq = list(self.ids) or b.encode("\n")
        for _ in range(self.max_tokens):
            logits = stepper.next_logits(seq).float().cpu()
            p = _sample_probs(logits, req.temperature, req.top_k, req.top_p)
            nxt = int(torch.multinomial(p, 1, generator=gen)) if req.temperature > 0 else int(p.argmax())
            self.rows.append(torch.log_softmax(logits, -1))
            self.out.append(nxt)
            seq.append(nxt)
            yield nxt
            if self.stops and self.stop_index(b.decode(self.out)) is not None:
                self.finish = "stop"
                return

    def stop_index(self, text: str) -> int | None:
        hits = [i for i in (text.find(s) for s in self.stops) if i >= 0]
        return min(hits) if hits else None


def _offsets(backend: Backend, ids: list[int], start: int) -> list[int]:
    """Character offset of each token in the decoded text (decode of the prefix, so byte tokens line up)."""
    return [start + len(backend.decode(ids[:i])) for i in range(len(ids))]


def complete_one(backend: Backend, prompt: str | list[int], req: CompletionRequest, index: int) -> dict[str, Any]:
    ids = _prompt_ids(backend, prompt)
    max_tokens = 16 if req.max_tokens is None else req.max_tokens
    stops = _stops(req.stop)
    want_lp = req.logprobs is not None
    if req.echo and want_lp and len(ids) > backend.block_size:
        raise APIError(400, f"echo with logprobs scores at most {backend.block_size} prompt tokens (the model's "
                            f"context); got {len(ids)}", "prompt", "context_length_exceeded")
    if not ids and (req.echo or max_tokens == 0):
        raise APIError(400, "empty prompt", "prompt")
    with backend.lock:
        g = _Gen(backend, ids, max_tokens, req, stops)
        for _ in g.tokens():
            pass
        prompt_rows = backend.logprobs(ids) if (req.echo and want_lp) else None
    gen_ids, gen_rows = g.out, g.rows
    text = backend.decode(gen_ids)
    if g.finish == "stop":
        cut = g.stop_index(text)
        assert cut is not None
        text = text[:cut]
        # keep the tokens whose text starts before the cut
        starts = _offsets(backend, gen_ids, 0)
        n_keep = sum(1 for s in starts if s < cut)
        gen_ids, gen_rows = gen_ids[:n_keep], gen_rows[:n_keep]
    prompt_text = backend.decode(ids) if isinstance(prompt, list) else prompt
    full_text = (prompt_text + text) if req.echo else text
    logprobs = None
    if want_lp:
        k = int(req.logprobs or 0)
        toks: list[str] = []
        tlp: list[float | None] = []
        top: list[dict[str, float] | None] = []
        offs: list[int] = []
        if req.echo:
            toks += [backend.decode([t]) for t in ids]
            offs += _offsets(backend, ids, 0)
            assert prompt_rows is not None
            tlp.append(None)
            top.append(None)
            for i in range(1, len(ids)):
                row = prompt_rows[i - 1]
                tlp.append(float(row[ids[i]]))
                top.append(_top_dict(backend, row, k))
        base = len(prompt_text) if req.echo else 0
        toks += [backend.decode([t]) for t in gen_ids]
        offs += _offsets(backend, gen_ids, base)
        for t, row in zip(gen_ids, gen_rows, strict=True):
            tlp.append(float(row[t]))
            top.append(_top_dict(backend, row, k))
        logprobs = {"tokens": toks, "token_logprobs": tlp, "top_logprobs": top, "text_offset": offs}
    return {"text": full_text, "index": index, "logprobs": logprobs, "finish_reason": g.finish,
            "_usage": (len(ids), len(gen_ids))}


def _envelope(kind: str, model: str, backend: Backend) -> dict[str, Any]:
    prefix = {"text_completion": "cmpl", "chat.completion": "chatcmpl"}.get(kind.removesuffix(".chunk"), "cmpl")
    return {"id": f"{prefix}-{uuid.uuid4().hex[:24]}", "object": kind, "created": int(time.time()),
            "model": model, "system_fingerprint": f"morrowglass-{backend.kind}"}


def _stream_events(backend: Backend, ids: list[int], req: _Sampling, stops: list[str], max_tokens: int,
                   env: dict[str, Any], chat: bool) -> Iterator[str]:
    """SSE: ``data: {chunk}`` per text delta, then ``data: [DONE]``. Holds back text that could still become
    a stop sequence or an incomplete UTF-8 character."""
    hold = max((len(s) for s in stops), default=1) - 1

    def chunk(delta: str | None, finish: str | None, first: bool = False) -> str:
        if chat:
            d: dict[str, Any] = {"role": "assistant", "content": ""} if first else ({"content": delta} if delta else {})
            choice: dict[str, Any] = {"index": 0, "delta": d, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": delta or "", "logprobs": None, "finish_reason": finish}
        return f"data: {json.dumps({**env, 'choices': [choice]})}\n\n"

    if chat:
        yield chunk(None, None, first=True)
    emitted = 0
    with backend.lock:
        g = _Gen(backend, ids, max_tokens, req, stops)
        for _ in g.tokens():
            text = backend.decode(g.out)
            cut = g.stop_index(text) if stops else None
            if cut is not None:
                if cut > emitted:
                    yield chunk(text[emitted:cut], None)
                emitted = cut
                break
            safe = min(len(text) - hold, len(text.rstrip("�")))
            if safe > emitted:
                yield chunk(text[emitted:safe], None)
                emitted = safe
    if g.finish != "stop":
        text = backend.decode(g.out)
        if len(text) > emitted:
            yield chunk(text[emitted:], None)
    yield chunk(None, g.finish)
    yield "data: [DONE]\n\n"


# ------------------------------------------------------------------------------------------------ router


def build_openai_router(resolve: Callable[[str | None], Backend],
                        list_models: Callable[[], list[dict[str, Any]]]) -> APIRouter:
    """``resolve(model)`` returns a backend (``None`` = the server's default model) or raises APIError."""
    r = APIRouter(tags=["openai-compatible"])

    def _resolve(model: str | None) -> Backend:
        return resolve(model)

    @r.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": list_models()}

    @r.post("/v1/completions", response_model=None)
    def completions(req: CompletionRequest) -> Any:
        try:
            _check_common(req)
            backend = _resolve(req.model)
            prompts = _split_prompts(req.prompt)
            env = _envelope("text_completion", backend.model_id, backend)
            if req.stream:
                if len(prompts) != 1 or req.echo or req.logprobs is not None:
                    raise APIError(400, "stream supports one prompt without echo/logprobs", "stream")
                ids = _prompt_ids(backend, prompts[0])
                max_tokens = 16 if req.max_tokens is None else req.max_tokens
                return StreamingResponse(_stream_events(backend, ids, req, _stops(req.stop), max_tokens,
                                                        {**env, "object": "text_completion"}, chat=False),
                                         media_type="text/event-stream")
            choices = [complete_one(backend, p, req, i) for i, p in enumerate(prompts)]
        except APIError as e:
            return e.response()
        pt = sum(c["_usage"][0] for c in choices)
        ct = sum(c.pop("_usage")[1] for c in choices)
        return {**env, "choices": choices,
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
                "morrowglass": {**backend.describe(), "logprobs": "natural log, untempered model distribution"}}

    @r.post("/v1/chat/completions", response_model=None)
    def chat(req: ChatRequest) -> Any:
        try:
            _check_common(req)
            backend = _resolve(req.model)
            prompt = render_chat([m.model_dump() for m in req.messages])
            stops = _stops(req.stop) if req.stop is not None else list(CHAT_DEFAULT_STOP)
            max_tokens = 128 if req.max_tokens is None else req.max_tokens
            ids = _prompt_ids(backend, prompt)
            env = _envelope("chat.completion", backend.model_id, backend)
            if req.stream:
                return StreamingResponse(_stream_events(backend, ids, req, stops, max_tokens,
                                                        {**env, "object": "chat.completion.chunk"}, chat=True),
                                         media_type="text/event-stream")
            creq = CompletionRequest(**{**req.model_dump(exclude={"messages", "stream", "stop", "max_tokens"}),
                                        "prompt": prompt, "max_tokens": max_tokens, "stop": stops})
            c = complete_one(backend, prompt, creq, 0)
        except APIError as e:
            return e.response()
        pt, ct = c["_usage"]
        return {**env, "choices": [{"index": 0, "message": {"role": "assistant", "content": c["text"]},
                                    "logprobs": None, "finish_reason": c["finish_reason"]}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
                "morrowglass": {**backend.describe(), "note": NOT_INSTRUCTION_TUNED, "chat_template": CHAT_TEMPLATE}}

    @r.post("/tokenize", response_model=None)
    def tokenize(req: TokenizeRequest) -> Any:
        try:
            b = _resolve(req.model)
        except APIError as e:
            return e.response()
        ids = b.encode(req.prompt)
        return {"tokens": ids, "count": len(ids), "max_model_len": b.block_size}

    @r.post("/detokenize", response_model=None)
    def detokenize(req: DetokenizeRequest) -> Any:
        try:
            b = _resolve(req.model)
            if any(t < 0 or t >= b.vocab_size for t in req.tokens):
                raise APIError(400, "token id out of range", "tokens")
        except APIError as e:
            return e.response()
        return {"prompt": b.decode(req.tokens)}

    @r.get("/tokenizer_info", response_model=None)
    def tokenizer_info(model: str | None = None) -> Any:
        try:
            b = _resolve(model)
        except APIError as e:
            return e.response()
        return {**b.tokenizer_info(), "model": b.model_id, "chat_template": CHAT_TEMPLATE}

    return r
