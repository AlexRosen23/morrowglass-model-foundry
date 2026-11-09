"""OpenAI-compatible endpoints: schema conformance, logprob maths, determinism, streaming, loopback-only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from conftest import tiny_cfg
from morrowglass.api import app as app_module
from morrowglass.api.store import Store
from morrowglass.cli import main as cli_main
from morrowglass.serving.openai_api import CHAT_TEMPLATE, render_chat
from morrowglass.train.trainer import train_from_config


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    root = tmp_path_factory.mktemp("oai")
    runs, ck = root / "runs", root / "ckpt"
    s = train_from_config(tiny_cfg(max_steps=20, experiment="unit"), quiet=True, runs_dir=runs, ckpt_root=ck)
    return runs, ck, s["run_id"]


@pytest.fixture
def client(trained: tuple[Path, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    runs, ck, rid = trained
    st = Store(sources={"measured": (runs, ck), "smoke": (tmp_path / "none", tmp_path / "none")},
               index_path=tmp_path / "idx.sqlite", experiments_dir=tmp_path / "exp")
    monkeypatch.setattr(app_module, "store", st)
    monkeypatch.setenv("MORROWGLASS_DEFAULT_MODEL", rid)
    monkeypatch.delenv("MORROWGLASS_ENABLE_HF", raising=False)
    return TestClient(app_module.app)


def _check_completion_schema(body: dict, n_choices: int = 1) -> None:
    assert body["object"] == "text_completion" and body["id"].startswith("cmpl-")
    assert isinstance(body["created"], int) and isinstance(body["model"], str)
    assert len(body["choices"]) == n_choices
    for i, c in enumerate(body["choices"]):
        assert c["index"] == i and isinstance(c["text"], str)
        assert c["finish_reason"] in ("stop", "length")
        assert set(c) >= {"text", "index", "logprobs", "finish_reason"}
    u = body["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_models_lists_runs_with_checkpoints(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == [trained[2]]
    assert body["data"][0]["object"] == "model" and body["data"][0]["morrowglass"]["instruction_tuned"] is False


def test_completion_schema_and_seed_determinism(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    req = {"model": trained[2], "prompt": "KING:", "max_tokens": 12, "temperature": 0.9, "seed": 5}
    a, b = client.post("/v1/completions", json=req).json(), client.post("/v1/completions", json=req).json()
    _check_completion_schema(a)
    assert a["choices"][0]["text"] == b["choices"][0]["text"]
    assert a["usage"]["completion_tokens"] == 12 and a["choices"][0]["finish_reason"] == "length"
    c = client.post("/v1/completions", json={**req, "seed": 6}).json()
    assert c["choices"][0]["text"] != a["choices"][0]["text"]
    assert a["morrowglass"]["kv_cache"] is True and a["morrowglass"]["instruction_tuned"] is False


def test_echo_logprobs_match_the_model(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    """What lm-evaluation-harness sends for loglikelihood: echo + logprobs + max_tokens=1, temperature 0."""
    store = app_module.store
    model, tok = store.model(trained[2])
    ids = tok.encode("To be, or not")
    body = client.post("/v1/completions", json={"model": trained[2], "prompt": ids, "max_tokens": 1, "echo": True,
                                                "logprobs": 1, "temperature": 0, "seed": 1234}).json()
    _check_completion_schema(body)
    lp = body["choices"][0]["logprobs"]
    assert len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == len(lp["text_offset"])
    assert len(lp["token_logprobs"]) == len(ids) + 1  # prompt + 1 generated token
    assert lp["token_logprobs"][0] is None and lp["top_logprobs"][0] is None
    with torch.no_grad():
        ref = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
    for i in range(1, len(ids)):
        assert lp["token_logprobs"][i] == pytest.approx(float(ref[i - 1, ids[i]]), abs=1e-5)
        assert len(lp["top_logprobs"][i]) == 1
    # Greedy generated token: its logprob is the top logprob of the last row (the harness's is_greedy check).
    assert lp["token_logprobs"][-1] == pytest.approx(max(lp["top_logprobs"][-1].values()), abs=0)
    assert lp["token_logprobs"][-1] == pytest.approx(float(ref[-1].max()), abs=1e-5)
    assert body["choices"][0]["text"].startswith("To be, or not")
    assert lp["text_offset"][:3] == [0, 1, 2]


def test_harness_style_sum_over_continuation(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    """Sum of token_logprobs[ctxlen:-1] is the continuation log-likelihood, as local-completions parses it."""
    _, tok = app_module.store.model(trained[2])
    ctx, cont = tok.encode("ROMEO:"), tok.encode(" love")
    body = client.post("/v1/completions", json={"prompt": [ctx + cont], "max_tokens": 1, "echo": True,
                                                "logprobs": 1, "temperature": 0}).json()
    lp = body["choices"][0]["logprobs"]["token_logprobs"]
    ll = sum(lp[len(ctx):-1])
    assert len(lp[len(ctx):-1]) == len(cont) and ll < 0 and math.isfinite(ll)


def test_batch_prompts_and_text_prompts(client: TestClient) -> None:
    body = client.post("/v1/completions", json={"prompt": ["A", "B", "C"], "max_tokens": 3, "temperature": 0}).json()
    _check_completion_schema(body, 3)
    assert body["usage"]["completion_tokens"] == 9


def test_max_tokens_zero_echo_and_stop(client: TestClient) -> None:
    body = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 0, "echo": True}).json()
    assert body["choices"][0]["text"] == "Hello" and body["usage"]["completion_tokens"] == 0
    base = {"prompt": "Hello", "max_tokens": 30, "temperature": 1.0, "seed": 4}
    text = client.post("/v1/completions", json=base).json()["choices"][0]["text"]
    # Pick a stop string whose first occurrence is inside the continuation, so the expected cut is known.
    picks = [(j, text[j : j + w]) for w in (2, 1) for j in range(2, len(text) - w) if text.find(text[j : j + w]) == j]
    assert picks, f"degenerate sample {text!r}"
    j, stop = picks[0]
    cut = client.post("/v1/completions", json={**base, "stop": [stop], "logprobs": 0}).json()["choices"][0]
    assert cut["text"] == text[:j] and cut["finish_reason"] == "stop"
    assert len(cut["logprobs"]["tokens"]) <= j + 1  # tokens after the cut are dropped from the logprobs too


def test_errors_are_openai_shaped(client: TestClient) -> None:
    r = client.post("/v1/completions", json={"model": "nope", "prompt": "x"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    r = client.post("/v1/completions", json={"prompt": "x", "frequency_penalty": 0.5})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r = client.post("/v1/completions", json={"prompt": [999999], "max_tokens": 1})
    assert r.status_code == 400
    r = client.post("/v1/completions", json={"prompt": "x" * 40, "echo": True, "logprobs": 1, "max_tokens": 1})
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"  # block_size is 32
    assert client.post("/v1/completions", json={"prompt": "x", "max_tokens": 5000}).status_code == 422
    assert client.post("/v1/completions", json={"prompt": "x", "n": 2}).status_code == 422


def _sse(text: str) -> list[dict]:
    events = [ln.removeprefix("data: ") for ln in text.split("\n\n") if ln.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def test_streaming_completion_equals_non_streaming(client: TestClient) -> None:
    req = {"prompt": "KING:", "max_tokens": 15, "temperature": 0.8, "seed": 3}
    whole = client.post("/v1/completions", json=req).json()["choices"][0]["text"]
    with client.stream("POST", "/v1/completions", json={**req, "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks = _sse(r.read().decode())
    assert all(c["object"] == "text_completion" for c in chunks)
    assert "".join(c["choices"][0]["text"] for c in chunks) == whole
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_chat_completion_schema_template_and_note(client: TestClient) -> None:
    msgs = [{"role": "system", "content": "Speak as a player."}, {"role": "user", "content": "Who goes there?"}]
    req = {"messages": msgs, "max_tokens": 20, "temperature": 0.7, "seed": 9}
    a, b = client.post("/v1/chat/completions", json=req).json(), client.post("/v1/chat/completions", json=req).json()
    assert a["object"] == "chat.completion" and a["id"].startswith("chatcmpl-")
    msg = a["choices"][0]["message"]
    assert msg["role"] == "assistant" and isinstance(msg["content"], str)
    assert msg["content"] == b["choices"][0]["message"]["content"]
    assert "not instruction-tuned" in a["morrowglass"]["note"]
    assert render_chat(msgs) == "SYSTEM.\nSpeak as a player.\n\nUSER.\nWho goes there?\n\nASSISTANT.\n"
    with client.stream("POST", "/v1/chat/completions", json={**req, "stream": True}) as r:
        chunks = _sse(r.read().decode())
    assert chunks[0]["object"] == "chat.completion.chunk" and chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == msg["content"]


def test_remote_tokenizer_endpoints(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    info = client.get("/tokenizer_info").json()
    assert "eos_token" in info and info["model"] == trained[2] and info["chat_template"] == CHAT_TEMPLATE
    toks = client.post("/tokenize", json={"prompt": "Ophelia", "add_special_tokens": False}).json()["tokens"]
    assert toks == list(b"Ophelia")
    assert client.post("/detokenize", json={"tokens": toks}).json()["prompt"] == "Ophelia"


def test_no_default_model_is_a_clear_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MORROWGLASS_DEFAULT_MODEL")
    r = client.get("/tokenizer_info")
    assert r.status_code == 400 and "--model" in r.json()["error"]["message"]


def test_hf_reference_is_off_unless_enabled(client: TestClient) -> None:
    r = client.post("/v1/completions", json={"model": "hf:distilgpt2", "prompt": "x"})
    assert r.status_code == 404 and "MORROWGLASS_ENABLE_HF" in r.json()["error"]["message"]


def test_serve_refuses_non_loopback_hosts(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["serve", "--host", "0.0.0.0", "--port", "5109"]) == 2
    assert "loopback-only" in capsys.readouterr().err


def test_serving_records_endpoint_matches_files(client: TestClient) -> None:
    from morrowglass.repro import REPO_ROOT, read_json

    body = client.get("/api/serving").json()
    assert set(body["registered"]) == {"kv-cache", "lm-eval-harness", "int8-dynamic"}
    for rid, rec in body["kv_cache"].items():
        assert rec == read_json(REPO_ROOT / "reports" / "serving" / "kv-cache" / f"{rid}.json")
        assert rec["provenance"] == "measured" and rec["verdict"] in ("supported", "not supported")
