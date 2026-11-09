# Engineering decisions

| decision | chosen | rejected alternatives | why |
|---|---|---|---|
| Model implementation | native PyTorch modules, explicit attention path | `AutoModelForCausalLM`, nanoGPT copy | The goal is to show the internals. The fused SDPA kernel is used for speed, and a test proves it matches the explicit maths. |
| Headline metric | bits per UTF-8 byte | per-token perplexity | Perplexity changes meaning with the tokenizer. bpb is comparable across byte, BPE and n-gram models. |
| Data split | by whole play | random lines or chunks | Random chunks leak style, characters and repeated lines. Play-level splits test generalisation to unseen works. |
| Test protocol | locked split + logged `final-test` | evaluate test after every run | Repeated test evaluation becomes model selection on test. The log makes every opening auditable. |
| Count baseline | interpolated Kneser-Ney, orders swept on val | add-k smoothing, unigram only | A weak baseline flatters the neural model. KN is the classic strong n-gram. |
| Uncertainty | bootstrap over eval windows + multi-seed ranges | single numbers | Small differences (about 0.02 bpb) are within noise; showing the interval prevents over-claiming. |
| Experiment tracking | JSON/JSONL files in git + derived SQLite index | MLflow / W&B | Zero services, diffable, reviewable in a PR, and survives without a server. MLflow would add a dependency without adding evidence. |
| Experiment design | registry with hypothesis + criterion written before running | ad-hoc notebooks | Prevents moving the goalposts after seeing results. |
| Checkpoint integrity | sha256 + verified atomic writes + `weights_only=True` | plain `torch.save` | Known memory fault on the dev machine; unsafe pickle loading. |
| Tokenizer library | HF `tokenizers` byte-level BPE | sentencepiece, a hand-written BPE | Fast, lossless, and familiar to reviewers. Training is on train text only. |
| Device | MPS if a real op runs, else CPU; no CUDA path | assume CUDA | The target hardware is Apple silicon. CPU keeps CI and smoke runs portable. |
| Positional encoding | learned absolute (default) + sinusoidal/none options | RoPE, ALiBi | Learned positions make the length-generalisation failure in `digitsum` easy to see. RoPE would be the next experiment. |
| Frontend charts | hand-written SVG | chart libraries | Small bundle, exact control over axes and labels, nothing that can invent data. |
| Frontend framework | Vite + React + TS | Next.js | No server-side rendering or routing is needed; the API is separate. |
| Long jobs in the API | only a bounded smoke job, in a subprocess, cancellable | train from request handlers | Never block the API, and never start long GPU runs from a browser on a shared machine. |
| LoRA / pretrained | not executed, recorded as planned | download GPT-2 automatically | The rules require owner approval for substantial downloads. A tiny random model would be a fake comparison. |
| Resource use | `torch.set_num_threads(4)`, standard runs under about 12 min each | longer runs | The machine is shared with other builders and has a known memory fault. |
