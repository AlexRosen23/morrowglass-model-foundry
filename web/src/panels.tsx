import { useEffect, useMemo, useState } from "react";
import { api, fmt, type AttentionResult, type Blueprint, type Experiment, type GenResult, type MetricRec, type RunDetail, type RunRow } from "./api";
import { Heatmap, LineChart, Legend, PALETTE, Reliability, type Series } from "./charts";
import { PairedNote } from "./paired";

export function ProvBadge({ p }: { p: string }) {
  const label = p === "measured" ? "MEASURED" : p === "smoke" ? "SMOKE" : p === "live" ? "LIVE SAMPLE" : p.toUpperCase();
  return <span className={`badge prov-${p}`} title={
    p === "measured" ? "Read from a saved run record of a standard-profile experiment"
      : p === "smoke" ? "Seconds-long smoke run on the offline sample: plumbing check, not evidence"
        : "Generated now from a saved checkpoint; text is a sample, not a metric"}>{label}</span>;
}

function Source({ path }: { path: string }) {
  return <div className="source mono">source: {path}</div>;
}

// ------------------------------------------------------------------ blueprint (left)

export function BlueprintPanel({ runId, onHead }: { runId: string | null; onHead: (layer: number) => void }) {
  const [bp, setBp] = useState<Blueprint | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!runId) return;
    setErr(null);
    api.blueprint(runId).then(setBp).catch((e) => setErr(String(e.message)));
  }, [runId]);
  if (!runId) return <div className="empty">Select a run in the ledger.</div>;
  if (err) return <div className="error">{err}</div>;
  if (!bp) return <div className="empty">Loading blueprint…</div>;
  const total = bp.budget.params_total;
  const max = Math.max(...bp.layers.map((l) => l.params), 1);
  let block = -1;
  return (
    <div className="blueprint">
      <div className="kv-grid">
        <span>parameters</span><b className="mono">{fmt.int(total)}</b>
        <span>non-embedding</span><b className="mono">{fmt.int(bp.budget.params_non_embedding)}</b>
        <span>layers × heads × d</span><b className="mono">{bp.config.n_layer} × {bp.config.n_head} × {bp.config.d_model}</b>
        <span>context / vocab</span><b className="mono">{bp.config.block_size} / {bp.config.vocab_size}</b>
        <span>positions</span><b className="mono">{bp.config.pos_encoding}{bp.config.causal ? "" : " · NO CAUSAL MASK"}</b>
        <span>train compute</span><b className="mono">{bp.budget.approx_train_flops.toExponential(2)} FLOPs</b>
      </div>
      <ol className="layers">
        {bp.layers.map((l) => {
          const m = l.name.match(/^block(\d+)\./);
          const b = m ? +m[1] : -1;
          const header = b !== block && b >= 0;
          block = b;
          return (
            <li key={l.name} className={`layer kind-${l.kind.split(":")[0]}`}>
              {header && <div className="blockhead">block {b}<span className="mono dim"> residual stream (B, T, {bp.config.d_model})</span></div>}
              <button className="layerbtn" disabled={l.kind !== "attention"} onClick={() => l.kind === "attention" && onHead(b)}
                title={l.kind === "attention" ? "Inspect this layer's attention heads" : l.note}>
                <span className="lname mono">{l.name.replace(/^block\d+\./, "")}</span>
                <span className="lshape mono">{l.shape_out}</span>
                <span className="lparams mono">{l.params ? fmt.si(l.params) : "0"}</span>
                <span className="lbar"><span style={{ width: `${(100 * l.params) / max}%` }} /></span>
                <span className="lnote">{l.note}</span>
              </button>
            </li>
          );
        })}
      </ol>
      <Source path={`model/transformer.py · blueprint() for ${runId}`} />
    </div>
  );
}

// ------------------------------------------------------------------ ledger (right)

export function LedgerPanel({ runs, cursor, selected, overlay, onPick, onToggle, filter, setFilter, testLog }: {
  runs: RunRow[]; cursor: number; selected: string | null; overlay: string[]; onPick: (id: string) => void;
  onToggle: (id: string) => void; filter: string; setFilter: (f: string) => void; testLog: any[];
}) {
  const experiments = useMemo(() => Array.from(new Set(runs.map((r) => r.experiment))).sort(), [runs]);
  return (
    <div className="ledger">
      <div className="ledger-filter">
        <label htmlFor="flt">experiment</label>
        <select id="flt" value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="">all ({runs.length})</option>
          {experiments.map((e) => <option key={e} value={e}>{e}</option>)}
        </select>
      </div>
      <ul className="runlist" role="listbox" aria-label="experiment runs">
        {runs.map((r, i) => (
          <li key={`${r.provenance}-${r.run_id}`} role="option" aria-selected={selected === r.run_id}
            className={`run ${selected === r.run_id ? "sel" : ""} ${i === cursor ? "cur" : ""}`}
            onClick={() => onPick(r.run_id)}>
            <div className="runtop">
              <input type="checkbox" aria-label={`overlay ${r.run_id}`} checked={overlay.includes(r.run_id)}
                onChange={() => onToggle(r.run_id)} onClick={(e) => e.stopPropagation()} />
              <span className="rid mono">{r.run_id}</span>
              <ProvBadge p={r.provenance} />
            </div>
            <div className="runmeta mono">
              <span className={`status st-${r.status}`}>{r.status}</span>
              <span>seed {r.seed}</span>
              <span>{r.tokenizer}</span>
              <span>{fmt.si(r.params)}p</span>
              <span>{r.task === "lm" ? `val ${fmt.n(r.val_bpb)} bpb` : `iid ${fmt.n(r.val_iid_acc)} acc`}</span>
              <span title={r.git_dirty ? "working tree had uncommitted changes" : "clean commit"}>@{fmt.sha(r.git_commit)}{r.git_dirty ? "*" : ""}</span>
              {r.test_opened && <span className="testmark" title="locked test split opened for this run">test↗</span>}
            </div>
          </li>
        ))}
      </ul>
      <details className="testlog">
        <summary>Test-set access log ({testLog.length})</summary>
        {testLog.length === 0 ? <p className="dim">The locked test split has not been opened.</p> : (
          <ul>{testLog.map((t, i) => <li key={i} className="mono"><b>{t.time}</b> {t.runs.length} run(s){t.baseline ? " + baseline" : ""}: {t.reason}</li>)}</ul>
        )}
        <Source path="reports/test_access_log.jsonl" />
      </details>
    </div>
  );
}

// ------------------------------------------------------------------ evidence (center)

const METRICS: { key: string; label: string; kind: "eval" | "train"; x: "tokens" | "step" }[] = [
  { key: "val_bits_per_byte", label: "val bits/byte (subset)", kind: "eval", x: "tokens" },
  { key: "train_eval_bits_per_byte", label: "train bits/byte (subset)", kind: "eval", x: "tokens" },
  { key: "val_loss", label: "val loss (nats/token)", kind: "eval", x: "tokens" },
  { key: "val_iid_accuracy", label: "digit-sum val accuracy", kind: "eval", x: "step" },
  { key: "train_loss", label: "train loss (batch)", kind: "train", x: "step" },
  { key: "lr", label: "learning rate", kind: "train", x: "step" },
  { key: "grad_norm", label: "gradient norm (pre-clip)", kind: "train", x: "step" },
  { key: "tokens_per_s", label: "tokens / second", kind: "train", x: "step" },
  { key: "mem_mb", label: "memory (MB)", kind: "train", x: "step" },
];

export function EvidencePanel({ ids, baseline }: { ids: string[]; baseline: any }) {
  const [metric, setMetric] = useState(METRICS[0].key);
  const [data, setData] = useState<Record<string, MetricRec[]>>({});
  const [diff, setDiff] = useState<Record<string, any[]> | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  useEffect(() => {
    ids.forEach((id) => {
      if (!data[id]) api.metrics(id).then((m) => setData((d) => ({ ...d, [id]: m.metrics }))).catch(() => undefined);
    });
    if (ids.length > 1) api.compare(ids).then((c) => setDiff(c.config_diff)).catch(() => setDiff(null));
    else setDiff(null);
    if (ids[0]) api.run(ids[0]).then(setDetail).catch(() => setDetail(null));
  }, [ids.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps
  const m = METRICS.find((x) => x.key === metric)!;
  const series: Series[] = ids.map((id, i) => {
    const recs = (data[id] ?? []).filter((r) => r.kind === m.kind && typeof r[m.key] === "number");
    return { id, label: id, x: recs.map((r) => Number(r[m.x] ?? r.step)), y: recs.map((r) => Number(r[m.key])), color: PALETTE[i % PALETTE.length] };
  });
  const hl = baseline && metric.endsWith("bits_per_byte")
    ? [{ y: baseline.best_val_bpb, label: `best n-gram val (${baseline.best.model}, ${baseline.best.tokenizer})`, color: "var(--warn)" }] : [];
  const val = detail?.summary?.final?.val;
  return (
    <div className="evidence">
      <div className="toolbar">
        <label htmlFor="metric">metric</label>
        <select id="metric" value={metric} onChange={(e) => setMetric(e.target.value)}>
          {METRICS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
        </select>
        <span className="dim">{ids.length} run(s) overlaid · tick runs in the ledger (space) to compare</span>
      </div>
      <LineChart series={series} xLabel={m.x === "tokens" ? "training tokens" : "optimizer step"} yLabel={m.label} hlines={hl}
        logY={metric === "lr"} />
      <Legend series={series} />
      <Source path={ids.map((id) => `reports/runs/${id}/metrics.jsonl`).join(" · ") || "—"} />
      {diff && (
        <section className="card">
          <h3>Exact configuration differences</h3>
          {Object.keys(diff).length === 0 ? <p className="dim">Identical configurations.</p> : (
            <table className="tbl mono"><thead><tr><th>field</th>{ids.map((id, i) => <th key={id} style={{ color: PALETTE[i % PALETTE.length] }}>run {i + 1}</th>)}</tr></thead>
              <tbody>{Object.entries(diff).map(([k, v]) => <tr key={k}><td>{k}</td>{v.map((x, i) => <td key={i}>{JSON.stringify(x)}</td>)}</tr>)}</tbody></table>
          )}
        </section>
      )}
      {detail && (
        <section className="card grid2">
          <div>
            <h3>Held-out result <ProvBadge p={detail.provenance} /></h3>
            {val ? (
              <div className="kv-grid">
                <span>val bits/byte (full)</span><b className="mono">{fmt.n(val.bits_per_byte, 4)}</b>
                <span>95% bootstrap CI</span><b className="mono">[{fmt.n(val.bits_per_byte_ci95?.[0], 4)}, {fmt.n(val.bits_per_byte_ci95?.[1], 4)}]</b>
                <span>loss / perplexity</span><b className="mono">{fmt.n(val.loss_nats_per_token)} nats · {fmt.n(val.perplexity_per_token, 2)} per {detail.config.train.tokenizer} token</b>
                <span>calibration ECE</span><b className="mono">{fmt.n(val.calibration?.ece, 4)} (top-1 acc {fmt.n(val.calibration?.top1_accuracy)})</b>
                <span>test split</span><b className="mono">{detail.test_eval ? `${fmt.n(detail.test_eval.results.test?.bits_per_byte, 4)} bpb (opened ${detail.test_eval.evaluated})` : "locked"}</b>
                <span>status / wall</span><b className="mono">{detail.summary.status} · {detail.summary.wall_time_s}s on {detail.metadata.device}</b>
                <span>commit / config</span><b className="mono">{fmt.sha(detail.metadata.git.commit)}{detail.metadata.git.dirty ? "*" : ""} / {fmt.sha(detail.metadata.config_hash)}</b>
                <span>dataset hash</span><b className="mono">{fmt.sha(detail.metadata.data.dataset_hash)}</b>
              </div>
            ) : detail.summary.final?.val_iid ? (
              <div className="kv-grid">
                <span>train acc</span><b className="mono">{fmt.n(detail.summary.final.train.accuracy)}</b>
                <span>val (iid) acc</span><b className="mono">{fmt.n(detail.summary.final.val_iid.accuracy)}</b>
                {detail.test_eval && Object.entries(detail.test_eval.results).map(([k, v]: [string, any]) => (
                  <><span key={k}>{k} acc</span><b className="mono">{fmt.n(v.accuracy)}</b></>
                ))}
              </div>
            ) : <p className="dim">No final evaluation (status {detail.summary.status}).</p>}
            <Source path={`${detail.source_dir}/summary.json`} />
          </div>
          {val?.calibration && (
            <div>
              <h3>Reliability (val, top-1)</h3>
              <Reliability bins={val.calibration.bins} />
              <p className="dim small">Bars: accuracy per confidence bin; dots: mean confidence. On the diagonal = calibrated.</p>
            </div>
          )}
        </section>
      )}
      {detail?.bench && (
        <section className="card">
          <h3>Inference cost (measured)</h3>
          <table className="tbl mono"><thead><tr><th>device</th><th>batch-1 latency (median)</th><th>throughput b32</th><th>greedy ms/token</th><th>memory</th></tr></thead>
            <tbody>{Object.entries(detail.bench.devices).map(([d, v]: [string, any]) => (
              <tr key={d}><td>{d}</td><td>{v.forward_latency_ms_batch1_fullctx.median} ms</td><td>{fmt.int(v.forward_tokens_per_s_batch32)} tok/s</td><td>{v.greedy_decode_ms_per_token}</td><td>{v.memory_mb} MB</td></tr>
            ))}</tbody></table>
          <p className="dim small">{detail.bench.caveat}</p>
        </section>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ generate

export function GeneratePanel({ runs, initial }: { runs: RunRow[]; initial: string | null }) {
  const lmRuns = runs.filter((r) => r.has_checkpoint && r.task === "lm");
  const [runId, setRunId] = useState(initial && lmRuns.some((r) => r.run_id === initial) ? initial : lmRuns[0]?.run_id ?? "");
  const [prompt, setPrompt] = useState("ROMEO:\n");
  const [temp, setTemp] = useState(0.8);
  const [topk, setTopk] = useState(40);
  const [seed, setSeed] = useState(0);
  const [n, setN] = useState(200);
  const [res, setRes] = useState<GenResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [inspect, setInspect] = useState<number | null>(null);
  const [base, setBase] = useState<{ model: string; text: string } | "loading" | null>(null);
  const go = () => {
    setBusy(true); setErr(null); setBase("loading"); setInspect(null);
    const body = { run_id: runId, prompt, max_new_tokens: n, temperature: temp, top_k: topk, seed };
    api.generate({ ...body, with_baseline: false }).then(setRes).catch((e) => setErr(String(e.message))).finally(() => setBusy(false));
    api.baselineGenerate(body).then(setBase).catch((e) => setBase({ model: "baseline unavailable", text: String(e.message) }));
  };
  if (!lmRuns.length) return <div className="empty">No local checkpoints. Run <code>make smoke</code> or an experiment first.</div>;
  return (
    <div className="generate">
      <div className="formrow">
        <label>checkpoint<select value={runId} onChange={(e) => setRunId(e.target.value)}>
          {lmRuns.map((r) => <option key={r.run_id} value={r.run_id}>{r.provenance === "smoke" ? "[smoke] " : ""}{r.run_id} ({fmt.n(r.val_bpb)} bpb)</option>)}
        </select></label>
        <label>temperature<input type="number" step={0.1} min={0} max={2} value={temp} onChange={(e) => setTemp(+e.target.value)} /></label>
        <label>top-k<input type="number" min={1} max={1024} value={topk} onChange={(e) => setTopk(+e.target.value)} /></label>
        <label>seed<input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} /></label>
        <label>tokens<input type="number" min={1} max={400} value={n} onChange={(e) => setN(+e.target.value)} /></label>
      </div>
      <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} aria-label="prompt" maxLength={500}
        onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !busy && runId) { e.preventDefault(); go(); } }} />
      <button className="primary" onClick={go} disabled={busy || !runId}>{busy ? "Sampling…" : "Generate (⌘↵)"}</button>
      {err && <div className="error">{err}</div>}
      {res && (
        <div className="grid2">
          <section className="card">
            <h3>Transformer <ProvBadge p="live" /> <span className="dim small">{res.run_id}</span></h3>
            <pre className="gen">
              <span className="dim">{res.prompt}</span>
              {res.tokens.map((t, i) => (
                <span key={i} className={`tok ${inspect === i ? "on" : ""}`} onClick={() => setInspect(i)}
                  style={{ background: `rgba(var(--heat-rgb), ${(0.08 + 0.5 * (1 - t.raw_prob)).toFixed(2)})` }}
                  title={`p=${t.raw_prob}`}>{t.text}</span>
              ))}
            </pre>
            <p className="dim small">Shading: darker = the model gave the sampled token lower probability. Click a token to inspect. {res.note}</p>
            {inspect !== null && res.tokens[inspect] && (
              <div className="inspect mono">
                token #{inspect} {JSON.stringify(res.tokens[inspect].text)} · p(sampled, after T/top-k) {res.tokens[inspect].prob} ·
                p(raw) {res.tokens[inspect].raw_prob} · entropy {res.tokens[inspect].entropy_bits} bits
                <div>top-5: {res.tokens[inspect].top.map(([t, p]) => `${JSON.stringify(t)} ${p}`).join(" · ")}</div>
              </div>
            )}
          </section>
          {base && (
            <section className="card">
              <h3>Baseline: {base === "loading" ? "fitting n-gram counts (first use can take a minute)…" : base.model} <ProvBadge p="live" /></h3>
              <pre className="gen"><span className="dim">{res.prompt}</span>{base === "loading" ? "" : base.text}</pre>
              <p className="dim small">Same prompt, seed and temperature, sampled from the count-based baseline fitted on the train split.</p>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ attention

export function AttentionPanel({ runs, initial, layer, setLayer }: { runs: RunRow[]; initial: string | null; layer: number; setLayer: (l: number) => void }) {
  const lmRuns = runs.filter((r) => r.has_checkpoint && r.task === "lm");
  const [runId, setRunId] = useState(initial && lmRuns.some((r) => r.run_id === initial) ? initial : lmRuns[0]?.run_id ?? "");
  const [text, setText] = useState("To be, or not to be, that is the question.");
  const [res, setRes] = useState<AttentionResult | null>(null);
  const [head, setHead] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const go = () => { setErr(null); api.attention(runId, text).then((r) => { setRes(r); if (layer >= r.n_layer) setLayer(0); }).catch((e) => setErr(String(e.message))); };
  useEffect(() => { if (runId) go(); }, [runId]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!lmRuns.length) return <div className="empty">No local checkpoints.</div>;
  return (
    <div className="attention">
      <div className="formrow">
        <label>checkpoint<select value={runId} onChange={(e) => setRunId(e.target.value)}>
          {lmRuns.map((r) => <option key={r.run_id} value={r.run_id}>{r.provenance === "smoke" ? "[smoke] " : ""}{r.run_id}</option>)}
        </select></label>
        <input className="grow" value={text} onChange={(e) => setText(e.target.value)} maxLength={400} aria-label="input text" />
        <button onClick={go}>Inspect</button>
      </div>
      {err && <div className="error">{err}</div>}
      {res && (
        <>
          <div className="heads">
            {res.attention.map((L, li) => (
              <div key={li} className="headrow">
                <span className="mono dim">layer {li}</span>
                {L.map((_, hi) => (
                  <button key={hi} className={`headbtn ${li === layer && hi === head ? "on" : ""}`} onClick={() => { setLayer(li); setHead(hi); }}
                    aria-label={`layer ${li} head ${hi}`}>h{hi}</button>
                ))}
              </div>
            ))}
          </div>
          {res.attention[layer] && <Heatmap matrix={res.attention[layer][head]} labels={res.tokens} />}
          <p className="caveat">{res.caveat}</p>
          <Source path={`live forward pass of artifacts/checkpoints/${runId}/best.pt`} />
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ findings / experiments

const VERDICT_CLASS: Record<string, string> = { supported: "ok", "partially supported": "mid", "not supported": "bad", contradicted: "bad", pending: "pend", inconclusive: "mid" };

export function FindingsPanel({ experiments, dataset, onShowRuns }: { experiments: Experiment[]; dataset: any; onShowRuns: (ids: string[]) => void }) {
  const [open, setOpen] = useState<string | null>(null);
  const exps = experiments.filter((e) => !e.experiment.startsWith("break-") && !e.experiment.startsWith("digitsum"));
  return (
    <div className="findings">
      <p className="dim">Each experiment was registered (hypothesis + success criterion) in <code>experiments/registry.py</code> before it ran; verdicts are computed by <code>experiments/analyze.py</code> from saved run records.</p>
      {exps.map((e) => <FindingCard key={e.experiment} e={e} open={open === e.experiment} onOpen={() => setOpen(open === e.experiment ? null : e.experiment)} dataset={dataset} onShowRuns={onShowRuns} />)}
    </div>
  );
}

function FindingCard({ e, open, onOpen, dataset, onShowRuns }: { e: Experiment; open: boolean; onOpen: () => void; dataset: any; onShowRuns: (ids: string[]) => void }) {
  return (
    <section className={`card finding ${open ? "open" : ""}`}>
      <button className="findhead" onClick={onOpen} aria-expanded={open}>
        <span className={`verdict v-${VERDICT_CLASS[e.verdict] ?? "mid"}`}>{e.verdict}</span>
        <span className="ftitle">{e.title}</span>
        <span className="dim mono small">{e.arms.length} run(s)</span>
      </button>
      <p className="detail mono small">{e.verdict_detail}</p>
      {open && (
        <div className="findbody">
          <dl>
            <dt>Hypothesis</dt><dd>{e.hypothesis}</dd>
            <dt>Pre-registered criterion</dt><dd>{e.criterion}</dd>
            <dt>Caveats</dt><dd><ul>{e.caveats.map((c) => <li key={c}>{c}</li>)}</ul></dd>
            {e.researcher_note && <><dt>Researcher note</dt><dd className="note">{e.researcher_note}</dd></>}
            <dt>Provenance</dt>
            <dd className="mono small">
              analysed {e.analyzed} at commit {fmt.sha(e.analysis_git)} · dataset: {dataset?.source?.name ?? "—"} · licence: {dataset?.source?.license ?? "—"}
              {e.track === "algorithmic" && " · synthetic data generated by data/synthetic.py (seeded)"}
            </dd>
          </dl>
          {e.table && (
            <table className="tbl mono small"><thead><tr><th>run</th><th>status</th><th>params</th><th>val bpb</th><th>95% CI</th><th>ECE</th><th>wall s</th><th>commit</th></tr></thead>
              <tbody>{e.table.map((r: any) => (
                <tr key={r.run_id}><td>{r.run_id}</td><td>{r.status}</td><td>{fmt.si(r.params)}</td><td>{fmt.n(r.val_bpb, 4)}</td>
                  <td>{r.val_bpb_ci95 ? `[${fmt.n(r.val_bpb_ci95[0], 3)}, ${fmt.n(r.val_bpb_ci95[1], 3)}]` : "—"}</td><td>{fmt.n(r.ece, 3)}</td><td>{r.wall_time_s}</td><td>{fmt.sha(r.git_commit)}{r.git_dirty ? "*" : ""}</td></tr>
              ))}</tbody></table>
          )}
          {e.experiment === "scratch" && <PairedNote />}
          <button onClick={() => onShowRuns(e.arms)}>Overlay these runs' curves</button>
          <div className="plots">{e.plots.map((p) => <img key={p} src={`/api/experiments/${e.experiment}/plots/${p}`} alt={`${e.title}: ${p}`} loading="lazy" />)}</div>
          <Source path={`reports/experiments/${e.experiment}/result.json`} />
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ break the model / what failed

export function BreakPanel({ experiments, onShowRuns }: { experiments: Experiment[]; onShowRuns: (ids: string[]) => void }) {
  const cases = experiments.filter((e) => e.experiment.startsWith("break-") || e.experiment.startsWith("digitsum"));
  return (
    <div className="break">
      <p className="dim">Controlled failures with fixed seeds. Each row follows <b>problem → hypothesis → experiment → result → decision</b>, read from the saved records.</p>
      {cases.map((e) => (
        <section className="card breakcase" key={e.experiment}>
          <h3>{e.title} <span className={`verdict v-${VERDICT_CLASS[e.verdict] ?? "mid"}`}>{e.verdict}</span></h3>
          <div className="chain">
            <div><h4>What failed</h4><p>{WHAT_FAILED[e.experiment] ?? e.hypothesis}</p></div>
            <div><h4>Hypothesis</h4><p>{e.hypothesis}</p></div>
            <div><h4>Experiment</h4><p className="mono small">{e.arms.join("\n")}</p></div>
            <div><h4>Measured result</h4><p className="mono small">{e.verdict_detail}</p></div>
            <div><h4>What I changed</h4><p>{e.researcher_note ? e.researcher_note.split("\n").find((l) => l.startsWith("Decision:"))?.replace("Decision:", "").trim() ?? WHAT_CHANGED[e.experiment] : WHAT_CHANGED[e.experiment]}</p></div>
          </div>
          <div className="plots">{e.plots.map((p) => <img key={p} src={`/api/experiments/${e.experiment}/plots/${p}`} alt={`${e.title}: ${p}`} loading="lazy" />)}</div>
          <button onClick={() => onShowRuns(e.arms)}>Overlay runs</button>
          <Source path={`reports/experiments/${e.experiment}/result.json`} />
        </section>
      ))}
    </div>
  );
}

// Problem statements are fixed descriptions of the induced fault, not results.
const WHAT_FAILED: Record<string, string> = {
  "break-lr": "Learning rate set ~30x too high (0.1), with no warmup and no gradient clipping.",
  "break-mask": "The causal mask is removed, so each position can attend to future tokens during training.",
  "break-mask-long": "Same as break-mask, but trained 5x longer to see whether the model finds the future-token shortcut.",
  "break-overfit": "A 4.8M-parameter model is trained on only 240 KB of text with no regularisation.",
  digitsum: "A model trained on 3-6-term sums is asked to sum 8-16 terms (unseen lengths and positions).",
  "digitsum-scratchpad": "With a running-sum scratchpad the rule is learnable; does it transfer to 8-16 terms at unseen positions?",
  "digitsum-scratchpad-seeds": "Seeds 0 and 1 disagreed (one transferred, one did not). Is transfer with position augmentation the rule or the exception?",
};
const WHAT_CHANGED: Record<string, string> = {
  "break-lr": "Keep warmup + clipping as defaults, and select the learning rate with the optimisation experiment.",
  "break-mask": "Keep the causal mask; the leakage unit test (tests/test_model.py) now guards it in CI.",
  "break-mask-long": "Keep the mask and the leakage unit test; judge non-causal models only with leak-free scoring.",
  "break-overfit": "Use dropout + early stopping on scarce data; report best-on-validation checkpoints.",
  digitsum: "The plain format never left chance (parity-like plateau); switched to a running-sum scratchpad follow-up.",
  "digitsum-scratchpad": "Random position offsets in training; measured per seed on the locked OOD splits.",
  "digitsum-scratchpad-seeds": "Three more seeds per arm, same recipe; transfer counted only when it beats the plain twin by more than 3 standard errors.",
};

export function JobPanel({ onDone }: { onDone: () => void }) {
  const [job, setJob] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const es = new EventSource(`/api/jobs/${job.id}/events`);
    es.onmessage = (ev) => { const v = JSON.parse(ev.data); setJob(v); if (v.status !== "running") { es.close(); onDone(); } };
    es.onerror = () => es.close();
    return () => es.close();
  }, [job?.id, job?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  const evals = (job?.metrics ?? []).filter((m: any) => m.kind === "eval");
  return (
    <div className="card">
      <h3>Run a bounded smoke job <ProvBadge p="smoke" /></h3>
      <p className="dim small">Trains a 1-layer model for 40 steps on the offline sample in a subprocess (CPU, ≤180 s, one at a time, cancellable). Standard experiments run only from the CLI.</p>
      <div className="formrow">
        <button className="primary" disabled={job?.status === "running"} onClick={() => { setErr(null); api.startJob().then(setJob).catch((e) => setErr(String(e.message))); }}>Start smoke job</button>
        <button disabled={job?.status !== "running"} onClick={() => api.cancelJob(job.id).then(setJob)}>Cancel</button>
        {job && <span className="mono">{job.status} · {job.elapsed_s}s · {job.metrics.length} records</span>}
      </div>
      {err && <div className="error">{err}</div>}
      {evals.length > 0 && (
        <LineChart height={180} xLabel="step" yLabel="val bits/byte"
          series={[{ id: "job", label: job.run_id, color: PALETTE[2], x: evals.map((e: any) => e.step), y: evals.map((e: any) => e.val_bits_per_byte) }]} />
      )}
    </div>
  );
}

// ------------------------------------------------------------------ serving (KV cache, int8, harness, OpenAI-compatible API)

function Verdict({ v }: { v: string }) {
  return <span className={`verdict v-${VERDICT_CLASS[v] ?? "mid"}`}>{v}</span>;
}

export function ServingPanel({ runs, initial }: { runs: RunRow[]; initial: string | null }) {
  const [s, setS] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.serving().then(setS).catch((e) => setErr(String(e.message))); }, []);
  if (err) return <div className="error">{err}</div>;
  if (!s) return <div className="empty">Loading serving records…</div>;
  const kv = Object.values(s.kv_cache ?? {}) as any[];
  const q = Object.values(s.int8 ?? {}) as any[];
  const le = s.lm_eval;
  return (
    <div className="serving">
      <p className="dim">Serving studies were registered (hypothesis + criterion) in <code>experiments/registry.py</code> before they ran, on existing checkpoints and validation text only. Shared machine: speeds are indicative.</p>
      {kv.map((k) => (
        <section className="card" key={k.run_id}>
          <h3>KV cache: decode speed <ProvBadge p="measured" /> <Verdict v={k.verdict} /> <span className="dim small">{k.run_id}</span></h3>
          <table className="tbl mono"><thead><tr><th>device</th><th>prompt</th><th>no cache</th><th>KV cache</th><th>speed-up</th><th>decode ms/token (cache)</th></tr></thead>
            <tbody>{(["cpu", "mps"] as const).flatMap((d) => k.protocol.prompt_tokens.map((L: number) => {
              const off = k.rows.find((r: any) => r.device === d && r.prompt_tokens === L && !r.kv_cache);
              const on = k.rows.find((r: any) => r.device === d && r.prompt_tokens === L && r.kv_cache);
              if (!on || !off) return null;
              return <tr key={`${d}${L}`}><td>{d}</td><td>{L} tok</td><td>{off.tokens_per_s} tok/s</td><td>{on.tokens_per_s} tok/s</td>
                <td>{fmt.n(on.tokens_per_s / off.tokens_per_s, 2)}x</td><td>{on.decode_ms_per_token}</td></tr>;
            }))}</tbody></table>
          <p className="dim small">Equivalence on CPU: same greedy tokens {String(k.checks.same_tokens)}, max |Δlogit| {Math.max(...k.equivalence_cpu.map((e: any) => e.max_abs_logit_diff)).toExponential(1)}. Batch 1, greedy, {k.protocol.new_tokens} new tokens, {k.threads} threads, median of {k.protocol.repeats}.</p>
          <Source path={`reports/serving/kv-cache/${k.run_id}.json`} />
        </section>
      ))}
      {q.length > 0 && (
        <section className="card">
          <h3>Dynamic int8 on CPU: quality vs speed <ProvBadge p="measured" /></h3>
          <table className="tbl mono"><thead><tr><th>run</th><th>val bpb fp32 → int8</th><th>Δ</th><th>decode tok/s fp32 → int8</th><th>weights MB</th><th>verdict</th></tr></thead>
            <tbody>{q.map((r) => (
              <tr key={r.run_id}><td>{r.run_id}</td><td>{fmt.n(r.fp32.val_bpb, 4)} → {fmt.n(r.int8.val_bpb, 4)}</td><td>{r.delta_val_bpb >= 0 ? "+" : ""}{fmt.n(r.delta_val_bpb, 4)}</td>
                <td>{r.fp32.tokens_per_s} → {r.int8.tokens_per_s}</td><td>{r.fp32.weights_mb} → {r.int8.weights_mb}</td><td><Verdict v={r.verdict} /></td></tr>
            ))}</tbody></table>
          <Source path="reports/serving/int8/*.json" />
        </section>
      )}
      {le && (
        <section className="card">
          <h3>lm-evaluation-harness through <code>/v1/completions</code> <ProvBadge p="measured" /> <Verdict v={le.verdict} /></h3>
          <table className="tbl mono"><thead><tr><th>model</th><th>lambada acc</th><th>wikitext bits/byte</th><th>Shakespeare val bpb: harness / own</th></tr></thead>
            <tbody>{Object.entries(le.models).map(([m, r]: [string, any]) => (
              <tr key={m}><td>{m}</td><td>{fmt.n(r.lambada_acc, 4)}</td><td>{fmt.n(r.wikitext_bits_per_byte, 3)}</td>
                <td>{fmt.n(r.shakespeare_val_bits_per_byte, 3)}{r.own_val_bpb !== undefined ? ` / ${fmt.n(r.own_val_bpb, 3)}` : ""}</td></tr>
            ))}</tbody></table>
          <p className="dim small">Expected to be weak on general English: about 2M parameters, Shakespeare only, 256 tokens of context. The Shakespeare row checks the plumbing: the harness, through the API, should reproduce the project's own validation number.</p>
          <Source path="reports/lm_eval/summary.json" />
        </section>
      )}
      <OpenAITry runs={runs} initial={initial} />
    </div>
  );
}

function OpenAITry({ runs, initial }: { runs: RunRow[]; initial: string | null }) {
  const lmRuns = runs.filter((r) => r.has_checkpoint && r.task === "lm");
  const [runId, setRunId] = useState(initial && lmRuns.some((r) => r.run_id === initial) ? initial : lmRuns[0]?.run_id ?? "");
  const [prompt, setPrompt] = useState("HAMLET.\n");
  const [res, setRes] = useState<{ body: any; ms: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const body = { model: runId, prompt, max_tokens: 80, temperature: 0.8, seed: 0, stop: ["\n\n"] };
  const go = () => {
    setBusy(true); setErr(null);
    const t = performance.now();
    api.completion(body).then((b) => setRes({ body: b, ms: performance.now() - t })).catch((e) => setErr(String(e.message))).finally(() => setBusy(false));
  };
  if (!lmRuns.length) return null;
  return (
    <section className="card">
      <h3>OpenAI-compatible API <ProvBadge p="live" /></h3>
      <div className="formrow">
        <label>model<select value={runId} onChange={(e) => setRunId(e.target.value)}>
          {lmRuns.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
        </select></label>
        <input value={prompt} onChange={(e) => setPrompt(e.target.value)} aria-label="completion prompt" maxLength={500} />
        <button className="primary" onClick={go} disabled={busy || !runId}>{busy ? "Calling…" : "POST /v1/completions"}</button>
      </div>
      <pre className="gen mono small">{`curl -s 127.0.0.1:5100/v1/completions -H 'Content-Type: application/json' -d '${JSON.stringify(body).replace(/'/g, "'\\''")}'`}</pre>
      {err && <div className="error">{err}</div>}
      {res && (
        <>
          <pre className="gen"><span className="dim">{prompt}</span>{res.body.choices?.[0]?.text}</pre>
          <p className="dim small">finish_reason {res.body.choices?.[0]?.finish_reason} · {res.body.usage?.completion_tokens} tokens · {Math.round(res.ms)} ms round trip (browser-measured) · KV cache {String(res.body.morrowglass?.kv_cache)} · not instruction-tuned: this is a continuation, not an answer.</p>
        </>
      )}
    </section>
  );
}
