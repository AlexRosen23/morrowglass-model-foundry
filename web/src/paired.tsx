// Paired re-analysis shown under the scratch finding: read from /api/paired (reports/paired/summary.json).
import { useEffect, useState } from "react";

const n = (x: number, d = 4) => (x > 0 ? "+" : "") + x.toFixed(d);

export function PairedNote() {
  const [s, setS] = useState<any>(null);
  const [missing, setMissing] = useState(false);
  useEffect(() => { fetch("/api/paired").then((r) => (r.ok ? r.json() : Promise.reject())).then(setS).catch(() => setMissing(true)); }, []);
  if (missing) return null;
  if (!s) return <p className="dim small">Loading the paired re-analysis…</p>;
  const blk = String(s.protocol.primary_block_bytes);
  const f = s.runs[s.primary_run];
  const p = f.paired[blk], u = f.unpaired[blk];
  return (
    <div className="paired">
      <h4>Paired re-analysis (post-hoc) <span className={`verdict v-${s.verdict === "supported" ? "ok" : "mid"}`}>{s.verdict}</span></h4>
      <p className="small">
        Both models are scored on the same validation bytes, so their errors are correlated. Compared block by block
        ({Number(blk) / 1024} KiB blocks, same resampled blocks for both), the flagship minus {s.baseline.model.model} ({s.baseline.model.tokenizer}) is{" "}
        <b className="mono">{n(p.difference)}</b> bpb, paired 95% interval <b className="mono">[{n(p.ci95[0])}, {n(p.ci95[1])}]</b>;
        lower in {p.blocks_where_a_is_lower} of {p.blocks} blocks and in every play ({f.plays.map((x: any) => n(x.difference, 3)).join(", ")}).
        On the same blocks the unpaired intervals overlap: [{u.model_ci95[0].toFixed(3)}, {u.model_ci95[1].toFixed(3)}] vs [{u.baseline_ci95[0].toFixed(3)}, {u.baseline_ci95[1].toFixed(3)}].
      </p>
      <p className="caveat">Registered {s.registered} after this verdict was known, so it cannot change it: the pre-registered verdict stays "not supported". One seed per model; validation only.</p>
      <div className="source mono">source: {s.source} · {s.report}</div>
    </div>
  );
}
