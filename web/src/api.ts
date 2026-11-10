// Typed client for the Morrowglass API. Every value shown in the UI comes from these calls;
// there are no hard-coded metrics or fixture plots in the frontend.

export type Provenance = "measured" | "smoke";

export interface RunRow {
  run_id: string;
  provenance: Provenance;
  experiment: string;
  name: string;
  status: string;
  task: string;
  tokenizer: string;
  dataset: string;
  seed: number;
  params: number;
  steps: number;
  tokens_seen: number;
  wall_time_s: number;
  val_bpb: number | null;
  val_iid_acc: number | null;
  git_commit: string;
  git_dirty: boolean;
  device: string;
  started: string;
  finished: string | null;
  has_checkpoint: boolean;
  test_opened: boolean;
  /** false for runs trained without the causal mask: their windowed val metric is leaky (newer API only). */
  causal?: boolean;
  /** the local checkpoint was re-trained by `morrowglass reproduce`, not the run's original weights. */
  checkpoint_reproduced?: boolean;
}

export interface MetricRec {
  kind: "train" | "eval" | "nonfinite";
  step: number;
  tokens?: number;
  [k: string]: number | string | boolean | null | undefined;
}

export interface CalBin { lo: number; hi: number; n: number; conf: number | null; acc: number | null }

export interface RunDetail {
  provenance: Provenance;
  source_dir: string;
  has_checkpoint: boolean;
  summary: any;
  metadata: any;
  config: { train: Record<string, any>; model: Record<string, any> };
  test_eval: any | null;
  bench: any | null;
}

export interface Layer {
  name: string; kind: string; shape_in: string; shape_out: string; params: number; note: string;
  heads?: number; head_dim?: number;
}

export interface Blueprint { run_id: string; config: Record<string, any>; layers: Layer[]; budget: Record<string, number> }

export interface Experiment {
  experiment: string; title: string; track: string; hypothesis: string; criterion: string; caveats: string[];
  arms: string[]; missing_runs: string[]; verdict: string; verdict_detail?: string; observations: Record<string, any>;
  table?: any[]; plots: string[]; researcher_note: string | null; analyzed: string; analysis_git: string;
}

export interface GenToken { id: number; text: string; prob: number; raw_prob: number; entropy_bits: number; top: [string, number][] }

export interface GenResult {
  prompt: string; text: string; tokens: GenToken[]; note: string; run_id: string; provenance: Provenance; label: string;
  checkpoint_origin?: "original" | "reproduced";
  mean_bits_per_token_self: number; baseline?: { model: string; text: string; order: number };
}

export interface AttentionResult { tokens: string[]; n_layer: number; n_head: number; attention: number[][][][]; caveat: string }

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const b = await r.json(); msg = b.detail ?? b.error?.message ?? msg; } catch { /* not json */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json() as Promise<T>;
}

const post = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  runs: () => j<{ runs: RunRow[]; source: string }>("/api/runs"),
  run: (id: string) => j<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
  metrics: (id: string) => j<{ metrics: MetricRec[]; source: string; provenance: Provenance }>(`/api/runs/${encodeURIComponent(id)}/metrics`),
  blueprint: (id: string) => j<Blueprint>(`/api/runs/${encodeURIComponent(id)}/blueprint`),
  compare: (ids: string[]) => j<{ runs: any[]; config_diff: Record<string, any[]> }>(`/api/compare?runs=${ids.map(encodeURIComponent).join(",")}`),
  experiments: () => j<{ experiments: Experiment[]; source: string }>("/api/experiments"),
  baselines: () => j<{ baselines: any; source: string }>("/api/baselines"),
  dataset: () => j<{ manifest: any; source: string }>("/api/dataset"),
  testLog: () => j<{ entries: any[]; source: string }>("/api/test-log"),
  hardware: () => j<Record<string, any>>("/api/hardware"),
  serving: () => j<any>("/api/serving"),
  completion: (b: Record<string, unknown>) => j<any>("/v1/completions", post(b)),
  generate: (b: { run_id: string; prompt: string; max_new_tokens: number; temperature: number; top_k: number | null; seed: number; with_baseline: boolean }) =>
    j<GenResult>("/api/generate", post(b)),
  baselineGenerate: (b: { run_id: string; prompt: string; max_new_tokens: number; temperature: number; top_k: number | null; seed: number }) =>
    j<{ model: string; text: string; order: number }>("/api/baseline-generate", post({ ...b, with_baseline: false })),
  attention: (run_id: string, text: string) => j<AttentionResult>("/api/attention", post({ run_id, text })),
  startJob: () => j<any>("/api/jobs?kind=smoke", { method: "POST" }),
  job: (id: string) => j<any>(`/api/jobs/${id}`),
  cancelJob: (id: string) => j<any>(`/api/jobs/${id}`, { method: "DELETE" }),
};

export const fmt = {
  n: (x: number | null | undefined, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(d)),
  int: (x: number | null | undefined) => (x === null || x === undefined ? "—" : x.toLocaleString("en-US")),
  si: (x: number) => (x >= 1e9 ? `${(x / 1e9).toFixed(2)}B` : x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : x >= 1e3 ? `${(x / 1e3).toFixed(1)}K` : `${x}`),
  sha: (s: string | undefined) => (s ? s.slice(0, 8) : "—"),
};
