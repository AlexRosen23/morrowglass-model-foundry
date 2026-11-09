import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Experiment, type RunRow } from "./api";
import { AttentionPanel, BlueprintPanel, BreakPanel, EvidencePanel, FindingsPanel, GeneratePanel, JobPanel, LedgerPanel, ProvBadge } from "./panels";

const TABS = ["Evidence", "Findings", "Break the Model", "Generate", "Attention", "Jobs"] as const;
type Tab = (typeof TABS)[number];
type Pane = "blueprint" | "center" | "ledger";

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => {
    try { return localStorage.getItem("mg-theme") ?? "auto"; } catch { return "auto"; }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", theme);
    try { localStorage.setItem("mg-theme", theme); } catch { /* private mode */ }
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "auto" ? "dark" : t === "dark" ? "light" : "auto"))];
}

export default function App() {
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [baseline, setBaseline] = useState<any>(null);
  const [dataset, setDataset] = useState<any>(null);
  const [testLog, setTestLog] = useState<any[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<string[]>([]);
  const [cursor, setCursor] = useState(0);
  const [filter, setFilter] = useState("");
  const [tab, setTab] = useState<Tab>("Evidence");
  const [pane, setPane] = useState<Pane>("center");
  const [layer, setLayer] = useState(0);
  const [help, setHelp] = useState(false);
  const [showSmoke, setShowSmoke] = useState(false);
  const [theme, cycleTheme] = useTheme();

  const load = useCallback(() => {
    api.runs().then((r) => {
      setRuns(r.runs);
      setSelected((s) => s ?? r.runs.find((x) => x.provenance === "measured" && x.name === "lm-byte-small")?.run_id ?? r.runs[0]?.run_id ?? null);
    }).catch((e) => setErr(`API unreachable (${e.message}). Start it with: make api`));
    api.experiments().then((r) => setExperiments(r.experiments)).catch(() => undefined);
    api.baselines().then((r) => setBaseline(r.baselines)).catch(() => undefined);
    api.dataset().then((r) => setDataset(r.manifest)).catch(() => undefined);
    api.testLog().then((r) => setTestLog(r.entries)).catch(() => undefined);
  }, []);
  useEffect(load, [load]);

  const visible = useMemo(() => runs.filter((r) => (showSmoke || r.provenance === "measured") && (!filter || r.experiment === filter)), [runs, filter, showSmoke]);
  const ids = useMemo(() => {
    const base = overlay.length ? overlay : selected ? [selected] : [];
    return base.filter((id) => runs.some((r) => r.run_id === id));
  }, [overlay, selected, runs]);
  const toggle = (id: string) => setOverlay((o) => (o.includes(id) ? o.filter((x) => x !== id) : [...o, id].slice(-8)));
  const showRuns = (arms: string[]) => { setOverlay(arms.filter((a) => runs.some((r) => r.run_id === a)).slice(0, 8)); setTab("Evidence"); setPane("center"); };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return;
      if (e.key === "j" || e.key === "ArrowDown") { setCursor((c) => Math.min(c + 1, visible.length - 1)); e.preventDefault(); }
      else if (e.key === "k" || e.key === "ArrowUp") { setCursor((c) => Math.max(c - 1, 0)); e.preventDefault(); }
      else if (e.key === "Enter" && visible[cursor]) setSelected(visible[cursor].run_id);
      else if (e.key === " " && visible[cursor]) { toggle(visible[cursor].run_id); e.preventDefault(); }
      else if (e.key === "c") setOverlay([]);
      else if (e.key === "t") cycleTheme();
      else if (e.key === "?") setHelp((h) => !h);
      else if (e.key === "Escape") setHelp(false);
      else if (/^[1-6]$/.test(e.key)) setTab(TABS[+e.key - 1]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, cursor, cycleTheme]);

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden>◇</span>
          <div>
            <h1>Morrowglass Model Foundry</h1>
            <p className="dim small">A miniature language-model laboratory. Every number on this page is read from a saved run record.</p>
          </div>
        </div>
        <div className="legendbar" aria-label="provenance legend">
          <ProvBadge p="measured" /><ProvBadge p="smoke" /><ProvBadge p="live" />
          <label className="small"><input type="checkbox" checked={showSmoke} onChange={(e) => setShowSmoke(e.target.checked)} /> show smoke runs</label>
          <a className="btnlink" href="/api/report/download">Download report</a>
          <button onClick={cycleTheme} aria-label="cycle colour theme">theme: {theme}</button>
          <button onClick={() => setHelp(true)} aria-label="keyboard shortcuts">?</button>
        </div>
      </header>
      {err && <div className="error banner">{err}</div>}
      <nav className="mobile-switch" aria-label="panels">
        {(["blueprint", "center", "ledger"] as Pane[]).map((p) => (
          <button key={p} className={pane === p ? "on" : ""} onClick={() => setPane(p)}>{p === "center" ? "evidence" : p}</button>
        ))}
      </nav>
      <main className="workbench">
        <aside className={`panel left ${pane === "blueprint" ? "show" : ""}`} aria-label="model blueprint">
          <h2>Model blueprint</h2>
          <BlueprintPanel runId={selected} onHead={(l) => { setLayer(l); setTab("Attention"); setPane("center"); }} />
        </aside>
        <section className={`panel center ${pane === "center" ? "show" : ""}`} aria-label="experimental evidence">
          <div className="tabs" role="tablist">
            {TABS.map((t, i) => (
              <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
                <span className="kbd">{i + 1}</span>{t}
              </button>
            ))}
          </div>
          <div className="tabbody">
            {tab === "Evidence" && <EvidencePanel ids={ids} baseline={baseline} />}
            {tab === "Findings" && <FindingsPanel experiments={experiments} dataset={dataset} onShowRuns={showRuns} />}
            {tab === "Break the Model" && <BreakPanel experiments={experiments} onShowRuns={showRuns} />}
            {tab === "Generate" && <GeneratePanel runs={runs} initial={selected} />}
            {tab === "Attention" && <AttentionPanel runs={runs} initial={selected} layer={layer} setLayer={setLayer} />}
            {tab === "Jobs" && <JobPanel onDone={load} />}
          </div>
        </section>
        <aside className={`panel right ${pane === "ledger" ? "show" : ""}`} aria-label="experiment ledger">
          <h2>Experiment ledger</h2>
          <LedgerPanel runs={visible} cursor={cursor} selected={selected} overlay={overlay} onPick={(id) => { setSelected(id); setCursor(visible.findIndex((r) => r.run_id === id)); }}
            onToggle={toggle} filter={filter} setFilter={setFilter} testLog={testLog} />
        </aside>
      </main>
      {help && (
        <div className="modal" role="dialog" aria-label="keyboard shortcuts" onClick={() => setHelp(false)}>
          <div className="card" onClick={(e) => e.stopPropagation()}>
            <h3>Keyboard</h3>
            <table className="tbl mono"><tbody>
              <tr><td>j / k, ↓ / ↑</td><td>move in the ledger</td></tr>
              <tr><td>Enter</td><td>select run (blueprint + details)</td></tr>
              <tr><td>Space</td><td>add / remove run from the curve overlay</td></tr>
              <tr><td>c</td><td>clear overlay</td></tr>
              <tr><td>1-6</td><td>switch tab</td></tr>
              <tr><td>t</td><td>theme (auto / dark / light)</td></tr>
              <tr><td>?</td><td>this help</td></tr>
            </tbody></table>
            <button onClick={() => setHelp(false)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
