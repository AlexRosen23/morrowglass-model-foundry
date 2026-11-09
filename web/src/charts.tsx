// Hand-built SVG charts: precise axes, no animation, every series labelled with its run id.
import { useMemo, useState } from "react";

export interface Series { id: string; label: string; x: number[]; y: number[]; color: string; dashed?: boolean }

export const PALETTE = ["#2f6fdf", "#d9622b", "#1f9d74", "#a347c2", "#c9a227", "#d6457a", "#4aa0b5", "#7a7f2a"];

function ticks(lo: number, hi: number, n = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (hi === lo) return [lo];
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(10));
  return out;
}

const compact = (v: number) =>
  Math.abs(v) >= 1e6 ? `${+(v / 1e6).toPrecision(3)}M` : Math.abs(v) >= 1e3 ? `${+(v / 1e3).toPrecision(3)}K` : `${+v.toPrecision(3)}`;

export function LineChart({ series, xLabel, yLabel, height = 300, hlines = [], logY = false }: {
  series: Series[]; xLabel: string; yLabel: string; height?: number;
  hlines?: { y: number; label: string; color: string }[]; logY?: boolean;
}) {
  const [hover, setHover] = useState<{ x: number; y: number; s: string } | null>(null);
  const W = 720, H = height, m = { l: 56, r: 16, t: 12, b: 40 };
  const pts = series.flatMap((s) => s.x.map((x, i) => [x, s.y[i]] as const)).filter(([, y]) => Number.isFinite(y));
  const ys = [...pts.map((p) => p[1]), ...hlines.map((h) => h.y)];
  const tf = (v: number) => (logY ? Math.log10(Math.max(v, 1e-9)) : v);
  const xMin = Math.min(...pts.map((p) => p[0]), 0), xMax = Math.max(...pts.map((p) => p[0]), 1);
  let yMin = Math.min(...ys.map(tf)), yMax = Math.max(...ys.map(tf));
  if (!Number.isFinite(yMin)) { yMin = 0; yMax = 1; }
  const pad = (yMax - yMin) * 0.06 || 0.5;
  yMin -= pad; yMax += pad;
  const sx = (x: number) => m.l + ((x - xMin) / (xMax - xMin || 1)) * (W - m.l - m.r);
  const sy = (y: number) => m.t + (1 - (tf(y) - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);
  const xt = ticks(xMin, xMax), yt = ticks(yMin, yMax);
  if (!pts.length) return <div className="empty">No observed data for this selection.</div>;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={`${yLabel} versus ${xLabel}`}
      onMouseLeave={() => setHover(null)}>
      {yt.map((v) => (
        <g key={`y${v}`}>
          <line x1={m.l} x2={W - m.r} y1={sy(logY ? 10 ** v : v)} y2={sy(logY ? 10 ** v : v)} className="grid" />
          <text x={m.l - 6} y={sy(logY ? 10 ** v : v) + 4} textAnchor="end" className="tick">{compact(logY ? 10 ** v : v)}</text>
        </g>
      ))}
      {xt.map((v) => (
        <g key={`x${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={m.t} y2={H - m.b} className="grid" />
          <text x={sx(v)} y={H - m.b + 16} textAnchor="middle" className="tick">{compact(v)}</text>
        </g>
      ))}
      <text x={(W + m.l) / 2} y={H - 6} textAnchor="middle" className="axis">{xLabel}</text>
      <text x={14} y={(H - m.b) / 2} textAnchor="middle" className="axis" transform={`rotate(-90 14 ${(H - m.b) / 2})`}>{yLabel}</text>
      {hlines.map((h) => (
        <g key={h.label}>
          <line x1={m.l} x2={W - m.r} y1={sy(h.y)} y2={sy(h.y)} stroke={h.color} strokeDasharray="5 4" strokeWidth={1.2} />
          <text x={W - m.r - 4} y={sy(h.y) - 5} textAnchor="end" className="hlabel" fill={h.color}>{h.label}</text>
        </g>
      ))}
      {series.map((s) => {
        const d = s.x.map((x, i) => (Number.isFinite(s.y[i]) ? `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(s.y[i]).toFixed(1)}` : "")).join("");
        return (
          <g key={s.id}>
            <path d={d} fill="none" stroke={s.color} strokeWidth={1.7} strokeDasharray={s.dashed ? "4 3" : undefined} />
            {s.x.map((x, i) => (
              <circle key={i} cx={sx(x)} cy={sy(s.y[i])} r={5} fill="transparent"
                onMouseEnter={() => setHover({ x, y: s.y[i], s: s.label })} />
            ))}
          </g>
        );
      })}
      {hover && (
        <g>
          <circle cx={sx(hover.x)} cy={sy(hover.y)} r={3.5} className="hoverdot" />
          <text x={Math.min(sx(hover.x) + 8, W - 210)} y={Math.max(sy(hover.y) - 8, 14)} className="hovertext">
            {hover.s}: {compact(hover.x)} → {hover.y.toFixed(4)}
          </text>
        </g>
      )}
    </svg>
  );
}

export function Legend({ series }: { series: Series[] }) {
  return (
    <ul className="legend">
      {series.map((s) => (
        <li key={s.id}><span className="swatch" style={{ background: s.color, opacity: s.dashed ? 0.6 : 1 }} />{s.label}</li>
      ))}
    </ul>
  );
}

export function Heatmap({ matrix, labels, size = 360 }: { matrix: number[][]; labels: string[]; size?: number }) {
  const n = matrix.length;
  const cell = size / Math.max(n, 1);
  const [hv, setHv] = useState<[number, number] | null>(null);
  const show = (t: string) => JSON.stringify(t).slice(1, -1);
  return (
    <div className="heatwrap">
      <svg viewBox={`0 0 ${size + 2} ${size + 2}`} className="heatmap" role="img" aria-label="attention matrix">
        {matrix.map((row, i) =>
          row.map((v, k) => (
            <rect key={`${i}-${k}`} x={k * cell + 1} y={i * cell + 1} width={cell} height={cell}
              fill={k > i ? "var(--masked)" : `rgba(var(--heat-rgb), ${Math.sqrt(v).toFixed(3)})`}
              onMouseEnter={() => setHv([i, k])} />
          )),
        )}
      </svg>
      <div className="heatinfo mono">
        {hv ? (
          <>query <b>{show(labels[hv[0]])}</b> (#{hv[0]}) reads key <b>{show(labels[hv[1]])}</b> (#{hv[1]}): weight {matrix[hv[0]][hv[1]].toFixed(4)}
            {hv[1] > hv[0] && " (masked: future position)"}</>
        ) : "Hover a cell. Rows are query positions, columns are key positions; the upper triangle is masked."}
      </div>
    </div>
  );
}

export function Reliability({ bins }: { bins: { lo: number; hi: number; n: number; conf: number | null; acc: number | null }[] }) {
  const S = 220, m = 28;
  const total = useMemo(() => bins.reduce((a, b) => a + b.n, 0), [bins]);
  const sx = (v: number) => m + v * (S - 2 * m), sy = (v: number) => S - m - v * (S - 2 * m);
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="reliability" role="img" aria-label="reliability diagram">
      <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} className="diag" />
      {bins.filter((b) => b.n > 0).map((b) => (
        <g key={b.lo}>
          <rect x={sx(b.lo) + 1} width={sx(b.hi) - sx(b.lo) - 2} y={sy(b.acc ?? 0)} height={sy(0) - sy(b.acc ?? 0)}
            className="relbar" style={{ opacity: 0.35 + 0.65 * Math.min(1, (b.n / total) * 8) }} />
          <circle cx={sx(b.conf ?? 0)} cy={sy(b.acc ?? 0)} r={2.4} className="reldot" />
        </g>
      ))}
      <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(0)} className="axisline" />
      <line x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(1)} className="axisline" />
      <text x={S / 2} y={S - 6} textAnchor="middle" className="tick">top-1 confidence</text>
      <text x={9} y={S / 2} textAnchor="middle" className="tick" transform={`rotate(-90 9 ${S / 2})`}>accuracy</text>
    </svg>
  );
}
