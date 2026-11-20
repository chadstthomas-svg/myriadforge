"""Render the self-contained HTML dashboard (zero external/CDN assets).

Inputs are myriadforge.result/1 + myriadforge.demo/1 records (report.collect).
Everything user-visible is HTML-escaped; charts are inline SVG.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

from myriadforge.report import collect

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #10141a; color: #d6dee8; margin: 0 auto; max-width: 1100px;
       padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.2rem; margin: 2.25rem 0 0.75rem; border-bottom: 1px solid #2a3340;
     padding-bottom: 0.4rem; }
h3 { font-size: 1rem; margin: 1.5rem 0 0.5rem; color: #9fb3c8; }
.sub { color: #7c8ea0; font-size: 0.85rem; margin-bottom: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.45rem 0.7rem; border-bottom: 1px solid #232b36; }
th { color: #9fb3c8; font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
     letter-spacing: 0.04em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.err td { color: #e07070; }
tr.err td:first-child { text-decoration: line-through; }
.sim { color: #d9a441; font-size: 0.72rem; border: 1px solid #6e5a24;
       border-radius: 3px; padding: 0 0.3rem; margin-left: 0.35rem; }
.speed { color: #6ecf97; }
.charts { margin: 1rem 0 2rem; }
.chart-row { margin: 0.9rem 0; }
.chart-label { font-size: 0.85rem; color: #9fb3c8; margin-bottom: 0.2rem; }
svg { display: block; width: 100%; }
.bar { fill: #3d7fd6; } .bar.sim { fill: #8a6d2f; }
.whisker { stroke: #b8cede; stroke-width: 1.5; }
.value { fill: #d6dee8; font-size: 11px; }
.card { background: #171e27; border: 1px solid #232b36; border-radius: 8px;
        padding: 0.9rem 1.1rem; margin: 0.75rem 0; }
.card .meta { color: #7c8ea0; font-size: 0.8rem; }
.chain { font-family: ui-monospace, Consolas, monospace; font-size: 0.82rem;
         color: #9fb3c8; margin-top: 0.4rem; }
.chain b { color: #d6dee8; }
pre.output { background: #0b0f14; border: 1px solid #232b36; border-radius: 6px;
             padding: 0.7rem; overflow-x: auto; font-size: 0.82rem; margin: 0.5rem 0; }
.note { color: #7c8ea0; font-size: 0.85rem; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_ms(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f} ms" if value is not None else "-"


def _bar_chart(rows: list[tuple[str, float | None, float | None, bool]],
               unit: str = "ms") -> str:
    """Horizontal bars; rows = (label, p50, p90, simulated). Log-ish scale."""
    max_v = max((p90 or p50 or 0) for _, p50, p90, _ in rows) or 1.0
    width, row_h = 720, 26
    bar_x = 190
    parts = [f'<svg viewBox="0 0 {width} {row_h * len(rows)}" role="img">']
    for i, (label, p50, p90, sim) in enumerate(rows):
        y = i * row_h
        scale = (bar_x + 10) / max_v if max_v else 1
        main_w = max((p50 or 0) * scale, 1)
        whisk_w = max(((p90 or p50 or 0)) * scale, main_w)
        cls = "bar sim" if sim else "bar"
        parts.append(f'<text x="0" y="{y + 17}" fill="#9fb3c8" font-size="11.5" '
                     f'font-family="sans-serif">{_esc(label[:34])}</text>')
        parts.append(f'<rect class="{cls}" x="{bar_x}" y="{y + 5}" width="{main_w:.1f}" '
                     f'height="13" rx="2"/>')
        if p90 and p90 > (p50 or 0):
            parts.append(f'<line class="whisker" x1="{bar_x + main_w:.1f}" y1="{y + 11.5}" '
                         f'x2="{bar_x + whisk_w:.1f}" y2="{y + 11.5}"/>')
        parts.append(f'<text class="value" x="{bar_x + whisk_w + 8:.1f}" y="{y + 17}" '
                     f'font-family="sans-serif">{_esc(_fmt_ms(p50))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _bench_table(model: str, runs: list[dict[str, Any]], baseline: float | None) -> str:
    rows_html = []
    for r in runs:
        ok = r.get("status", "ok") == "ok"
        m = r.get("metrics", {})
        ss = m.get("steady_state_ms", {})
        label = collect.device_label(r)
        sim = ' <span class="sim">SIM</span>' if collect.is_simulated(r) else ""
        if ok:
            sp = collect.speedup(r, baseline)
            sp_html = (f'<td class="num speed">{sp:.2f}x</td>'
                       if sp else '<td class="num">-</td>')
            rows_html.append(
                f"<tr><td>{_esc(label)}{sim}</td>"
                f'<td class="num">{_fmt_ms(collect.p50(r))}</td>'
                f'<td class="num">{_fmt_ms(ss.get("p90"))}</td>'
                f'<td class="num">{_fmt_ms(ss.get("p99"))}</td>'
                f'<td class="num">{m.get("throughput_fps", "-")}</td>'
                f"<td>{_esc(r.get('backend', {}).get('runtime', ''))}</td>{sp_html}</tr>"
            )
        else:
            rows_html.append(
                f'<tr class="err"><td>{_esc(label)}{sim}</td>'
                f'<td colspan="6">{_esc(r.get("error", "failed"))[:160]}</td></tr>'
            )
    return (
        f"<table><thead><tr><th>Device</th><th class='num'>p50</th>"
        f"<th class='num'>p90</th><th class='num'>p99</th>"
        f"<th class='num'>inf/s</th><th>Backend</th><th class='num'>vs CPU</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _demo_cards(demo_runs: list[dict[str, Any]]) -> str:
    cards = []
    for r in demo_runs:
        label = collect.device_label(r)
        sim = ' <span class="sim">SIM</span>' if collect.is_simulated(r) else ""
        outputs = r.get("demo", {}).get("outputs", {})
        chain_parts = []
        for k, v in outputs.get("latency_chain", {}).items():
            chain_parts.append(f"{_esc(k)} <b>{v:.1f} ms</b>" if isinstance(v, (int, float))
                               else f"{_esc(k)} <b>{_esc(v)}</b>")
        chain = f'<div class="chain">{" &middot; ".join(chain_parts)}</div>' if chain_parts else ""
        body = ""
        if "text" in outputs:
            body += f"<pre class='output'>{_esc(outputs['text'])}</pre>"
        if "top" in outputs:
            lines = [f"{i}: {item.get('label', item.get('class', '?'))} "
                     f"({item.get('score', 0):.3f})"
                     for i, item in enumerate(outputs["top"], 1)]
            body += f"<pre class='output'>{_esc(chr(10).join(lines))}</pre>"
        if outputs.get("note"):
            body += f"<div class='note'>{_esc(outputs['note'])}</div>"
        cards.append(
            f"<div class='card'><b>{_esc(label)}</b>{sim} "
            f"<span class='meta'>&middot; {_esc(r['model']['name'])} &middot; "
            f"{_esc(r.get('timestamp', ''))}</span>"
            f"{chain}{body}</div>"
        )
    return "".join(cards)


def render_html(records: list[dict[str, Any]]) -> str:
    grouped = collect.group_bench(records)
    demos = collect.group_demos(records)
    baselines = collect.baseline_map(grouped)
    devices = sorted({collect.device_id(r) for r in records})
    n_sim = sum(1 for r in records if collect.is_simulated(r))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = [f"<h1>MyriadForge report</h1><div class='sub'>generated {generated} &middot; "
                f"{len(records)} records &middot; {len(devices)} devices "
                f"({n_sim} simulated runs, amber bars/SIM tags)</div>"]

    sections.append("<h2>Benchmarks</h2>")
    for model, runs in grouped.items():
        baseline = baselines.get(model)
        baseline_note = (f" &middot; baseline host-cpu p50 {_fmt_ms(baseline)}"
                         if baseline else "")
        sections.append(f"<h3>{_esc(model)}{baseline_note}</h3>")
        sections.append(_bench_table(model, runs, baseline))
        rows = [
            (collect.device_label(r), collect.p50(r),
             (r.get("metrics", {}).get("steady_state_ms", {}) or {}).get("p90"),
             collect.is_simulated(r))
            for r in runs
        ]
        sections.append(f"<div class='charts'>{_bar_chart(rows)}</div>")

    if demos:
        sections.append("<h2>Demo views</h2>")
        for name, runs in demos.items():
            sections.append(f"<h3>{_esc(name)} ({len(runs)} run(s))</h3>")
            sections.append(_demo_cards(runs))
    else:
        sections.append("<h2>Demo views</h2><p class='note'>No demo records yet - "
                        "run `myriadforge demo run &lt;name&gt; --device &lt;id&gt;`.</p>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>MyriadForge report</title><style>{_CSS}</style></head>"
        f"<body>{''.join(sections)}</body></html>"
    )
