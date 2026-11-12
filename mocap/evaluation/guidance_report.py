"""Offline HTML report of a guidance evaluation (mocap.finetuning.guidance evaluate output).

    .venv/bin/python -m mocap.evaluation.guidance_report --eval runs/<run>/eval_best_test

Reads metrics.json and predictions.jsonl from the evaluation folder and the checkpoint config
(to locate the frames), and writes <eval>/report.html: headline numbers, error by distance from the
goal (model vs "stay still"), and an episode explorer with true vs predicted motion to the goal over
time; hovering a time point shows the ultrasound frame the prediction was made on. The page embeds
plotly.js and the frames, so it opens offline (copy it to a laptop and open it in a browser).
"""
import argparse
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs
from scipy.spatial.transform import Rotation
from mocap.finetuning.guidance import DISTANCE_BINS, datasets


def jpeg(pixels, size=160):
    image = Image.fromarray(np.asarray(pixels)).resize((size, size), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=80)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def angle(rotvec_deg):
    return float(np.linalg.norm(rotvec_deg))


def build(eval_dir, checkpoint=None):
    eval_dir = Path(eval_dir)
    metrics = json.loads((eval_dir / 'metrics.json').read_text())
    rows = [json.loads(line) for line in (eval_dir / 'predictions.jsonl').read_text().splitlines() if line]
    # eval_<ckpt-stem>_<split> next to the checkpoint, unless given explicitly.
    stem, split = eval_dir.name[len('eval_'):].rsplit('_', 1)
    checkpoint = Path(checkpoint) if checkpoint else eval_dir.parent / f'{stem}.pt'
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    data, = datasets(state['config'], (split,))
    by_id = {e['episode_id']: (i, e) for i, e in enumerate(data.episodes)}
    episodes, frames = {}, {}
    for row in rows:
        _, e = by_id[row['episode']]
        if row['episode'] not in frames:
            frames[row['episode']] = np.load(e['folder'] / 'images.npy', mmap_mode='r')
        images = frames[row['episode']]
        ep = episodes.setdefault(row['episode'], dict(id=row['episode'], points=[],
                                 first=jpeg(images[e['valid'][0]], 224), goal=jpeg(images[e['goal']], 224)))
        label, pred = np.array(row['label']), np.array(row['ensemble'])
        ep['points'].append(dict(t=-row['time_to_goal_s'], frame=row['frame'], image=jpeg(images[row['frame']]),
                                 label=label.round(2).tolist(), pred=pred.round(2).tolist(),
                                 dist=float(np.linalg.norm(label[:3])), pred_dist=float(np.linalg.norm(pred[:3])),
                                 ang=angle(label[3:]), pred_ang=angle(pred[3:]),
                                 trans_err=float(np.linalg.norm(label[:3] - pred[:3])),
                                 rot_err=float(np.rad2deg((Rotation.from_rotvec(np.deg2rad(pred[3:])).inv() *
                                                           Rotation.from_rotvec(np.deg2rad(label[3:]))).magnitude()))))
    for ep in episodes.values():
        ep['points'].sort(key=lambda p: p['t'])
        ep['trans_err'] = float(np.mean([p['trans_err'] for p in ep['points']]))
        ep['rot_err'] = float(np.mean([p['rot_err'] for p in ep['points']]))
        ep['start_dist'] = ep['points'][0]['dist']
    bins = []
    for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        tag = f'{lo:g}-{hi:g}mm'.replace('inf', 'up')
        if f'bin_{tag}_count' in metrics:
            bins.append(dict(label=f'{lo:g}–{hi:g} mm'.replace('–inf mm', '+ mm'), count=metrics[f'bin_{tag}_count'],
                             **{k: metrics[f'bin_{tag}_{k}'] for k in ('ensemble_trans_err_mm', 'baseline_stay_trans_err_mm',
                                                                       'ensemble_rot_err_deg', 'baseline_stay_rot_err_deg')}))
    payload = dict(run=str(checkpoint), split=split, step=state['step'], metrics=metrics, bins=bins,
                   episodes=sorted(episodes.values(), key=lambda e: e['id']))
    html = TEMPLATE.replace('/*PLOTLY*/', get_plotlyjs()).replace('/*DATA*/', json.dumps(payload))
    out = eval_dir / 'report.html'
    out.write_text(html)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Guidance Evaluation</title>
<style>
:root { --surface-0:#f4f4f2; --surface-1:#fcfcfb; --line:#e3e2de; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#8a8984; --series-1:#2a78d6; --series-2:#eb6834; --truth:#0b0b0b; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { --surface-0:#111110; --surface-1:#1a1a19;
  --line:#2e2e2c; --text-primary:#f0efec; --text-secondary:#c3c2b7; --muted:#8f8e86; --series-1:#3987e5; --series-2:#d95926; --truth:#f0efec; } }
:root[data-theme="dark"] { --surface-0:#111110; --surface-1:#1a1a19; --line:#2e2e2c; --text-primary:#f0efec;
  --text-secondary:#c3c2b7; --muted:#8f8e86; --series-1:#3987e5; --series-2:#d95926; --truth:#f0efec; }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface-0); color:var(--text-primary); font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:0 0 4px; }
.sub { color:var(--text-secondary); margin:0 0 20px; font-size:13px; word-break:break-all; }
.card { background:var(--surface-1); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:16px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:16px; }
.tile { background:var(--surface-1); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.tile .v { font-size:26px; font-weight:600; font-variant-numeric:tabular-nums; }
.tile .k { color:var(--text-secondary); font-size:12px; } .tile .c { color:var(--muted); font-size:12px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.explain { color:var(--text-secondary); font-size:13px; margin:0 0 8px; }
.explorer { display:grid; grid-template-columns:minmax(0,1fr) 250px; gap:16px; }
.frames { display:flex; flex-direction:column; gap:10px; }
.frames figure { margin:0; } .frames img { width:100%; border-radius:6px; background:#000; display:block; }
.frames figcaption { font-size:12px; color:var(--text-secondary); margin-top:3px; font-variant-numeric:tabular-nums; }
select { font:inherit; padding:6px 8px; border-radius:6px; border:1px solid var(--line); background:var(--surface-1); color:var(--text-primary); max-width:100%; }
.bar { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
.legend { display:flex; gap:14px; font-size:12px; color:var(--text-secondary); }
.legend i { display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:5px; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { padding:5px 8px; border-bottom:1px solid var(--line); text-align:right; } th:first-child, td:first-child { text-align:left; }
th { color:var(--text-secondary); font-weight:500; cursor:pointer; } tr.sel td { background:color-mix(in srgb, var(--series-1) 12%, transparent); }
tbody tr { cursor:pointer; }
@media (max-width:860px) { .grid2, .explorer { grid-template-columns:1fr; } }
</style>
<script>/*PLOTLY*/</script></head>
<body><div class="wrap">
<h1>Valutazione guidance</h1>
<p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>
<div class="card">
  <h2>Errore per distanza dal goal</h2>
  <p class="explain">Errore della predizione (ensemble) per il frame corrente, per fascia di distanza dal goal, confrontato con "resta fermo" (errore = distanza dal goal). Sotto ogni fascia: numero di sequenze.</p>
  <div class="grid2"><div id="binsT"></div><div id="binsR"></div></div>
</div>
<div class="card">
  <h2>Esploratore per episodio</h2>
  <p class="explain">Movimento verso il goal espresso nel frame della sonda corrente: valore vero e predetto, nel tempo che manca al goal (0 = vista finale). Passa sul grafico per vedere il frame ecografico su cui è stata fatta la predizione.</p>
  <div class="bar"><select id="ep"></select>
    <div class="legend"><span><i style="background:var(--truth)"></i>vero</span><span><i style="background:var(--series-1)"></i>predetto (ensemble)</span></div></div>
  <div class="explorer">
    <div><div id="dist"></div><div id="axes"></div></div>
    <div class="frames">
      <figure><img id="hoverImg" alt="frame corrente"><figcaption id="hoverCap">Passa sul grafico</figcaption></figure>
      <figure><img id="goalImg" alt="goal"><figcaption>Goal (ultimo frame del taglio)</figcaption></figure>
      <figure><img id="firstImg" alt="primo frame"><figcaption>Primo frame</figcaption></figure>
    </div>
  </div>
</div>
<div class="card">
  <h2>Episodi</h2>
  <p class="explain">Errore medio per episodio sui frame valutati. Clic su una riga per aprirla nell'esploratore; clic sulle intestazioni per ordinare.</p>
  <table id="table"><thead><tr><th data-k="id">Episodio</th><th data-k="n">Frame valutati</th><th data-k="start_dist">Distanza iniziale (mm)</th><th data-k="trans_err">Errore traslazione (mm)</th><th data-k="rot_err">Errore rotazione (°)</th></tr></thead><tbody></tbody></table>
</div>
</div>
<script>
const D = /*DATA*/;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 1) => v.toFixed(d).replace('.', ',');
const M = D.metrics;
document.getElementById('sub').textContent = `${D.run} · split ${D.split} · step ${D.step} · ${M.samples} sequenze, ${D.episodes.length} episodi`;
const tiles = [
  ['Errore traslazione', fmt(M.ensemble_trans_err_mm) + ' mm', `resta fermo: ${fmt(M.baseline_stay_trans_err_mm)} mm`],
  ['Errore rotazione', fmt(M.ensemble_rot_err_deg) + '°', `resta fermo: ${fmt(M.baseline_stay_rot_err_deg)}°`],
  ['Coseno direzione', fmt(M.ensemble_direction_cos, 2), '1 = direzione di traslazione corretta'],
  ['Movimenti azzerati', fmt(M.zero_motion_input_trans_err_mm) + ' mm', 'stesso modello senza storia dei movimenti'],
];
document.getElementById('tiles').innerHTML = tiles.map(([k, v, c]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="c">${c}</div></div>`).join('');

function layout(extra) {
  return Object.assign({ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: css('--text-secondary'), size: 12 },
    margin: { l: 52, r: 12, t: 30, b: 44 }, hovermode: 'x unified', showlegend: false,
    hoverlabel: { bgcolor: css('--surface-1'), bordercolor: css('--line'), font: { color: css('--text-primary') } } }, extra);
}
const axis = (title, extra) => Object.assign({ title: { text: title, font: { size: 12 } }, gridcolor: css('--line'), zerolinecolor: css('--line'), linecolor: css('--line') }, extra || {});
const cfg = { displayModeBar: false, responsive: true };

function bins(div, key, unit, title) {
  const x = D.bins.map(b => `${b.label}<br><span style="font-size:10px">n=${b.count}</span>`);
  Plotly.newPlot(div, [
    { type: 'bar', name: 'modello', x, y: D.bins.map(b => b['ensemble_' + key]), marker: { color: css('--series-1') },
      hovertemplate: `modello: %{y:.1f} ${unit}<extra></extra>` },
    { type: 'bar', name: 'resta fermo', x, y: D.bins.map(b => b['baseline_stay_' + key]), marker: { color: css('--series-2') },
      hovertemplate: `resta fermo: %{y:.1f} ${unit}<extra></extra>` },
  ], layout({ title: { text: title, font: { size: 13, color: css('--text-primary') }, x: 0 }, barmode: 'group', bargap: .3, bargroupgap: .08,
    showlegend: true, legend: { orientation: 'h', y: 1.14, x: 1, xanchor: 'right', font: { color: css('--text-secondary') } }, height: 320,
    margin: { l: 52, r: 12, t: 40, b: 52 }, yaxis: axis(unit), xaxis: axis('') }), cfg);
}
bins('binsT', 'trans_err_mm', 'mm', 'Traslazione');
bins('binsR', 'rot_err_deg', '°', 'Rotazione');

const sel = document.getElementById('ep');
sel.innerHTML = D.episodes.map((e, i) => `<option value="${i}">${e.id} — ${fmt(e.trans_err)} mm, ${fmt(e.rot_err)}°</option>`).join('');
let current = 0;
function hoverFrame(ep, i) {
  const p = ep.points[i];
  document.getElementById('hoverImg').src = p.image;
  document.getElementById('hoverCap').textContent = `frame ${p.frame} · ${fmt(-p.t)} s al goal · vero ${fmt(p.dist)} mm / ${fmt(p.ang)}° · predetto ${fmt(p.pred_dist)} mm / ${fmt(p.pred_ang)}°`;
}
function show(i) {
  current = i; sel.value = i;
  const ep = D.episodes[i], t = ep.points.map(p => p.t);
  document.getElementById('goalImg').src = ep.goal; document.getElementById('firstImg').src = ep.first;
  const truth = css('--truth'), pred = css('--series-1');
  const pair = (yT, yP, name, unit, xa, ya) => [
    { x: t, y: yT, mode: 'lines+markers', line: { color: truth, width: 2 }, marker: { size: 5 }, name: 'vero', xaxis: xa, yaxis: ya,
      hovertemplate: `${name} vero: %{y:.1f} ${unit}<extra></extra>` },
    { x: t, y: yP, mode: 'lines+markers', line: { color: pred, width: 2 }, marker: { size: 5 }, name: 'predetto', xaxis: xa, yaxis: ya,
      hovertemplate: `${name} predetto: %{y:.1f} ${unit}<extra></extra>` }];
  Plotly.newPlot('dist', [
    ...pair(ep.points.map(p => p.dist), ep.points.map(p => p.pred_dist), 'distanza', 'mm', 'x', 'y'),
    ...pair(ep.points.map(p => p.ang), ep.points.map(p => p.pred_ang), 'rotazione', '°', 'x2', 'y2'),
  ], layout({ height: 280, grid: { rows: 1, columns: 2, pattern: 'independent' },
    xaxis: axis('s al goal'), yaxis: axis('distanza dal goal (mm)', { rangemode: 'tozero' }),
    xaxis2: axis('s al goal'), yaxis2: axis('angolo dal goal (°)', { rangemode: 'tozero' }) }), cfg);
  const names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'], traces = [], extra = { grid: { rows: 2, columns: 3, pattern: 'independent' } };
  names.forEach((n, k) => {
    const a = k ? String(k + 1) : '', unit = k < 3 ? 'mm' : '°';
    traces.push(...pair(ep.points.map(p => p.label[k]), ep.points.map(p => p.pred[k]), n, unit, 'x' + a, 'y' + a));
    extra['xaxis' + a] = axis(k >= 3 ? 's al goal' : ''); extra['yaxis' + a] = axis(`${n} (${unit})`);
  });
  Plotly.newPlot('axes', traces, layout(Object.assign({ height: 420, margin: { l: 52, r: 12, t: 16, b: 44 } }, extra)), cfg);
  for (const div of ['dist', 'axes']) document.getElementById(div).on('plotly_hover', ev => hoverFrame(ep, ev.points[0].pointIndex));
  hoverFrame(ep, 0);
  document.querySelectorAll('#table tbody tr').forEach(tr => tr.classList.toggle('sel', +tr.dataset.i === i));
}
sel.onchange = () => show(+sel.value);

let sortKey = 'trans_err', sortDir = -1;
function table() {
  const rows = D.episodes.map((e, i) => ({ i, id: e.id, n: e.points.length, start_dist: e.start_dist, trans_err: e.trans_err, rot_err: e.rot_err }));
  rows.sort((a, b) => (a[sortKey] > b[sortKey] ? 1 : -1) * sortDir);
  document.querySelector('#table tbody').innerHTML = rows.map(r => `<tr data-i="${r.i}" class="${r.i === current ? 'sel' : ''}"><td>${r.id}</td><td>${r.n}</td><td>${fmt(r.start_dist)}</td><td>${fmt(r.trans_err)}</td><td>${fmt(r.rot_err)}</td></tr>`).join('');
  document.querySelectorAll('#table tbody tr').forEach(tr => tr.onclick = () => { show(+tr.dataset.i); document.getElementById('ep').scrollIntoView({ behavior: 'smooth', block: 'center' }); });
}
document.querySelectorAll('#table th').forEach(th => th.onclick = () => { const k = th.dataset.k; sortDir = k === sortKey ? -sortDir : -1; sortKey = k; table(); });
table();
show(D.episodes.indexOf(D.episodes.reduce((a, b) => (a.points.length >= b.points.length ? a : b))));
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval', required=True, type=Path, help='folder written by mocap.finetuning.guidance evaluate')
    p.add_argument('--checkpoint', type=Path, help='defaults to the checkpoint the folder name refers to')
    args = p.parse_args()
    print(build(args.eval, args.checkpoint))


if __name__ == '__main__':
    main()
