"""Offline canonical trajectories with subject selection and doctor colours.

python -m mocap.visualization.canonical_dataset --manifest data/ifl_canonical/manifest.json \
    --body ../Canonical_US/outputs/probe_reconstruction_trims/shard0/canonical_smpl.ply

Reads the same canonical world_T_probe poses used for training. Participant labels
are viewer-only annotations; original groups, splits and arrays remain unchanged.
"""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np

from .participants import DEFAULT_MAPPING, load_mapping, recording_identity

HERE = Path(__file__).resolve().parent
DOCTOR_COLORS = {'Amir': '#8056d9', 'Davide': '#1476c9', 'Feng': '#15916a', 'Yuan': '#e47b20'}
EXTRA_COLORS = ['#d34d82', '#777319', '#535d8a', '#a65337']
CAMERA = dict(up=dict(x=0, y=1, z=0), eye=dict(x=0, y=0, z=2.2))


def display_indices(segments, stride):
    """Retain endpoints and both sides of segment boundaries when subsampling."""
    if not len(segments):
        return np.empty(0, dtype=int)
    boundaries = np.flatnonzero(np.diff(segments) != 0) + 1
    return np.unique(np.r_[np.arange(0, len(segments), stride), len(segments) - 1,
                           boundaries - 1, boundaries]).astype(int)


def prepare_groups(manifest_path, mapping, stride=3, axis_every=15, lift_mm=4.):
    if stride < 1 or axis_every < 1:
        raise ValueError('stride and axis_every must be positive')
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not manifest['episodes']:
        raise ValueError('The canonical manifest has no episodes')
    groups = defaultdict(lambda: dict(points=[], text=[], starts=[], ends=[],
                                     names=[], hover=[], axes=[], bounds=[]))
    for entry in sorted(manifest['episodes'], key=lambda e: e['episode_id']):
        identity = recording_identity(entry['episode_id'], mapping)
        doctor, subject = identity['doctor'], identity['subject']
        bag = identity['bag'].removeprefix('bag_')
        with np.load(manifest_path.parent / entry['path'] / 'poses.npz', allow_pickle=False) as saved:
            poses, segments = saved['world_T_probe'].copy(), saved['segment_ids'].copy()
        if not len(poses) or len(poses) != len(segments) or not np.isfinite(poses).all():
            raise ValueError(f'Empty or invalid canonical poses: {bag}')
        # Display lift only, retaining the same convention as the original viewer.
        poses[:, :3, 3] -= lift_mm / 1000 * poses[:, :3, 2]
        ids = display_indices(segments, stride)
        xyz = poses[ids, :3, 3]
        group = groups[(subject, doctor)]
        hover = f'{html.escape(bag)}<br>Soggetto: {html.escape(subject)}<br>Medico: {html.escape(doctor)}'
        for k, i in enumerate(ids):
            if k and segments[i] != segments[ids[k - 1]]:
                group['points'].append([None] * 3)
                group['text'].append('')
            group['points'].append(xyz[k].tolist())
            group['text'].append(hover)
        group['points'].append([None] * 3)
        group['text'].append('')
        group['starts'].append(poses[0, :3, 3].tolist())
        group['ends'].append(poses[-1, :3, 3].tolist())
        group['names'].append(bag)
        group['hover'].append(hover)
        group['bounds'].append(xyz)
        for i in ids[::axis_every]:
            origin, z = poses[i, :3, 3], poses[i, :3, 2]
            group['axes'].extend([origin.tolist(), (origin + .025 * z).tolist(), [None] * 3])
    return groups, len(manifest['episodes'])


def build_figure(manifest_path, body_path, mapping, *, stride=3, axis_every=15,
                 lift_mm=4., zoom=False, subject=None):
    import open3d as o3d
    import plotly.graph_objects as go

    groups, total = prepare_groups(manifest_path, mapping, stride, axis_every, lift_mm)
    mesh = o3d.io.read_triangle_mesh(str(body_path))
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(vertices) or not len(faces):
        raise ValueError('Canonical mesh is empty')
    subjects = sorted({s for s, _ in groups})
    default_subject = subject or ('Davide' if 'Davide' in subjects else subjects[0])
    if default_subject not in subjects:
        raise ValueError(f'Unknown subject {default_subject!r}; expected one of {subjects}')
    colors = dict(DOCTOR_COLORS)
    for i, doctor in enumerate(sorted({d for _, d in groups} - colors.keys())):
        colors[doctor] = EXTRA_COLORS[i % len(EXTRA_COLORS)]
    fig = go.Figure([go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='#c9d1d9', opacity=.35,
        name='Corpo canonico', hoverinfo='skip', showlegend=False,
        lighting=dict(ambient=.6, diffuse=.6))])
    metadata = []
    for (patient, doctor), group in sorted(groups.items()):
        show = patient == default_subject
        tint = colors[doctor]
        common = dict(legendgroup=doctor, showlegend=False, visible=show,
                      meta=dict(subject=patient, doctor=doctor))
        indices = {}
        for kind in ('path', 'start', 'end', 'axes'):
            indices[kind] = len(fig.data)
            points = group[{'path':'points', 'start':'starts', 'end':'ends', 'axes':'axes'}[kind]]
            columns = dict(zip(('x','y','z'), zip(*points)))
            if kind in ('path', 'axes'):
                options = dict(common)
                if kind == 'axes':
                    options['visible'] = False
                fig.add_trace(go.Scatter3d(
                    **columns, **options, name=f'{doctor} · {patient}', mode='lines',
                    line=dict(color=tint, width=4 if kind == 'path' else 2),
                    opacity=.9 if kind == 'path' else .65, connectgaps=False,
                    text=group['text'] if kind == 'path' else None,
                    hovertemplate='%{text}<extra></extra>' if kind == 'path' else None,
                    hoverinfo=None if kind == 'path' else 'skip'))
            else:
                fig.add_trace(go.Scatter3d(
                    **columns, **common, name=f'{doctor} · {patient}', mode='markers',
                    marker=dict(size=4 if kind == 'start' else 6,
                                symbol='circle' if kind == 'start' else 'diamond',
                                color='white' if kind == 'start' else tint,
                                line=dict(color=tint if kind == 'start' else '#203047', width=1.5)),
                    text=group['hover'],
                    hovertemplate=('Inizio' if kind == 'start' else 'Fine / goal') + '<br>%{text}<extra></extra>'))
        points = np.concatenate(group['bounds'])
        metadata.append(dict(subject=patient, doctor=doctor, count=len(group['names']),
                             episodes=group['names'], traces=indices,
                             low=points.min(0).tolist(), high=points.max(0).tolist()))
    body_bounds = dict(low=(vertices.min(0) - .04).tolist(), high=(vertices.max(0) + .04).tolist())
    fig.update_layout(
        template='plotly_white', autosize=True, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, paper_bgcolor='white', uirevision='canonical-participants',
        scene=dict(aspectmode='data', xaxis_title='X [m]', yaxis_title='Y [m]',
                   zaxis_title='Z [m]', camera=CAMERA,
                   xaxis=dict(range=[body_bounds['low'][0],body_bounds['high'][0]]),
                   yaxis=dict(range=[body_bounds['low'][1],body_bounds['high'][1]]),
                   zaxis=dict(range=[body_bounds['low'][2],body_bounds['high'][2]])))
    return fig, dict(groups=metadata, subjects=subjects, colors=colors,
                     default_subject=default_subject, zoom=bool(zoom), camera=CAMERA,
                     total=total, body_bounds=body_bounds, lift_mm=lift_mm)


def write_viewer(fig, metadata, output):
    # Initialize controls only once Plotly has finished creating the WebGL scene.
    packed = json.dumps(metadata, ensure_ascii=False).replace('<', '\u003c')
    script = (HERE / 'canonical_viewer.js').read_text().replace('__METADATA__', packed)
    chart = fig.to_html(full_html=False, include_plotlyjs=True, div_id='plot',
                        default_width='100%', default_height='100%', post_script=script,
                        config=dict(responsive=True, displaylogo=False, scrollZoom=True,
                                    toImageButtonOptions=dict(format='png', filename='traiettorie_per_medico', scale=2)))
    template = (HERE / 'canonical_viewer.html').read_text()
    lift = (f"Le curve sono sollevate di {metadata['lift_mm']:g} mm dalla pelle solo per la visualizzazione."
            if metadata['lift_mm'] else 'Nessuno spostamento di visualizzazione delle curve.')
    rendered = (template.replace('__TOTAL__', str(metadata['total']))
                .replace('__SUBJECTS__', str(len(metadata['subjects'])))
                .replace('__LIFT_NOTE__', lift).replace('__SCRIPT__', '')
                .replace('__PLOT__', chart))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.html.tmp')
    temp.write_text(rendered)
    temp.replace(output)
    print(f"{metadata['total']} episodes · {len(metadata['subjects'])} subjects -> {output}")
    for group in metadata['groups']:
        print(f"  {group['subject']:12s} / {group['doctor']:8s}: {group['count']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=Path('data/ifl_canonical/manifest.json'))
    parser.add_argument('--body', type=Path, required=True, help='Canonical body mesh')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--participants', type=Path, default=DEFAULT_MAPPING)
    parser.add_argument('--subject', help='Initially selected subject (default: Davide if available)')
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--axis-every', type=int, default=15)
    parser.add_argument('--zoom', action='store_true')
    parser.add_argument('--lift-mm', type=float, default=4., help='Display-only offset along -probe Z')
    args = parser.parse_args()
    fig, metadata = build_figure(args.manifest, args.body, load_mapping(args.participants),
                                stride=args.stride, axis_every=args.axis_every, lift_mm=args.lift_mm,
                                zoom=args.zoom, subject=args.subject)
    output = args.output or args.manifest.parent / ('canonical_trajectories_zoom.html' if args.zoom else 'canonical_trajectories.html')
    write_viewer(fig, metadata, output)


if __name__ == '__main__':
    main()
