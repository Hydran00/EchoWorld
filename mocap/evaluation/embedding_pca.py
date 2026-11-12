"""Plot complete held-out episodes in one shared 2-D PCA or t-SNE of image embeddings.

Run from EchoWorld/: python -m mocap.evaluation.embedding_pca --checkpoint runs/.../best.pt
Add --method tsne for t-SNE (default: PCA), or --sampling start-middle-end for three frames per episode.
Each point encodes an observed frame, not a predicted rollout. No pair sampler,
augmentation, brightness filter or episode cap is applied.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mocap.processing.dataset import assign_splits
from mocap.evaluation.evaluate import load_model


def sample_frames(timestamps_ns, segment_ids, interval_seconds=1., sampling='interval'):
    """Sample a time grid, or the first, temporal middle and last frame per episode.

    Three-point sampling selects the nearest interior frame to the temporal
    midpoint, when available. Episodes with one or two frames keep all frames.

    Restart the grid after tracking changes or recording gaps longer than the
    interval. Return separate parts so plots never draw across these gaps.
    Integer relative timestamps avoid losing precision on epoch nanoseconds.
    """
    if sampling not in ('interval', 'start-middle-end'):
        raise ValueError('Sampling must be interval or start-middle-end')
    times = np.asarray(timestamps_ns)
    segments = np.asarray(segment_ids)
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError('Sampling interval must be finite and positive')
    step = round(interval_seconds * 1e9)
    if step < 1:
        raise ValueError('Sampling interval must be at least one nanosecond')
    if (times.ndim != 1 or not len(times) or segments.shape != times.shape
            or not np.issubdtype(times.dtype, np.integer) or np.any(np.diff(times) <= 0)):
        raise ValueError('Expected nonempty, strictly increasing integer timestamps and matching segment IDs')
    breaks = np.flatnonzero((segments[1:] != segments[:-1]) | (np.diff(times) > step)) + 1
    bounds = np.r_[0, breaks, len(times)]
    selected = None
    if sampling == 'start-middle-end':
        if len(times) < 3:
            selected = np.arange(len(times))
        else:
            relative = times - times[0]
            middle = 1 + int(np.argmin(np.abs(relative[1:-1] - int(relative[-1]) // 2)))
            selected = np.array([0, middle, len(times) - 1])
    parts = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if selected is not None:
            indices = selected[(selected >= start) & (selected < stop)]
            if len(indices):
                parts.append(indices.astype(np.int64))
            continue
        relative = times[start:stop] - times[start]
        grid = np.arange(0, int(relative[-1]) + 1, step, dtype=np.int64)
        right = np.searchsorted(relative, grid).clip(0, len(relative) - 1)
        left = (right - 1).clip(0)
        nearest = np.where(grid - relative[left] <= relative[right] - grid, left, right)
        indices = np.unique(np.r_[0, nearest, len(relative) - 1]) + start
        parts.append(indices.astype(np.int64))
    return parts


@torch.inference_mode()
def extract_embeddings(model, manifest_path, split, device, interval_seconds=1.,
                       batch_size=32, representation='encoder', split_options=None, sampling='interval'):
    """Encode every episode in the split, including short and dark episodes.

    split_options: split_mode / split_fractions / split_seed of the checkpoint (default: leave-out)."""
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    if representation not in ('encoder', 'target'):
        raise ValueError('Representation must be encoder or target')
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported cache schema')
    splits = assign_splits(manifest, **(split_options or {}))
    entries = sorted((e for e in manifest['episodes'] if splits[e['episode_id']] == split), key=lambda e: e['episode_id'])
    if not entries:
        raise ValueError(f'No episodes in {split}')
    model.eval()
    size = model.config['image_size']
    features, rows, episodes = [], [], []
    for ei, entry in enumerate(entries):
        folder = manifest_path.parent / entry['path']
        with np.load(folder / 'poses.npz', allow_pickle=False) as data:
            times, segments = data['timestamps_ns'], data['segment_ids']
        images = np.load(folder / 'images.npy', mmap_mode='r', allow_pickle=False)
        if images.ndim != 3 or images.dtype != np.uint8 or len(images) != len(times):
            raise ValueError(f'Corrupt image cache: {entry["episode_id"]}')
        parts = sample_frames(times, segments, interval_seconds, sampling)
        indices = np.concatenate(parts)
        middle_frame = int(indices[1]) if sampling == 'start-middle-end' and len(indices) == 3 else None
        for start in range(0, len(indices), batch_size):
            pixels = np.array(images[indices[start:start + batch_size]], copy=True)
            batch = torch.from_numpy(pixels).to(device=device, dtype=torch.float32)[:, None] / 255.
            if batch.shape[-2:] != (size, size):
                batch = F.interpolate(batch, (size, size), mode='bilinear', align_corners=False)
            tokens = (model.encoder(model.encoder_input(batch)) if representation == 'encoder'
                      else model.target_latent(batch))
            embedding = tokens.mean(dim=1).float().cpu().numpy()
            if not np.isfinite(embedding).all():
                raise FloatingPointError(f'Non-finite embeddings: {entry["episode_id"]}')
            features.append(embedding)
        label = f'E{ei + 1:02d}'
        for part_id, part in enumerate(parts):
            for frame in part:
                rows.append(dict(episode_index=ei, label=label, episode_id=entry['episode_id'],
                    group=entry['group'], part=part_id, segment_id=int(segments[frame]),
                    frame_index=int(frame), timestamp_ns=int(times[frame]),
                    mean_intensity=float(images[frame].mean()),
                    time_seconds=float((times[frame] - times[0]) / 1e9),
                    is_start=bool(frame == 0), is_middle=bool(frame == middle_frame), is_end=bool(frame == len(times) - 1),
                    is_part_start=bool(frame == part[0]), is_part_end=bool(frame == part[-1])))
        episodes.append(dict(label=label, episode_id=entry['episode_id'], group=entry['group'],
            source_frames=len(times), sampled_frames=len(indices), parts=len(parts),
            duration_seconds=float((times[-1] - times[0]) / 1e9),
            start_mean_intensity=float(images[0].mean()), end_mean_intensity=float(images[-1].mean())))
        print(f'[{ei + 1}/{len(entries)}] {entry["episode_id"]}: {len(indices)} embeddings, {len(parts)} parts', flush=True)
    return np.concatenate(features), rows, episodes


def fit_pca(embeddings):
    from sklearn.decomposition import PCA

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if (embeddings.ndim != 2 or min(embeddings.shape) < 2
            or not np.isfinite(embeddings).all()):
        raise ValueError('PCA requires at least two finite embeddings with at least two dimensions')
    if not np.any(embeddings != embeddings[0]):
        raise ValueError('All embeddings are identical; PCA has zero total variance')
    # One centered fit over ALL sampled frames. No whitening, L2 normalization,
    # per-feature standardization or independent per-episode coordinate systems.
    pca = PCA(n_components=2, svd_solver='full')
    return pca.fit_transform(embeddings), pca


def fit_tsne(embeddings, perplexity=30., seed=42):
    """One reproducible t-SNE over all frames, with at most 50 input PCA dimensions.

    The preliminary PCA uses the original embeddings, never the 2-D PCA plot.
    No whitening, feature standardization or L2 normalization is applied.
    """
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if (embeddings.ndim != 2 or embeddings.shape[0] < 3 or embeddings.shape[1] < 2
            or not np.isfinite(embeddings).all()):
        raise ValueError('t-SNE requires at least three finite embeddings with at least two dimensions')
    if not np.any(embeddings != embeddings[0]):
        raise ValueError('All embeddings are identical; t-SNE has no structure to visualize')
    if not math.isfinite(perplexity) or not 0 < perplexity < len(embeddings):
        raise ValueError(f't-SNE perplexity must be positive and smaller than {len(embeddings)} samples; '
                         'choose a smaller --perplexity')
    dimensions = min(50, len(embeddings) - 1, embeddings.shape[1])
    inputs, retained_variance = embeddings, 1.
    if dimensions < embeddings.shape[1]:
        pca = PCA(n_components=dimensions, svd_solver='full')
        inputs = pca.fit_transform(embeddings)
        retained_variance = float(pca.explained_variance_ratio_.sum())
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto',
                random_state=seed, metric='euclidean', method='barnes_hut', n_jobs=4)
    coordinates = tsne.fit_transform(inputs)
    if not np.isfinite(coordinates).all():
        raise FloatingPointError('Non-finite t-SNE coordinates')
    info = dict(parameters=tsne.get_params(), sklearn_version=sklearn.__version__,
                input_dimensions=dimensions, preprocessing='centered PCA, full SVD' if inputs is not embeddings else 'none',
                preprocessing_explained_variance_ratio=retained_variance,
                kl_divergence=float(tsne.kl_divergence_), n_iter=int(tsne.n_iter_))
    return coordinates, info


def write_plots(output, z, rows, episodes, variance, representation, interval_seconds, split,
                method='pca', tsne_info=None, sampling='interval'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plotly.graph_objects as go

    colors = [matplotlib.colors.to_hex(plt.get_cmap('tab20')(i % 20)) for i in range(len(episodes))]
    axes = ('PC1', 'PC2') if method == 'pca' else ('t-SNE 1', 't-SNE 2')
    x_label, y_label = ([f'{axis} ({100 * v:.1f}% variance)' for axis, v in zip(axes, variance)]
                        if method == 'pca' else axes)
    episode_index = np.array([r['episode_index'] for r in rows])
    part_index = np.array([r['part'] for r in rows])
    sampling_label = 'start / middle / end' if sampling == 'start-middle-end' else f'{interval_seconds:g} s'
    title = (f'{split.capitalize()} embedding trajectories | {"PCA" if method == "pca" else "t-SNE"} | '
             f'{representation} | {sampling_label}')
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(left=.07, right=.66, bottom=.1, top=.88 if method == 'pca' else .85)
    legend = []
    interactive = go.Figure()

    for ei, (episode, color) in enumerate(zip(episodes, colors)):
        selected = np.flatnonzero(episode_index == ei)
        first, last = selected[0], selected[-1]
        legend.append(Line2D([0], [0], color=color, linewidth=2,
            label=f'{episode["label"]} · {episode["episode_id"].split("/")[-1]}'))
        for part_id in np.unique(part_index[selected]):
            ids = selected[part_index[selected] == part_id]
            xy = z[ids]
            ax.plot(*xy.T, color=color, alpha=.75, linewidth=1.5)
            ax.scatter(*xy.T, color=color, s=18, alpha=.75)
            # An arrow on the final nonzero displacement shows time direction.
            movement = np.flatnonzero(np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-10)
            if len(movement):
                j = movement[-1]
                ax.annotate('', xy=xy[j + 1], xytext=xy[j],
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
            custom = [[rows[i]['episode_id'], rows[i]['time_seconds'], rows[i]['frame_index'],
                       rows[i]['segment_id'], rows[i]['mean_intensity']] for i in ids]
            interactive.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='lines+markers',
                name=f'{episode["label"]} · {episode["episode_id"].split("/")[-1]}',
                legendgroup=episode['label'], showlegend=bool(part_id == 0),
                line=dict(color=color, width=1.5), marker=dict(size=5, color=color),
                customdata=custom,
                hovertemplate='%{customdata[0]}<br>t=%{customdata[1]:.3f} s · frame %{customdata[2]}'
                    '<br>segment %{customdata[3]} · mean intensity %{customdata[4]:.1f}/255'
                    f'<br>{axes[0]}=%{{x:.3f}} · {axes[1]}=%{{y:.3f}}<extra></extra>'))
        ax.scatter(*z[first], s=85, facecolors='white', edgecolors=color, linewidths=1.8, zorder=4)
        ax.scatter(*z[last], s=105, marker='X', c=[color], edgecolors='white', linewidths=.6, zorder=5)
        markers = [(first, 'circle-open', 'Start'), (last, 'x', 'End')]
        middle = next((i for i in selected if rows[i].get('is_middle') is True), None)
        if middle is not None:
            ax.scatter(*z[middle], s=70, marker='D', c=[color], edgecolors='white', linewidths=.6, zorder=4)
            markers.insert(1, (middle, 'diamond', 'Middle'))
        for index, symbol, name in markers:
            interactive.add_trace(go.Scatter(x=[z[index, 0]], y=[z[index, 1]], mode='markers',
                name=f'{name} {episode["label"]}', legendgroup=episode['label'], showlegend=False,
                marker=dict(symbol=symbol, size=12, color=color, line=dict(width=2)),
                text=[f'{name} · {episode["episode_id"]}<br>t={rows[index]["time_seconds"]:.3f} s'],
                hovertemplate='%{text}<extra></extra>'))

    marker_legend = [Line2D([0], [0], marker='o', color='gray', markerfacecolor='white',
                           linestyle='None', label='Start'),
                     Line2D([0], [0], marker='X', color='gray', linestyle='None', label='End')]
    if sampling == 'start-middle-end':
        marker_legend.insert(1, Line2D([0], [0], marker='D', color='gray', linestyle='None', label='Middle'))
    ax.set(xlabel=x_label, ylabel=y_label, aspect='equal')
    ax.grid(alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    endpoints = np.array([z[np.flatnonzero(episode_index == ei)[-1]] for ei in range(len(episodes))])
    center = (endpoints.min(axis=0) + endpoints.max(axis=0)) / 2
    radius = max(float(np.ptp(endpoints, axis=0).max()) * .8, .05)
    end_xlim = [float(center[0] - radius), float(center[0] + radius)]
    end_ylim = [float(center[1] - radius), float(center[1] + radius)]
    # Anchor the legend to the figure: equal-aspect axes can be short and would
    # otherwise push the final episode labels below the image.
    fig.legend(handles=legend + marker_legend, loc='upper left', bbox_to_anchor=(.68, .88),
               fontsize=8.5, title=f'All {split} trajectories', frameon=False, labelspacing=1.1)
    if method == 'pca':
        projection_note = f'PC1 + PC2 = {100 * sum(variance):.1f}% variance'
    else:
        parameters = tsne_info['parameters']
        projection_note = (f'perplexity {parameters["perplexity"]:g} · seed {parameters["random_state"]}\n'
                           'Local neighborhoods; inter-cluster distances and path lengths are not preserved')
    sampling_note = ('First, temporal middle and last frame per episode' if sampling == 'start-middle-end'
                     else 'True segment endpoints included')
    note = (f'{len(episodes)} episodes · {len(z)} observed frames · '
            f'{projection_note}\n'
            f'{sampling_note}; lines break at tracking/recording gaps')
    fig.suptitle(f'{title}\n{note}', x=.37, y=.97, fontsize=12)
    fig.savefig(output / f'trajectories_{method}.png', dpi=180)
    fig.savefig(output / f'trajectories_{method}.pdf')
    plt.close(fig)
    interactive.update_layout(title=dict(text=f'{title}<br><sup>{note.replace(chr(10), " · ")}</sup>'),
        template='plotly_white', xaxis_title=x_label, yaxis_title=y_label,
        yaxis=dict(scaleanchor='x', scaleratio=1), height=850,
        legend=dict(title='Click to hide · double-click to isolate', groupclick='togglegroup'),
        hovermode='closest', margin=dict(t=150 if method == 'pca' else 180),
        updatemenus=[dict(type='buttons', direction='right', x=0, y=1.12,
            buttons=[dict(label='All trajectories', method='relayout',
                          args=[{'xaxis.autorange': True, 'yaxis.autorange': True}]),
                     dict(label='Final positions (zoom)', method='relayout',
                          args=[{'xaxis.range': end_xlim, 'yaxis.range': end_ylim,
                                 'xaxis.autorange': False, 'yaxis.autorange': False}])])])
    # Bundle Plotly so the artifact is usable offline with no external service.
    interactive.write_html(output / f'trajectories_{method}.html', include_plotlyjs=True,
                           config=dict(displaylogo=False, scrollZoom=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=['test', 'val'], default='test')
    parser.add_argument('--interval-seconds', type=float, default=1.,
                        help='Time-grid interval and threshold for recording gaps, in seconds')
    parser.add_argument('--sampling', choices=['interval', 'start-middle-end'], default='interval',
                        help='Sample a time grid or only the first, temporal middle and last frame per episode')
    parser.add_argument('--representation', choices=['encoder', 'target'], default='encoder',
                        help='Mean patch tokens: online encoder (default), or EMA target with its LayerNorm')
    parser.add_argument('--method', choices=['pca', 'tsne'], default='pca')
    parser.add_argument('--perplexity', type=float, default=30.,
                        help='t-SNE neighborhood parameter; must be smaller than the sampled frame count')
    parser.add_argument('--seed', type=int, default=42, help='t-SNE random seed')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size < 1 or not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0:
        parser.error('Batch size and interval must be positive and finite')
    if args.method == 'tsne' and (not math.isfinite(args.perplexity) or args.perplexity <= 0):
        parser.error('t-SNE perplexity must be positive and finite')
    sampling_suffix = 'start_middle_end' if args.sampling == 'start-middle-end' else f'{args.interval_seconds:g}s'
    output = args.output or args.checkpoint.parent / f'{args.method}_{args.split}_{args.representation}_{sampling_suffix}'
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output directory is not empty: {output}; choose a new --output')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    model, state = load_model(args.checkpoint, device)
    # Diffusion checkpoints also contain the frozen world model.
    world = model.world if state.get('kind') == 'diffusion' else model
    manifest = args.manifest or Path(state['config']['data']['manifest'])
    if args.manifest is None and args.split == 'val':
        manifest = Path(state['config'].get('validation_manifest', manifest))
    split_options = {k: v for k, v in state['config']['data'].items()
                     if k in ('split_mode', 'split_fractions', 'split_seed')}
    embeddings, rows, episodes = extract_embeddings(world, manifest, args.split, device,
        args.interval_seconds, args.batch_size, args.representation, split_options, args.sampling)
    variance, tsne_info = None, None
    if args.method == 'pca':
        z, pca = fit_pca(embeddings)
        variance = pca.explained_variance_ratio_
        projection_arrays = dict(pca_coordinates=z, pca_mean=pca.mean_, pca_components=pca.components_,
                                 explained_variance=pca.explained_variance_, explained_variance_ratio=variance)
        projection_summary = dict(pca_fit='all sampled frames, centered only, full SVD',
                                  explained_variance_ratio=variance.tolist())
        coordinate_columns = ('pc1', 'pc2')
    else:
        z, tsne_info = fit_tsne(embeddings, args.perplexity, args.seed)
        projection_arrays = dict(tsne_coordinates=z)
        projection_summary = dict(tsne_fit=tsne_info)
        coordinate_columns = ('tsne1', 'tsne2')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'embeddings_{args.method}.npz', embeddings=embeddings, **projection_arrays,
        episode_index=np.array([r['episode_index'] for r in rows]),
        frame_index=np.array([r['frame_index'] for r in rows]),
        timestamps_ns=np.array([r['timestamp_ns'] for r in rows]),
        part=np.array([r['part'] for r in rows]))
    with (output / 'samples.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], *coordinate_columns])
        writer.writeheader()
        writer.writerows(dict(row, **dict(zip(coordinate_columns, map(float, xy)))) for row, xy in zip(rows, z))
    summary = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_step=state['step'],
        checkpoint_kind=state.get('kind', 'world_model'), manifest=str(manifest.resolve()),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), split=args.split,
        representation=args.representation, pooling='mean over all patch tokens',
        input='observed cached frames, no augmentation or intensity filter', precision='float32',
        interval_seconds=args.interval_seconds, sampling_mode=args.sampling,
        sampling=('first, nearest interior frame to temporal midpoint, last; no lines across gaps'
                  if args.sampling == 'start-middle-end' else 'nearest timestamp per contiguous segment, plus true endpoints'),
        projection_method=args.method, embedding_dim=embeddings.shape[1],
        sampled_frames=len(rows), episode_count=len(episodes),
        episodes=episodes, **projection_summary)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_plots(output, z, rows, episodes, variance,
                args.representation, args.interval_seconds, args.split, args.method, tsne_info, args.sampling)
    print(f'Saved {len(rows)} embeddings from all {len(episodes)} {args.split} episodes to {output}', flush=True)
    if args.method == 'pca':
        print(f'PC1 + PC2 explain {100 * sum(variance):.2f}% of variance', flush=True)
    else:
        print(f't-SNE: perplexity {args.perplexity:g}, seed {args.seed}, '
              f'KL divergence {tsne_info["kl_divergence"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
