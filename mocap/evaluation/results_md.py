"""Append a results section to RESULTS.md from world-model and decoder metrics.json files.

    python -m mocap.evaluation.results_md --title "Canonical-frame poses" --note "..." \
        --world "Random (`mocap_canonical_random`)=runs/mocap_canonical_random/eval_val_20pairs/metrics.json" \
        --decoder "Random (`diffusion_canonical_random`)=runs/diffusion_canonical_random/eval_best/metrics.json" \
        --append RESULTS.md

World-model files come from `mocap.evaluation.evaluate` on a world checkpoint (recall, losses);
decoder files from the same command on a decoder checkpoint (val forward, reverse_motion, test).
"""
import argparse
import json
from pathlib import Path

METHODS = [('model', 'model'), ('zero_motion', 'zero motion'), ('copy_context', 'copy context')]
DECODER_METHODS = [('model', 'model'), ('copy_context', 'copy context'), ('zero_motion', 'zero motion')]


def pairs(items):
    out = []
    for item in items or []:
        label, _, path = item.rpartition('=')
        out.append((label, json.loads(Path(path).read_text()), path))
    return out


def section(title, note, worlds, decoders):
    L = [f'# {title}', '']
    if note:
        L += [note, '']
    if worlds:
        n = {m['pairs'] for _, m, _ in worlds}
        L += [f'## World model (validation, {"/".join(map(str, sorted(n)))} pairs)', '',
              '| Run | | R@1 ↑ | R@5 ↑ | R@10 ↑ |', '|---|---|---|---|---|']
        for label, m, _ in worlds:
            for key, name in METHODS:
                L.append(f'| {label if key == "model" else ""} | {name} | ' +
                         ' | '.join(f'{m[f"{key}_recall@{k}"]:.3f}' for k in (1, 5, 10)) + ' |')
        L += ['', '| Run | val loss ↓ | spatial ↓ | motion ↓ | pairs |', '|---|---|---|---|---|']
        for label, m, _ in worlds:
            L.append(f'| {label} | {m["val_loss"]:.3f} | {m["val_spatial"]:.3f} | {m["val_motion"]:.3f} | {m["pairs"]} |')
        L.append('')
    if decoders:
        L += ['## Decoder (generated image vs real frame)', '']
        for title_, get in [('Validation, forward motion', lambda m: m), ('Validation, reverse motion', lambda m: m.get('reverse_motion')),
                            ('Test, forward motion', lambda m: m.get('test'))]:
            rows = [(label, get(m)) for label, m, _ in decoders if get(m)]
            if not rows:
                continue
            L += [f'### {title_}', '', '| Run | | MAE ↓ | MAE fg ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |', '|---|---|---|---|---|---|---|']
            for label, s in rows:
                for key, name in DECODER_METHODS:
                    L.append(f'| {label if key == "model" else ""} | {name} | {s[key + "_mae"]:.4f} | {s[key + "_foreground_mae"]:.4f} | '
                             f'{s[key + "_mse"]:.4f} | {s[key + "_psnr"]:.2f} | {s[key + "_ssim"]:.3f} |')
            L.append('')
            for label, s in rows:
                L.append(f'- {label}: {s["pairs"]} pairs, mean Δt {s["mean_pair_dt_seconds"]:.2f} s, mean translation '
                         f'{s["mean_translation_mm"]:.1f} mm, mean rotation {s["mean_rotation_rad"] * 57.2958:.1f}°, '
                         f'motion sensitivity {s["motion_sensitivity_mae"]:.4f}')
            L.append('')
    L += ['## Files', ''] + [f'- {label}: `{path}`' for label, _, path in worlds + decoders] + ['']
    return '\n'.join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--title', required=True)
    p.add_argument('--note', default='')
    p.add_argument('--world', action='append', help='"label=path/to/metrics.json" (world-model evaluation)')
    p.add_argument('--decoder', action='append', help='"label=path/to/metrics.json" (decoder evaluation)')
    p.add_argument('--append', type=Path, help='append to this markdown file (otherwise print)')
    args = p.parse_args()
    text = section(args.title, args.note, pairs(args.world), pairs(args.decoder))
    if args.append:
        with args.append.open('a') as f:
            f.write('\n\n' + text)
        print(f'Appended "{args.title}" to {args.append}')
    else:
        print(text)


if __name__ == '__main__':
    main()
