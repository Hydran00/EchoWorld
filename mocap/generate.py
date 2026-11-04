"""Generate ultrasound from one context image and measured/planned probe poses."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from .evaluate import load_model
from .geometry import relative_action
from .prepare import preprocess


def validate_trajectory(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError('Expected finite (N+1,4,4) world_T_probe matrices, including the context pose')
    rotation = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5) or
            not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4) or
            not np.allclose(np.linalg.det(rotation), 1., atol=1e-4)):
        raise ValueError('Trajectory contains invalid rigid transforms')
    return poses


@torch.no_grad()
def generate(model, context, poses, device, rollout=False):
    # poses[0] is the pose of the context image. In anchored mode (default), frame i is generated
    # from the context image and the motion poses[0] -> poses[i]; there is no recurrent state.
    # In rollout mode, frame i is generated from frame i-1 and the motion poses[i-1] -> poses[i],
    # so errors can accumulate along the trajectory.
    poses = validate_trajectory(poses)
    model.eval()
    initial = context.to(device)
    current = initial
    images = []
    for i in range(1, len(poses)):
        action = torch.from_numpy(relative_action(poses[i - 1] if rollout else poses[0], poses[i])).to(device)[None]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            prediction = model.predict(current if rollout else initial, action)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Non-finite generated image')
        images.append(prediction[0, 0].float().cpu().numpy())
        current = prediction.float()
    return np.stack(images)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--episode', type=Path, help='Cached episode containing images.npy and poses.npz')
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--frames', type=int, default=12)
    p.add_argument('--context', type=Path, help='Raw ultrasound PNG/JPEG for an external trajectory')
    p.add_argument('--context-is-preprocessed', action='store_true')
    p.add_argument('--trajectory', type=Path, help='NPZ with world_T_probe, in metres; first matrix is the context pose')
    p.add_argument('--rollout', action='store_true', help='Feed generated frames back into the model; default anchors to the initial context')
    args = p.parse_args()
    if args.frames < 1 or args.stride < 1 or args.start_index < 0:
        p.error('Frame count/stride must be positive and start index nonnegative')
    if bool(args.episode) == bool(args.context) or (args.context and not args.trajectory):
        p.error('Use --episode OR --context with --trajectory')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, state = load_model(args.checkpoint, device)
    if state.get('kind') != 'diffusion':
        p.error('Pixel generation needs a decoder trained on the frozen world model: pass a mocap.diffusion checkpoint')
    size = state['config']['model']['image_size']
    targets = None
    if args.episode:
        cache = np.load(args.episode / 'images.npy', mmap_mode='r', allow_pickle=False)
        indices = args.start_index + np.arange(args.frames + 1) * args.stride
        if indices[-1] >= len(cache):
            p.error('Requested sequence exceeds this episode')
        with np.load(args.episode / 'poses.npz', allow_pickle=False) as f:
            segment = f['segment_ids'][indices[0]:indices[-1] + 1]
            if len(np.unique(segment)) != 1:
                p.error('Requested sequence crosses a tracking gap')
            poses = f['world_T_probe'][indices]
        context = np.asarray(Image.fromarray(cache[indices[0]]).resize((size, size), Image.Resampling.BILINEAR)).copy()
        targets = [Image.fromarray(cache[i]).resize((size, size), Image.Resampling.BILINEAR) for i in indices[1:]]
    else:
        image = Image.open(args.context)
        if args.context_is_preprocessed:
            context = np.asarray(ImageOps.pad(image.convert('L'), (size, size), method=Image.Resampling.BILINEAR)).copy()
        else:
            context = preprocess(np.asarray(image), tuple(state['preprocessing']['crop']), size).copy()
        with np.load(args.trajectory, allow_pickle=False) as f:
            poses = f['world_T_probe']
    tensor = torch.from_numpy(context).float()[None, None] / 255.
    generated = generate(model, tensor, poses, device, args.rollout)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'generated.npy', generated, allow_pickle=False)
    frames = [Image.fromarray((im.clip(0, 1) * 255).round().astype(np.uint8)) for im in generated]
    Image.fromarray(context).save(args.output / 'context.png')
    for i, im in enumerate(frames):
        im.save(args.output / f'frame_{i:04d}.png')
    frames[0].save(args.output / 'generated.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)
    strip = Image.new('L', (size * len(frames), size * (2 if targets else 1)))
    for i, im in enumerate(frames):
        strip.paste(im, (size * i, 0))
        if targets:
            strip.paste(targets[i], (size * i, size))
    strip.save(args.output / 'sequence.png')
    (args.output / 'generation.json').write_text(json.dumps(dict(checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_step=state['step'], mode='autoregressive' if args.rollout else 'anchored', frames=len(frames),
        source_episode=str(args.episode) if args.episode else None,
        start_index=args.start_index, stride=args.stride, trajectory_units='metres',
        sequence_rows=['generated', 'ground_truth'] if targets else ['generated']), indent=2) + '\n')
    print(f'Saved {len(frames)} generated frames to {args.output}')


if __name__ == '__main__':
    main()
