"""EchoWorld probe-guidance fine-tuning on the IFL recordings (paper Sec. 3.3, released finetune/).

Task. Every episode is a scan that ends on the same standard view (kidney and liver both visible).
The last valid frame of each episode is therefore its *standard plane*: one plane per episode,
instead of the ten cardiac planes of EchoWorld. For every frame the label is the probe motion that
reaches that plane, expressed in the current probe frame:

    label_i = inv(world_T_i) world_T_goal   ->  [translation mm (3), rotation vector deg (3)]

The goal image is used only to define the labels. At inference the model sees the current frame and
its past frames together with their measured relative poses, never the goal.

Model (released finetune --model gnn --gnn_cfg geomlp, MultiFrameClassifierGNN):

    N frames [B, N, 1, S, S] --ViT-S/16 (world-model target encoder)--> mean token  [B, N, 384]
    pairwise motion rel[i, j] = inv(T_i) T_j [B, N, N, 6] --action encoder--> [B, N, N, 384]
        (initialized from the pre-trained policy_net, same input normalization)
    motion-aware attention (GeometricAttentionMLP): node i attends to node j with key/value
        MLP([f_j ; A(rel_ij)])                                   -> [B, N, 384]
    MLP head per node                                            -> [B, N, 6] = goal in frame i

All N nodes are supervised (smooth L1 in mm/deg, as released). At evaluation the prediction of
the current (last) frame is reported as "policy"; the "ensemble" composes every node prediction with
the known motion from the current frame to that node, inv(T_last) T_n (inv(T_n) T_goal)_pred, and
averages the N resulting estimates (translation mean, chordal rotation mean). The released code
reports the ensemble for the GNN model; it is the selection metric here as well.

History sampling follows released MultiFrameDatasetPolicy with ar_train: the current frame is the
last node; with probability seq_prob the N-1 past frames are exponentially spaced in the past
(denser close to the current frame, exp_alpha), otherwise uniformly drawn. Only the causal direction
is used: the released reverse direction would place frames closer to the goal in the history, and
its AR mask would discard the single goal plane anyway.

Subcommands:
    train     --config configs/guidance.json --output runs/<run> [--resume runs/<run>/last.pt]
    evaluate  --checkpoint runs/<run>/best.pt [--split val|test]
    goals     --config configs/guidance.json --output <folder>   (contact sheet of the goal frames)
"""
import argparse
import copy
import hashlib
import json
import math
from functools import partial
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation
from mocap.processing.dataset import assign_splits
from mocap.processing.geometry import relative_action
from mocap.pretraining.model import MotionWorldModel
from mocap.processing.trims import outside_fan_mean
from mocap.pretraining.train import StepBatches, check_split, same_run
from pretrain.src.models.vision_transformer import VisionTransformer


# --------------------------------------------------------------------------------------------- data

def to_label(action):
    """[..., 6] (mm, rad) -> (mm, deg): the units of the released labels and loss."""
    action = np.array(action, dtype=np.float32, copy=True)
    action[..., 3:] = np.rad2deg(action[..., 3:])
    return action


def pairwise_actions(poses):
    """[N, 4, 4] world_T_probe -> [N, N, 6] with [i, j] = inv(T_i) T_j (mm, rad); the released act_mode 'full'."""
    n = len(poses)
    source = np.broadcast_to(poses[:, None], (n, n, 4, 4)).reshape(-1, 4, 4)
    target = np.broadcast_to(poses[None, :], (n, n, 4, 4)).reshape(-1, 4, 4)
    return relative_action(source, target).reshape(n, n, 6)


def exponential_history(t, k, alpha):
    """Released MultiFrameDatasetPolicy.sample_timesteps_exponential: k indices in [0, t], ascending."""
    lambd = alpha * k / (t + 1)
    q = (np.arange(1, k + 1) - .5) / k
    samples = np.maximum(t + np.log(1 - q) / lambd, 0)
    return np.round(samples).astype(int)[::-1]


class GuidanceSequences(Dataset):
    """N-frame causal histories labelled with the motion to the episode's final (standard) view.

    Item: images [N, 1, S, S] in [0, 1], rel [N, N, 6] (mm, rad; rel[i, j] = inv(T_i) T_j),
    labels [N, 6] (mm, deg; goal in frame i), plus traceability fields.
    """
    def __init__(self, manifest, split='train', image_size=224, num_frames=8, min_mean=8., exp_alpha=.4,
                 seq_prob=.5, goal_offset_s=0., min_episode_s=2., eval_stride=10, color_jitter=.2, max_outside_fan_mean=20.,
                 split_mode='random', split_fractions=(.7, .15, .15), split_seed=42, **_):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest['schema_version'] != 1:
            raise ValueError('Unsupported cache schema')
        if num_frames < 1:
            raise ValueError('num_frames must be positive')
        self.split, self.image_size, self.num_frames = split, image_size, num_frames
        self.exp_alpha, self.seq_prob, self.color_jitter = exp_alpha, seq_prob, color_jitter
        self.episodes, self._images = [], {}
        splits = assign_splits(self.manifest, split_mode, split_fractions, split_seed)
        for entry in self.manifest['episodes']:
            if splits[entry['episode_id']] != split:
                continue
            folder = self.manifest_path.parent / entry['path']
            with np.load(folder / 'poses.npz', allow_pickle=False) as f:
                poses, times = f['world_T_probe'], f['timestamps_ns']
            images = np.load(folder / 'images.npy', mmap_mode='r', allow_pickle=False)
            if len(images) != len(times) or len(poses) != len(times) or not np.isfinite(poses).all():
                raise ValueError(f'Corrupt cache: {entry["episode_id"]}')
            # Same brightness filter as pre-training (probe lifted / no contact), plus a rejection of
            # non-ultrasound screens (e.g. the scanner wallpaper recorded at the end of nardi6):
            # outside the fan the upper corners of a real frame are black.
            valid = np.flatnonzero((images.mean(axis=(1, 2)) >= min_mean) &
                                   (outside_fan_mean(images) <= max_outside_fan_mean))
            if not len(valid):
                continue
            # Goal: last valid frame, optionally goal_offset_s before it (to skip the release phase).
            # Poses share the tracker frame for the whole episode, so the label is defined across
            # tracking gaps too; only frames with a measured pose exist in the cache.
            last = times[valid[-1]] - round(goal_offset_s * 1e9)
            valid = valid[times[valid] <= last]
            if not len(valid) or (times[valid[-1]] - times[valid[0]]) / 1e9 < min_episode_s:
                continue
            goal = int(valid[-1])
            self.episodes.append(dict(entry, folder=folder, poses=poses, times=times, valid=valid, goal=goal,
                                      labels=to_label(relative_action(poses[valid], poses[goal][None]))))
        if not self.episodes:
            raise ValueError(f'No usable episodes in {split}')
        # Validation/test: every eval_stride-th valid frame of every episode is a current frame,
        # with a deterministic exponential history (released validate_ar, stride 10).
        self.items = [(ei, int(p)) for ei, e in enumerate(self.episodes)
                      for p in range(0, len(e['valid']), eval_stride)]

    def __len__(self):
        # Training samples episodes uniformly (released 'case' sampling); evaluation enumerates items.
        return len(self.episodes) if self.split == 'train' else len(self.items)

    def image(self, episode, index):
        if episode not in self._images:
            self._images[episode] = np.load(self.episodes[episode]['folder'] / 'images.npy', mmap_mode='r', allow_pickle=False)
        tensor = torch.from_numpy(np.array(self._images[episode][index], copy=True)).float().unsqueeze(0) / 255.
        if tensor.shape[-1] != self.image_size:
            tensor = F.interpolate(tensor[None], (self.image_size, self.image_size), mode='bilinear', align_corners=False)[0]
        return tensor

    def jitter(self, image, generator):
        # Released transform: RandomApply(ColorJitter(brightness=cj, contrast=cj), p=0.6), per frame.
        # No geometric augmentation: crops would change the image/probe-motion relation.
        if not self.color_jitter or torch.rand((), generator=generator) >= .6:
            return image
        b, c = (1 + self.color_jitter * (2 * torch.rand(2, generator=generator) - 1)).tolist()
        image = (image * b).clamp(0, 1)
        return ((image - image.mean()) * c + image.mean()).clamp(0, 1)

    def history(self, pivot, generator):
        k = self.num_frames - 1
        if k == 0:
            return [pivot]
        if self.split == 'train' and torch.rand((), generator=generator) >= self.seq_prob:
            past = torch.randint(pivot + 1, (k,), generator=generator).sort().values.tolist()
        else:
            past = exponential_history(pivot, k, self.exp_alpha).tolist()
        return past + [pivot]

    def __getitem__(self, index):
        generator = None
        if isinstance(index, tuple):
            index, sample_seed = index
            generator = torch.Generator().manual_seed(sample_seed)
        if self.split == 'train':
            ei = index % len(self.episodes)
            pivot = int(torch.randint(len(self.episodes[ei]['valid']), (), generator=generator))
        else:
            ei, pivot = self.items[index]
        episode = self.episodes[ei]
        nodes = self.history(pivot, generator)                  # positions in episode['valid']
        frames = episode['valid'][nodes]
        poses = episode['poses'][frames]
        rel = pairwise_actions(poses)                              # [N, N, 6]
        images = [self.image(ei, int(f)) for f in frames]
        if self.split == 'train':
            images = [self.jitter(x, generator) for x in images]
        times = episode['times']
        return dict(images=torch.stack(images), rel=torch.from_numpy(rel),
                    labels=torch.from_numpy(episode['labels'][nodes]), episode=ei, frame=int(frames[-1]),
                    time_to_goal_s=float((times[episode['goal']] - times[frames[-1]]) / 1e9))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_images'] = {}
        return state


# -------------------------------------------------------------------------------------------- model

class GeometricAttentionMLP(nn.Module):
    """Released finetune/models/gnn.py GeometricAttentionMLP (motion-aware attention), unchanged."""
    def __init__(self, feature_dim, rel_pos_dim, num_heads=4):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, feature_dim // num_heads
        mlp = lambda i: nn.Sequential(nn.Linear(i, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
        self.query_linear = mlp(feature_dim)
        self.key_linear = mlp(feature_dim + rel_pos_dim)
        self.value_linear = mlp(feature_dim + rel_pos_dim)
        self.output_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, nodes, rel, return_attn=False):
        # nodes [B, N, D]; rel [B, N, N, R] with rel[:, i, j] describing node j as seen from node i.
        b, n, _ = nodes.shape
        q = self.query_linear(nodes).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, N, d]
        kv = torch.cat([nodes.unsqueeze(1).expand(-1, n, -1, -1), rel], dim=-1)                  # [B, N(i), N(j), D+R]
        k = self.key_linear(kv).view(b, n, n, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.value_linear(kv).view(b, n, n, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        attn = torch.einsum('bhid,bhijd->bhij', q, k).div(self.head_dim ** .5).softmax(-1)
        out = torch.einsum('bhij,bhijd->bhid', attn, v).transpose(1, 2).reshape(b, n, -1)
        out = nodes + self.output_linear(out)
        return (out, attn) if return_attn else out


class GuidanceModel(nn.Module):
    def __init__(self, world_config, fusion='geomlp', drop_path=.2, num_planes=1, num_heads=4):
        super().__init__()
        if fusion not in ('geomlp', 'none'):
            raise ValueError('fusion must be geomlp or none')
        dim = world_config['embed_dim']
        self.config = dict(world=world_config, fusion=fusion, drop_path=drop_path, num_planes=num_planes, num_heads=num_heads)
        self.feature_model = VisionTransformer(img_size=[world_config['image_size']], patch_size=16, in_chans=3,
                                               embed_dim=dim, depth=world_config['encoder_depth'],
                                               num_heads=world_config['num_heads'], drop_path_rate=drop_path,
                                               norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.action_encoder = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.attn = GeometricAttentionMLP(dim, dim, num_heads) if fusion == 'geomlp' else None
        # Released build_proj('mlp') head; one 6-DoF output per standard plane.
        self.head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 6 * num_planes))
        # Same normalization as the pre-training motion input (policy_net), so its weights transfer.
        self.register_buffer('action_scale', torch.tensor(world_config.get('action_scale', (30., 30., 30., .3, .3, .3))))

    def load_world(self, state, backbone='target_encoder'):
        """Backbone and action encoder from a MotionWorldModel state dict."""
        prefix = backbone + '.'
        self.feature_model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        prefix = 'predictor.policy_net.'
        self.action_encoder.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)

    def forward(self, images, rel, zero_motion=False):
        # images [B, N, 1, S, S] in [0, 1]; rel [B, N, N, 6] (mm, rad) -> [B, N, 6 * planes] (mm, deg).
        b, n = images.shape[:2]
        feats = self.feature_model(MotionWorldModel.encoder_input(images.flatten(0, 1))).mean(1).view(b, n, -1)
        if self.attn is not None:
            rel = torch.zeros_like(rel) if zero_motion else rel / self.action_scale
            feats = self.attn(feats, self.action_encoder(rel))
        return self.head(feats)


def param_groups(model, weight_decay, layer_decay):
    """Released param_groups_lrd: lr scale layer_decay^(L + 1 - layer_id); no decay on 1-D/bias/pos_embed."""
    depth = len(model.feature_model.blocks)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('feature_model.'):
            sub = name[len('feature_model.'):]
            layer = int(sub.split('.')[1]) + 1 if sub.startswith('blocks.') else 0 if sub.startswith(('patch_embed', 'pos_embed')) else depth + 1
        else:
            layer = depth + 1
        decay = not (p.ndim == 1 or name.endswith('.bias') or 'pos_embed' in name)
        key = (layer, decay)
        groups.setdefault(key, dict(params=[], weight_decay=weight_decay if decay else 0., lr_scale=layer_decay ** (depth + 1 - layer)))
        groups[key]['params'].append(p)
    return list(groups.values())


def lr_at(step, config):
    # Released adjust_learning_rate: linear warm-up, half-cycle cosine to min_lr.
    warmup, total = config['warmup_steps'], config['steps']
    if step < warmup:
        return config['lr'] * (step + 1) / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return config['min_lr'] + (config['lr'] - config['min_lr']) * .5 * (1 + math.cos(math.pi * progress))


# ------------------------------------------------------------------------------------------ metrics

def to_matrix(vec):
    """[..., 6] (mm, deg) -> [..., 4, 4] (mm)."""
    vec = np.asarray(vec, dtype=np.float64)
    out = np.zeros(vec.shape[:-1] + (4, 4))
    out[..., :3, :3] = Rotation.from_rotvec(np.deg2rad(vec[..., 3:]).reshape(-1, 3)).as_matrix().reshape(vec.shape[:-1] + (3, 3))
    out[..., :3, 3], out[..., 3, 3] = vec[..., :3], 1
    return out


def ensemble(preds, rel):
    """preds [B, N, 6] (goal in node frames, mm/deg), rel [B, N, N, 6] (mm, rad) -> [B, 6] in the last frame."""
    to_node = to_matrix(to_label(rel[:, -1]))                       # inv(T_last) T_n            [B, N, 4, 4]
    estimates = to_node @ to_matrix(preds)                           # inv(T_last) T_goal per node [B, N, 4, 4]
    translation = estimates[..., :3, 3].mean(1)
    u, _, vt = np.linalg.svd(estimates[..., :3, :3].sum(1))          # chordal L2 rotation mean
    d = np.sign(np.linalg.det(u @ vt))
    u[:, :, -1] *= d[:, None]
    rotation = Rotation.from_matrix(u @ vt).as_rotvec(degrees=True)
    return np.concatenate([translation, rotation], 1).astype(np.float32)


def errors(pred, label):
    """Per-sample error summary of [B, 6] predictions (mm, deg)."""
    ae = np.abs(pred - label)
    rot = (Rotation.from_rotvec(np.deg2rad(pred[:, 3:])).inv() * Rotation.from_rotvec(np.deg2rad(label[:, 3:]))).magnitude()
    cos = (pred[:, :3] * label[:, :3]).sum(1) / (np.linalg.norm(pred[:, :3], axis=1) * np.linalg.norm(label[:, :3], axis=1) + 1e-6)
    return dict(ae=ae, trans=np.linalg.norm(pred[:, :3] - label[:, :3], axis=1), rot=np.rad2deg(rot), cos=cos)


def summarize(e, prefix):
    ae = e['ae'].mean(0)
    out = {f'{prefix}_mae': float(ae.mean()), f'{prefix}_trans_mae_mm': float(ae[:3].mean()),
           f'{prefix}_rot_mae_deg': float(ae[3:].mean()), f'{prefix}_trans_err_mm': float(e['trans'].mean()),
           f'{prefix}_rot_err_deg': float(e['rot'].mean()), f'{prefix}_direction_cos': float(e['cos'].mean())}
    out.update({f'{prefix}_ae_{k}': float(v) for k, v in zip(['x', 'y', 'z', 'rx', 'ry', 'rz'], ae)})
    return out


DISTANCE_BINS = (0, 20, 50, 100, 200, float('inf'))


@torch.no_grad()
def evaluate_guidance(model, data, device, output=None, batch_size=32, amp=True, train_mean=None, workers=4):
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=workers)
    policy, ens, zero_motion, labels, meta = [], [], [], [], []
    for batch in loader:
        images, rel = batch['images'].to(device, non_blocking=True), batch['rel'].to(device).float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            preds = model(images, rel)[..., :6].float().cpu().numpy()
            preds_zero = model(images, rel, zero_motion=True)[..., -1:, :6].float().cpu().numpy() if model.attn is not None else preds[:, -1:]
        policy.append(preds[:, -1])
        ens.append(ensemble(preds, batch['rel'].numpy()))
        zero_motion.append(preds_zero[:, 0])
        labels.append(batch['labels'][:, -1].numpy())
        meta += [dict(episode=data.episodes[int(e)]['episode_id'], frame=int(f), time_to_goal_s=float(t))
                 for e, f, t in zip(batch['episode'], batch['frame'], batch['time_to_goal_s'])]
    model.train()
    policy, ens, zero_motion, labels = map(np.concatenate, (policy, ens, zero_motion, labels))
    results = {'policy': errors(policy, labels), 'ensemble': errors(ens, labels),
               'zero_motion_input': errors(zero_motion, labels),
               # Baselines: stay still (predict no motion); predict the mean training label.
               'baseline_stay': errors(np.zeros_like(labels), labels)}
    if train_mean is not None:
        results['baseline_train_mean'] = errors(np.broadcast_to(np.asarray(train_mean, np.float32), labels.shape).copy(), labels)
    metrics = dict(samples=len(labels))
    for name, e in results.items():
        metrics.update(summarize(e, name))
    # Error versus distance from the goal (the task is harder far from the standard view).
    distance = np.linalg.norm(labels[:, :3], axis=1)
    for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        sel = (distance >= lo) & (distance < hi)
        if sel.any():
            tag = f'{lo:g}-{hi:g}mm'.replace('inf', 'up')
            metrics[f'bin_{tag}_count'] = int(sel.sum())
            for name in ('ensemble', 'baseline_stay'):
                metrics[f'bin_{tag}_{name}_trans_err_mm'] = float(results[name]['trans'][sel].mean())
                metrics[f'bin_{tag}_{name}_rot_err_deg'] = float(results[name]['rot'][sel].mean())
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        with (output / 'predictions.jsonl').open('w') as f:
            for m, p, q, y in zip(meta, policy, ens, labels):
                f.write(json.dumps(dict(m, label=y.tolist(), policy=p.tolist(), ensemble=q.tolist())) + '\n')
    return metrics


def train_label_mean(data, samples=4096, seed=0):
    """Mean label of current frames under the training sampler (baseline)."""
    g = np.random.default_rng(seed)
    rows = [data.episodes[i]['labels'][g.integers(len(data.episodes[i]['valid']))]
            for i in g.integers(len(data.episodes), size=samples)]
    return np.mean(rows, 0).tolist()


# ---------------------------------------------------------------------------------------------- CLI

def resolve_config(config):
    """Inherit model, manifest and split from the world checkpoint (like mocap.decoder.diffusion)."""
    world = torch.load(config['world_checkpoint'], map_location='cpu', weights_only=False)
    if world.get('kind') == 'diffusion':
        raise ValueError('world_checkpoint must be a world-model checkpoint (mocap.pretraining.train), not a decoder')
    config['model'] = world['config']['model']
    data = {k: v for k, v in world['config']['data'].items() if k in ('manifest', 'min_mean', 'split_mode', 'split_fractions', 'split_seed')}
    data.setdefault('split_mode', 'leave-out')
    world_split = data['split_mode']
    data.update(config.get('data', {}))
    if data['split_mode'] != world_split:
        print(f'Warning: guidance split {data["split_mode"]!r} differs from the world-model split {world_split!r}; '
              'validation/test episodes may have been seen in pre-training', flush=True)
    config['data'] = data
    return config, world


def build(config, world=None):
    model = GuidanceModel(config['model'], **config['guidance'])
    if world is not None:
        model.load_world(world['model'], config.get('backbone', 'target_encoder'))
    return model


def datasets(config, splits=('train', 'val')):
    common = dict(config['data'], image_size=config['model']['image_size'], **config['sequence'])
    return [GuidanceSequences(split=s, **common) for s in splits]


def train(args):
    config = json.loads(Path(args.config).read_text())
    config, world = resolve_config(config)
    device = torch.device(args.device)
    torch.set_num_threads(config.get('cpu_threads', 4))
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and args.resume is None:
        raise FileExistsError('Run already has a checkpoint; use --resume or a new output directory')
    check_split(config['data'], config['data']['manifest'])
    manifest_hash = hashlib.sha256(Path(config['data']['manifest']).read_bytes()).hexdigest()
    world_hash = hashlib.sha256(Path(config['world_checkpoint']).read_bytes()).hexdigest()
    train_data, val_data = datasets(config)
    config['train_label_mean'] = train_label_mean(train_data)
    model = build(config, world).to(device)
    groups = param_groups(model, config['weight_decay'], config['layer_decay'])
    optimizer = torch.optim.AdamW(groups, lr=config['lr'], betas=(.9, .999), eps=1e-6)
    step, best, stale = 0, float('inf'), 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (not same_run(state['config'], config) or state['manifest_sha256'] != manifest_hash or
                state['world_sha256'] != world_hash):
            raise ValueError('Resume config, dataset or world checkpoint changed')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, best, stale = state['step'], state['best_score'], state['stale_validations']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        print(f'Resumed guidance at step {step}', flush=True)
    stop_at = min(config['steps'], args.max_steps) if args.max_steps else config['steps']
    if stop_at <= step:
        raise ValueError('Stop step must exceed the saved step')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'run_info.json').write_text(json.dumps(dict(torch=torch.__version__, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        train_episodes=len(train_data.episodes), val_episodes=len(val_data.episodes), val_sequences=len(val_data),
        manifest_sha256=manifest_hash, world_sha256=world_hash,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)), indent=2) + '\n')

    def save(path):
        state = dict(kind='guidance', format_version=1, config=config, step=step, best_score=best, stale_validations=stale,
                     manifest_sha256=manifest_hash, world_sha256=world_hash, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        temp = path.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(path)

    loader = DataLoader(train_data, batch_sampler=StepBatches(len(train_data), config['batch_size'], step, stop_at, config['seed']),
                        num_workers=config.get('workers', 4), pin_memory=device.type == 'cuda',
                        persistent_workers=config.get('workers', 4) > 0)
    stopping = []

    def stop(signum, _):
        stopping.append(signum)
        print(f'Received signal {signum}; saving after the current optimizer step.', flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    (output / 'pid.txt').write_text(str(__import__('os').getpid()) + '\n')
    amp = config.get('amp', True) and device.type == 'cuda'
    zero_motion = config.get('zero_motion', False)          # released --zero_act ablation
    pred_last = config.get('pred_last', False)              # released --pred_last
    model.train()
    started, start_step, early_stop = time.monotonic(), step, False
    print(f'Guidance: {len(train_data.episodes)} train episodes, {len(val_data)} val sequences '
          f'({len(val_data.episodes)} episodes) on {device}; steps {step} -> {stop_at}', flush=True)
    with (output / 'train.jsonl').open('a') as log:
        for batch in loader:
            lr = lr_at(step, config)
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_scale']
            images, rel, labels = (batch[k].to(device, non_blocking=True).float() for k in ('images', 'rel', 'labels'))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                preds = model(images, rel, zero_motion=zero_motion)[..., :6]
            if pred_last:
                preds, labels = preds[:, -1], labels[:, -1]
            loss = F.smooth_l1_loss(preds.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('clip_grad', 1.), error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % config.get('log_every', 50) == 0 or step == start_step + 1:
                with torch.no_grad():
                    last_ae = (preds[:, -1] if preds.ndim == 3 else preds).float().sub(labels[:, -1] if labels.ndim == 3 else labels).abs()
                row = dict(step=step, loss=float(loss.detach()),train_trans_mae_mm=float(last_ae[:, :3].mean()),
                           train_rot_mae_deg=float(last_ae[:, 3:].mean()), grad_norm=float(grad_norm), lr=lr,
                           seconds=time.monotonic() - started,
                           steps_per_second=(step - start_step) / (time.monotonic() - started))
                if device.type == 'cuda':
                    row['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
                log.write(json.dumps(row) + '\n'); log.flush()
                print(json.dumps(row), flush=True)
            if (step % config['eval_every'] == 0 or step == stop_at) and not stopping:
                metrics = evaluate_guidance(model, val_data, device, output / f'val_{step:06d}',
                                            config.get('eval_batch_size', 32), amp, config['train_label_mean'],
                                            config.get('workers', 4))
                print('Validation ' + json.dumps(dict(step=step, **{k: v for k, v in metrics.items() if not k.startswith('bin_')})), flush=True)
                score = metrics[config.get('selection', 'ensemble_mae')]   # lower is better
                if score < best:
                    best, stale = score, 0
                    save(output / 'best.pt')
                else:
                    stale += 1
                if config.get('patience') and stale >= config['patience']:
                    early_stop = True
                    print(f'Early stopping after {stale} validations without improvement', flush=True)
            if step % config['save_every'] == 0 or step == stop_at or stopping or early_stop:
                save(output / 'last.pt')
            if stopping or early_stop:
                break
    print(f'Stopped cleanly at step {step}; checkpoint {output / "last.pt"}', flush=True)
    (output / 'status.json').write_text(json.dumps(dict(step=step, stop_signal=stopping[0] if stopping else None,
        early_stopping=early_stop, best_score=best, elapsed_seconds=time.monotonic() - started), indent=2) + '\n')


def load_guidance(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('kind') != 'guidance':
        raise ValueError('Not a guidance checkpoint')
    model = build(state['config'])
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), state


def evaluate_cli(args):
    device = torch.device(args.device)
    model, state = load_guidance(args.checkpoint, device)
    config = state['config']
    data, = datasets(config, (args.split,))
    output = args.output or args.checkpoint.parent / f'eval_{args.checkpoint.stem}_{args.split}'
    metrics = evaluate_guidance(model, data, device, output, args.batch_size, True, config['train_label_mean'], args.workers)
    lines = [f'# Guidance evaluation: {args.checkpoint} ({args.split}, step {state["step"]}, {metrics["samples"]} sequences)', '',
             '| Method | MAE (mean 6 comp.) | trans err mm | rot err deg | trans MAE mm | rot MAE deg | direction cos |',
             '|---|---|---|---|---|---|---|']
    for name in ('policy', 'ensemble', 'zero_motion_input', 'baseline_stay', 'baseline_train_mean'):
        if f'{name}_mae' in metrics:
            m = lambda k: metrics[f'{name}_{k}']
            lines.append(f'| {name} | {m("mae"):.2f} | {m("trans_err_mm"):.1f} | {m("rot_err_deg"):.1f} | '
                         f'{m("trans_mae_mm"):.1f} | {m("rot_mae_deg"):.1f} | {m("direction_cos"):.3f} |')
    lines += ['', '| Distance to goal | count | ensemble trans mm | stay trans mm | ensemble rot deg | stay rot deg |', '|---|---|---|---|---|---|']
    for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        tag = f'{lo:g}-{hi:g}mm'.replace('inf', 'up')
        if f'bin_{tag}_count' in metrics:
            m = lambda k: metrics[f'bin_{tag}_{k}']
            lines.append(f'| {tag} | {m("count")} | {m("ensemble_trans_err_mm"):.1f} | {m("baseline_stay_trans_err_mm"):.1f} | '
                         f'{m("ensemble_rot_err_deg"):.1f} | {m("baseline_stay_rot_err_deg"):.1f} |')
    (Path(output) / 'metrics.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def goals_cli(args):
    """Contact sheet of goal frames, to check that each episode really ends on the standard view."""
    from PIL import Image, ImageDraw
    config, _ = resolve_config(json.loads(Path(args.config).read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        data, = datasets(config, (split,))
        tiles = []
        for ei, e in enumerate(data.episodes):
            tile = Image.fromarray((data.image(ei, e['goal'])[0].numpy() * 255).astype(np.uint8)).convert('RGB')
            ImageDraw.Draw(tile).text((3, 3), e['episode_id'].split('/')[-1][4:20], fill=(255, 255, 0))
            tiles.append(tile)
        cols, size = 8, tiles[0].width
        sheet = Image.new('RGB', (cols * size, math.ceil(len(tiles) / cols) * size))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * size, (i // cols) * size))
        sheet.save(args.output / f'goals_{split}.png')
        (args.output / f'goals_{split}.json').write_text(json.dumps([dict(episode=e['episode_id'], goal_frame=e['goal'],
            duration_s=float((e['times'][e['goal']] - e['times'][e['valid'][0]]) / 1e9),
            start_distance_mm=float(np.linalg.norm(e['labels'][0, :3]))) for e in data.episodes], indent=2) + '\n')
        print(f'{split}: {len(tiles)} goal frames -> {args.output / f"goals_{split}.png"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--config', required=True)
    t.add_argument('--output', required=True, type=Path)
    t.add_argument('--resume', type=Path)
    t.add_argument('--max-steps', type=int)
    t.add_argument('--device', default='cuda:0')
    e = sub.add_parser('evaluate')
    e.add_argument('--checkpoint', required=True, type=Path)
    e.add_argument('--split', default='val', choices=['val', 'test'])
    e.add_argument('--output', type=Path)
    e.add_argument('--batch-size', type=int, default=32)
    e.add_argument('--workers', type=int, default=4)
    e.add_argument('--device', default='cuda:0')
    g = sub.add_parser('goals')
    g.add_argument('--config', required=True)
    g.add_argument('--output', required=True, type=Path)
    g.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = p.parse_args()
    dict(train=train, evaluate=evaluate_cli, goals=goals_cli)[args.command](args)


if __name__ == '__main__':
    main()
