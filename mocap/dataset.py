"""Episode-local image/action pairs with a persisted group split.

Each sample is a triplet (I_a, I_b, a -> b motion) drawn from one recorded episode:

    source, target   [1, S, S] float in [0, 1]   preprocessed ultrasound frames (S = image_size)
    action           [6]                         inv(T_a) T_b: translation (mm), rotation vector (rad)
    reverse_action   [6]                         inv(T_b) T_a, used by the symmetric objective

Pairs never cross episode boundaries or tracking gaps, and the train/val/test split is made
by recording group before pairing, so that no frame of a held-out group is seen in training.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .geometry import relative_action


class MocapPairs(Dataset):
    def __init__(self, manifest, split='train', image_size=224, min_dt=.4, max_dt=3.,
                 min_mean=8., max_pairs_per_episode=0, seed=42, eval_dt=1., crop_scale=None):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest['schema_version'] != 1:
            raise ValueError('Unsupported cache schema')
        if min_dt <= 0 or max_dt < min_dt:
            raise ValueError('Invalid pair time interval')
        if image_size % 16:
            raise ValueError('Image size must be divisible by 16')
        self.split, self.image_size, self.seed = split, image_size, seed
        self.eval_dt, self.min_dt, self.max_dt = eval_dt, min_dt, max_dt
        # Training augmentation of the released config (RandomResizedCrop, scale 0.3-1.0); None disables it.
        self.crop_scale = tuple(crop_scale) if crop_scale else None
        if self.crop_scale and not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1:
            raise ValueError('crop_scale must satisfy 0 < min <= max <= 1')
        self.episodes, self.items, self._images = [], [], {}
        # Leakage guard: a group (folder name, approximately one subject/operator) must belong
        # to exactly one split; otherwise near-identical frames could appear in train and val.
        group_splits = {}
        seen = set()
        for entry in self.manifest['episodes']:
            group_splits.setdefault(entry['group'], set()).add(entry['split'])
            if entry['episode_id'] in seen:
                raise ValueError('Duplicate episode in manifest')
            seen.add(entry['episode_id'])
        if any(len(s) > 1 for s in group_splits.values()):
            raise ValueError('Group leakage across splits')
        for entry in self.manifest['episodes']:
            if entry['split'] != split:
                continue
            folder = self.manifest_path.parent / entry['path']
            with np.load(folder / 'poses.npz', allow_pickle=False) as f:
                poses, times, segments = f['world_T_probe'], f['timestamps_ns'], f['segment_ids']
            images = np.load(folder / 'images.npy', mmap_mode='r', allow_pickle=False)
            if len(images) != len(times) or len(poses) != len(times) or np.any(np.diff(times) <= 0):
                raise ValueError(f'Corrupt cache: {entry["episode_id"]}')
            if not np.isfinite(poses).all():
                raise ValueError('Non-finite poses')
            # Discard nearly black frames (probe lifted or not in contact). This is an intensity
            # heuristic on the cropped image, not an annotation of acoustic contact.
            valid = images.mean(axis=(1, 2)) >= min_mean
            episode = dict(entry, folder=folder, poses=poses, times=times, segments=segments)
            ei = len(self.episodes)
            self.episodes.append(episode)
            # For each valid source frame i, the admissible targets are the valid frames between
            # min_dt and max_dt seconds later, within the same tracking segment. Time, not frame
            # count, defines the interval, so the pairing is independent of the export frame rate.
            choices = []
            for i in np.flatnonzero(valid):
                left = np.searchsorted(times, times[i] + round(min_dt * 1e9), side='left')
                right = np.searchsorted(times, times[i] + round(max_dt * 1e9), side='right')
                targets = np.arange(left, right)
                targets = targets[valid[targets] & (segments[targets] == segments[i])]
                if len(targets):
                    choices.append((ei, int(i), targets))
            # Optional cap (used for validation) with evenly spaced sources across the episode.
            if max_pairs_per_episode and len(choices) > max_pairs_per_episode:
                choices = [choices[i] for i in np.linspace(0, len(choices) - 1, max_pairs_per_episode).astype(int)]
            self.items.extend(choices)
        if not self.items:
            raise ValueError(f'No valid pairs in {split}: check temporal interval, image filter and tracking')

    def __len__(self):
        return len(self.items)

    def image(self, episode, index):
        # Memory-mapped cache read: uint8 [S0, S0] -> float [1, S, S] in [0, 1]. Resizing only
        # occurs if the model resolution differs from the exported cache resolution.
        if episode not in self._images:
            self._images[episode] = np.load(self.episodes[episode]['folder'] / 'images.npy', mmap_mode='r', allow_pickle=False)
        pixels = np.array(self._images[episode][index], copy=True)
        tensor = torch.from_numpy(pixels).float().unsqueeze(0) / 255.
        if tensor.shape[-1] != self.image_size:
            tensor = torch.nn.functional.interpolate(tensor[None], (self.image_size, self.image_size), mode='bilinear', align_corners=False)[0]
        return tensor

    def crop_box(self, generator):
        # RandomResizedCrop parameters with torchvision defaults (area fraction in crop_scale,
        # aspect ratio log-uniform in [3/4, 4/3], ten attempts, then the full image).
        size = self.image_size
        for _ in range(10):
            area = size * size * (self.crop_scale[0] + (self.crop_scale[1] - self.crop_scale[0]) *
                                  torch.rand((), generator=generator).item())
            ratio = math.exp(math.log(3 / 4) + math.log(16 / 9) * torch.rand((), generator=generator).item())
            w, h = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
            if 0 < w <= size and 0 < h <= size:
                top = int(torch.randint(size - h + 1, (), generator=generator))
                left = int(torch.randint(size - w + 1, (), generator=generator))
                return top, left, h, w
        return 0, 0, size, size

    def crop(self, image, box):
        top, left, h, w = box
        patch = image[:, top:top + h, left:left + w]
        return torch.nn.functional.interpolate(patch[None], (self.image_size, self.image_size),
                                               mode='bilinear', align_corners=False)[0]

    def __getitem__(self, index):
        generator = None
        if isinstance(index, tuple):
            index, sample_seed = index
            generator = torch.Generator().manual_seed(sample_seed)
        ei, source, targets = self.items[index]
        # The sampler passes (index, seed): random target choices are then a function of the
        # global training step only, which makes interrupted and resumed runs identical.
        if self.split == 'train':
            target = int(targets[torch.randint(len(targets), (), generator=generator).item()])
            # Random temporal reversal: forward and backward probe motions are equally likely.
            if torch.rand((), generator=generator) < .5:
                source, target = target, source
        else:
            # Fixed, reproducible validation target, near one second if available.
            episode = self.episodes[ei]
            wanted = episode['times'][source] + round(self.eval_dt * 1e9)
            target = int(targets[np.argmin(abs(episode['times'][targets] - wanted))])
        poses = self.episodes[ei]['poses']
        action = relative_action(poses[source], poses[target])
        source_image, target_image = self.image(ei, source), self.image(ei, target)
        if self.split == 'train' and self.crop_scale:
            # One crop box per pair, shared by both frames: the two views keep the same pixel-to-probe
            # geometry, so the appearance change between them is still explained by the measured motion.
            box = self.crop_box(generator)
            source_image, target_image = self.crop(source_image, box), self.crop(target_image, box)
        return dict(source=source_image, target=target_image,
                    action=torch.from_numpy(action),
                    reverse_action=torch.from_numpy(relative_action(poses[target], poses[source])), episode=ei, source_index=source, target_index=target,
                    dt_seconds=float((self.episodes[ei]['times'][target] - self.episodes[ei]['times'][source]) / 1e9))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_images'] = {}
        return state
