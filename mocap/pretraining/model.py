"""EchoWorld pre-training on measured probe motion, following the released pretrain/main.py.

Spatial world modeling (I-JEPA masked-feature prediction) and motion world modeling
(SimCLR-style InfoNCE between the motion-conditioned prediction and the EMA target),
both symmetric over the image pair: L = L_spatial + 0.1 L_motion.

Data flow (shapes for the ViT-S configuration, 224x224 images, batch size B):

    I_a, I_b        [B, 1, 224, 224]  grayscale ultrasound frames with intensities in [0, 1]
    action          [B, 6]            relative probe motion a -> b: translation (mm) and rotation vector (rad)
    reverse_action  [B, 6]            relative probe motion b -> a, required by the symmetric objective

    encoder_input   [B, 3, 224, 224]  grayscale replicated over three channels and rescaled to [-1, 1]
    patch embedding [B, 196, 384]     196 = (224/16)^2 non-overlapping 16x16 patches, each projected to 384
    context encoder [B, K, 384]       only the K patches of the visible context block
    target encoder  [B, 196, 384]     complete image, EMA weights, no gradient

    Spatial: predictor(context tokens + positional mask tokens) -> [4B, M, 384],
             regressed onto the target features of the M patches in each of the four target blocks.
    Motion:  predictor(context tokens + one action token)       -> [B, 384],
             projected to [B, 1024] and contrasted (InfoNCE) with the projected, spatially averaged
             target features of the *other* image of the pair.

Probe motion is never an output of the model; it is a conditioning input only. The model
learns to anticipate the representation of the image that would be observed after the motion.

Optional anatomical conditioning (model config `anatomy`, canonical caches only; absent = the
original model, with identical parameters and checkpoints):

    attachment of the current frame   face f [B], barycentric (l1, l2, l3) [B, 3] on the canonical body
    anatomical embedding              e = l1 E[v1] + l2 E[v2] + l3 E[v3]        [B, 384]
                                      E [V, 384] learnable vertex embeddings, (v1, v2, v3) = faces[f]
                                      ("encoding": "fourier": e = MLP(Fourier(l1 v1 + l2 v2 + l3 v3)),
                                      fixed geometric encoding of the canonical position)
    anatomy token                     e + anatomy_type_token, appended after the context tokens

    predictor input (motion)          [context tokens; anatomy token; action token] -> action token output

Alternative (model config `anatomy_target`): no anatomical input; an auxiliary head regresses the
canonical probe position of each image from its online encoder features (smooth L1, weight `weight`,
positions in units of `scale` metres). The predictor and inference are those of the original model.

The 6-D action is unchanged. Only the attachment of the frame the prediction starts from enters the
predictor (a for a -> b, b for the reverse direction b -> a); the attachment of the predicted frame is
never an input. The spatial (I-JEPA) branch does not use it.
"""
import copy
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F

from pretrain.src.masks.multiblock import MaskCollator
from pretrain.src.masks.utils import apply_masks
from pretrain.src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from pretrain.src.utils.tensors import repeat_interleave_batch

# Released configs/cardiac_vits16_p384x6_simclr.yaml.
# - One context ("enc") block covering 85-100% of the patch grid, with unit aspect ratio.
# - Four target ("pred") blocks, each covering 15-20% of the grid, aspect ratio 0.75-1.5.
# - allow_overlap=False removes target patches from the context, so the encoder never
#   observes the content that the predictor is asked to infer.
# - min_keep is the minimum number of patches per block; it must be lowered for small grids
#   (e.g. 8x8 patches at 128 px), where target blocks contain fewer than ten patches.
DEFAULT_MASK = dict(enc_mask_scale=(.85, 1.), pred_mask_scale=(.15, .2), aspect_ratio=(.75, 1.5),
                    nenc=1, npred=4, min_keep=10, allow_overlap=False)


def info_nce(features, temperature=.1):
    """NT-Xent of released pretrain/main.py: row i and row i + N/2 are positives."""
    # features = [P(z_ab); P(z_ba); P'(h_b); P'(h_a)], each [B, 1024], hence [4B, 1024] in total.
    # Each prediction (row i) has its own target (row i + 2B) as the positive, and vice versa.
    # The remaining 4B - 2 rows, i.e. other predictions and the targets of other pairs, act as
    # negatives. The difficulty of the task therefore grows with the batch size.
    features = F.normalize(features.float(), dim=1)        # cosine similarity
    logits = (features @ features.T / temperature).masked_fill(
        torch.eye(len(features), dtype=torch.bool, device=features.device), float('-inf'))  # exclude self-pairs
    labels = torch.arange(len(features), device=features.device).roll(len(features) // 2)   # i -> i + N/2
    return F.cross_entropy(logits, labels)


class AnatomicalEmbedding(nn.Module):
    """Barycentric interpolation of learnable vertex embeddings on a fixed triangle mesh."""
    def __init__(self, body, dim, init_std=.02):
        super().__init__()
        import numpy as np
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(body))
        faces = np.asarray(mesh.triangles)
        if not len(faces):
            raise ValueError(f'No triangles in anatomy body {body}')
        # Stored in the checkpoint: the topology the embeddings were trained on.
        self.register_buffer('faces', torch.from_numpy(faces.astype(np.int64)))
        self.vertex_embeddings = nn.Parameter(torch.zeros(len(mesh.vertices), dim))
        nn.init.trunc_normal_(self.vertex_embeddings, std=init_std)

    def forward(self, face, barycentric):
        # face [B] int64, barycentric [B, 3] -> [B, dim]
        corners = self.vertex_embeddings[self.faces[face]]                     # [B, 3, dim]
        return (barycentric.to(corners.dtype).unsqueeze(-1) * corners).sum(1)


class FourierAnatomicalEmbedding(nn.Module):
    """Deterministic geometric encoding of the canonical position, then a small learned projection.

    p = l1 v_a + l2 v_b + l3 v_c on the canonical body (metres, centred on the body centroid);
    features = [p / 0.5, sin(2 pi p / w_k), cos(2 pi p / w_k)] for K wavelengths w_k log-spaced in
    [min_wavelength, max_wavelength] (3 + 6K values); MLP -> [B, dim]. Nearby skin points get similar
    encodings from the first step, unlike free per-vertex embeddings.
    """
    def __init__(self, body, dim, num_frequencies=8, min_wavelength=.01, max_wavelength=1., project=True):
        super().__init__()
        import numpy as np
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(body))
        vertices, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles)
        if not len(faces):
            raise ValueError(f'No triangles in anatomy body {body}')
        # num_frequencies = 0: naive encoding, the (scaled) 3-D position alone ("encoding": "xyz").
        if not 0 < min_wavelength <= max_wavelength or num_frequencies < 0:
            raise ValueError('Invalid Fourier anatomy settings')
        self.register_buffer('faces', torch.from_numpy(faces.astype(np.int64)))
        self.register_buffer('vertices', torch.from_numpy(vertices - vertices.mean(0)))
        wavelengths = (np.geomspace(max_wavelength, min_wavelength, num_frequencies) if num_frequencies
                       else np.ones(0)).astype(np.float32)
        self.register_buffer('angular', torch.from_numpy(2 * np.pi / wavelengths))
        self.num_features = 3 + 6 * num_frequencies
        # project=False ("mode": "query"): the raw features are returned, policy_net consumes them.
        self.projection = (nn.Sequential(nn.Linear(self.num_features, dim), nn.SiLU(), nn.Linear(dim, dim))
                           if project else nn.Identity())

    def position(self, face, barycentric):
        corners = self.vertices[self.faces[face]]                                  # [B, 3, 3]
        return (barycentric.float().unsqueeze(-1) * corners).sum(1)                # [B, 3]

    def forward(self, face, barycentric, noise=0.):
        p = self.position(face, barycentric)
        if noise:
            # Training-only position noise (metres, isotropic Gaussian): the token then tells the region,
            # not the exact point, so it cannot identify the episode.
            p = p + noise * torch.randn_like(p)
        phase = (p.unsqueeze(-1) * self.angular).flatten(1)                        # [B, 3K]
        features = torch.cat([p / .5, phase.sin(), phase.cos()], dim=1)
        return self.projection(features)


class AnatomicalPositionTarget(nn.Module):
    """Auxiliary head: pooled encoder features -> canonical probe position (body-centred, / scale)."""
    def __init__(self, body, dim, scale=.1):
        super().__init__()
        import numpy as np
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(body))
        vertices, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles)
        if not len(faces):
            raise ValueError(f'No triangles in anatomy body {body}')
        self.register_buffer('faces', torch.from_numpy(faces.astype(np.int64)))
        self.register_buffer('vertices', torch.from_numpy(vertices - vertices.mean(0)))
        self.scale = float(scale)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))

    def target(self, face, barycentric):
        corners = self.vertices[self.faces[face]]                                  # [B, 3, 3]
        return (barycentric.float().unsqueeze(-1) * corners).sum(1) / self.scale   # [B, 3]

    def forward(self, features, attachment):
        """(smooth L1 loss, mean position error in mm) for features [B, K, D]."""
        prediction = self.head(features.mean(1)).float()
        target = self.target(*attachment)
        error_mm = (prediction - target).norm(dim=1).mean().detach() * self.scale * 1000
        return F.smooth_l1_loss(prediction, target), error_mm


ANATOMY_KEYS = {'body', 'dropout', 'encoding', 'num_frequencies', 'min_wavelength', 'max_wavelength', 'noise', 'mode'}
# How the anatomical position enters the motion predictor ("mode"):
#   token (default): one extra input token [context; anatomy token; action token]
#   query:           inside the action token, policy_net([action 6D, encoding(p), known]) (fourier/xyz only)
#   adaln:           every predictor block modulated by the position, LN(x) (1 + scale(e)) + shift(e),
#                    zero-initialized (starts as the model without anatomy); motion branch only
ANATOMY_MODES = ('token', 'query', 'adaln')


class MotionWorldModel(nn.Module):
    def __init__(self, image_size=224, embed_dim=384, encoder_depth=12, predictor_depth=6, num_heads=6,
                 action_scale=(30., 30., 30., .3, .3, .3), projector_dim=1024, motion_weight=.1, mask=None,
                 anatomy=None, anatomy_target=None):
        super().__init__()
        mask = dict(DEFAULT_MASK, **(mask or {}))
        # The complete configuration is stored in checkpoints and suffices to rebuild the model.
        self.config = dict(image_size=image_size, embed_dim=embed_dim, encoder_depth=encoder_depth,
                           predictor_depth=predictor_depth, num_heads=num_heads, action_scale=list(action_scale),
                           projector_dim=projector_dim, motion_weight=motion_weight, mask=mask)
        # anatomy: None (original model) or {"body": canonical body mesh}; see the module docstring.
        if anatomy is not None:
            if set(anatomy) - ANATOMY_KEYS or 'body' not in anatomy:
                raise ValueError(f'model.anatomy needs "body" and accepts {sorted(ANATOMY_KEYS)}')
            encoding = anatomy.get('encoding', 'vertex')
            if encoding not in ('vertex', 'fourier', 'xyz'):
                raise ValueError('model.anatomy.encoding must be "vertex" (learned per-vertex), "fourier" or "xyz"')
            if not 0 <= anatomy.get('dropout', 0) < 1:
                raise ValueError('model.anatomy.dropout must be in [0, 1)')
            if anatomy.get('noise', 0) and encoding == 'vertex':
                raise ValueError('model.anatomy.noise (metres) needs "encoding": "fourier" or "xyz"')
            self.anatomy_noise = float(anatomy.get('noise', 0))
            self.anatomy_mode = anatomy.get('mode', 'token')
            if self.anatomy_mode not in ANATOMY_MODES:
                raise ValueError(f'model.anatomy.mode must be one of {ANATOMY_MODES}')
            if self.anatomy_mode == 'query' and encoding == 'vertex':
                raise ValueError('model.anatomy.mode "query" needs "encoding": "fourier" or "xyz"')
            self.config['anatomy'] = dict(anatomy)
            # Training only: with probability `dropout` per sample the position is removed (e = 0, the
            # token reduces to anatomy_type_token, "position unknown"), so the motion prediction cannot
            # rely on the position alone. Evaluation always uses the position.
            self.anatomy_dropout = float(anatomy.get('dropout', 0))
            if encoding in ('fourier', 'xyz'):
                self.anatomy = FourierAnatomicalEmbedding(
                    anatomy['body'], embed_dim,
                    num_frequencies=anatomy.get('num_frequencies', 8) if encoding == 'fourier' else 0,
                    min_wavelength=anatomy.get('min_wavelength', .01), max_wavelength=anatomy.get('max_wavelength', 1.),
                    project=self.anatomy_mode != 'query')
            else:
                self.anatomy = AnatomicalEmbedding(anatomy['body'], embed_dim)
            self.anatomy_type_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.anatomy_type_token, std=.02)
        else:
            self.anatomy = None
        norm = partial(nn.LayerNorm, eps=1e-6)
        # Context encoder f_theta, ViT-S/16 by default: 16x16 patches, linear patch embedding
        # (16x16 convolution with stride 16), fixed 2-D sine-cosine positional embedding,
        # 12 transformer blocks (6 heads, MLP ratio 4) and a final LayerNorm. This network is
        # the outcome of pre-training and is the backbone transferred to fine-tuning.
        self.encoder = VisionTransformer(img_size=[image_size], patch_size=16, in_chans=3, embed_dim=embed_dim,
                                         depth=encoder_depth, num_heads=num_heads, norm_layer=norm)
        # Predictor g_phi, a shallower transformer (6 blocks) operating on context tokens.
        # A single predictor serves both objectives; only the appended query tokens differ
        # (positional mask tokens for spatial modeling, one action token for motion modeling).
        self.predictor = VisionTransformerPredictor(num_patches=(image_size // 16)**2,
                            embed_dim=embed_dim, predictor_embed_dim=embed_dim,
                            depth=predictor_depth, num_heads=num_heads, norm_layer=norm)
        # Modules used by upstream forward_extra, initialized as in pretrain/main.py.
        # policy_net is the motion encoder A_psi of the paper, mapping the 6-DoF motion to z in R^384.
        # The action token is z + mask_token_extra, a learned token without positional embedding.
        # With anatomy mode "query" its input is [action 6D, position features, known flag].
        policy_inputs = 6 + (self.anatomy.num_features + 1 if getattr(self, 'anatomy_mode', None) == 'query' else 0)
        self.predictor.policy_net = nn.Sequential(nn.Linear(policy_inputs, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.predictor.mask_token_extra = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.predictor.mask_token_extra, std=.02)
        # Projector P, used by the motion objective only. The contrastive loss is computed in a
        # separate 1024-d space rather than directly on the 384-d encoder representation.
        self.projector = nn.Sequential(nn.Linear(embed_dim, projector_dim), nn.BatchNorm1d(projector_dim),
                                       nn.ReLU(inplace=True), nn.Linear(projector_dim, projector_dim, bias=False))
        # Target networks f'_theta' and P': identical initialization, never updated by gradients,
        # only by an exponential moving average (update_target). They provide slowly varying
        # regression/contrastive targets and prevent representational collapse.
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.mask_collator = MaskCollator(input_size=image_size, patch_size=16, **mask)
        # Motion normalization before policy_net: 30 mm and 0.3 rad (~17 deg) map to unit scale.
        # This is a local choice; the released code divides its pose vector by 200.
        self.register_buffer('action_scale', torch.tensor(action_scale))
        self.motion_weight = motion_weight
        # Auxiliary anatomical target (no anatomy input): the online encoder features of each image
        # regress the canonical position of its probe on the canonical body. Created last, so that all
        # other modules keep the initialization of the model without it for the same seed.
        if getattr(self, 'anatomy_mode', None) == 'adaln':
            # Per block: shift/scale before attention and before the MLP, from the anatomy embedding.
            self.anatomy_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, predictor_depth * 4 * embed_dim))
            nn.init.zeros_(self.anatomy_modulation[1].weight)
            nn.init.zeros_(self.anatomy_modulation[1].bias)
        self.anatomy_target = None
        if anatomy_target is not None:
            if set(anatomy_target) - {'body', 'weight', 'scale'} or 'body' not in anatomy_target:
                raise ValueError('model.anatomy_target must be {"body": mesh, "weight": w, "scale": metres}')
            self.config['anatomy_target'] = dict(anatomy_target)
            self.anatomy_target = AnatomicalPositionTarget(
                anatomy_target['body'], embed_dim, scale=anatomy_target.get('scale', .1))
            self.anatomy_target_weight = float(anatomy_target.get('weight', .1))

    @staticmethod
    def encoder_input(images):
        # [B, 1, H, W] in [0, 1] -> [B, 3, H, W] in [-1, 1]. The three channels are identical;
        # replication only matches the three-channel patch embedding of the ViT.
        return (images.expand(-1, 3, -1, -1) - .5) / .5

    def masks(self, batch, device):
        # Block sizes come from the collator counter, positions from the global (checkpointed) RNG.
        # Masks are patch indices, not binary images:
        #   enc  = [tensor B x K]        patches visible to the context encoder
        #   pred = 4 x [tensor B x M]    patches to be predicted, one tensor per target block
        # Block sizes are shared within the batch; block locations differ for every image.
        self.mask_collator._itr_counter.value = int(torch.randint(2**31 - 1, ())) - 1
        _, enc, pred = self.mask_collator([0] * batch)
        return [m.to(device) for m in enc], [m.to(device) for m in pred]

    @staticmethod
    def batch_attachment(batch, frame, device):
        """(face, barycentric) of the 'source' or 'target' frames of a MocapPairs batch, or None."""
        if f'{frame}_face' not in batch:
            return None
        return batch[f'{frame}_face'].to(device), batch[f'{frame}_barycentric'].to(device)

    def condition(self, attachment):
        """Anatomy token [B, 1, D] of the current frame, or None for the original model."""
        if self.anatomy is None:
            return None                       # an attachment given to a model without anatomy is ignored
        if attachment is None:
            raise ValueError('This world model is conditioned on anatomy: pass the attachment (face, barycentric) '
                             'of the current frame (MocapPairs(anatomy=True))')
        if self.training and getattr(self, 'anatomy_noise', 0):
            embedding = self.anatomy(*attachment, noise=self.anatomy_noise)      # [B, D] (query: [B, F])
        else:
            embedding = self.anatomy(*attachment)                               # [B, D]
        keep = torch.ones(len(embedding), 1, device=embedding.device, dtype=embedding.dtype)
        if self.training and self.anatomy_dropout:
            keep = (torch.rand(len(embedding), 1, device=embedding.device) >= self.anatomy_dropout).to(embedding.dtype)
            embedding = embedding * keep
        if getattr(self, 'anatomy_mode', 'token') == 'query':
            return torch.cat([embedding, keep], dim=1)                          # [B, F + 1]
        return (embedding + self.anatomy_type_token[0]).unsqueeze(1)

    def motion(self, z, visible, action, attachment):
        """Predictor output of the action token [B, 1, D] for normalized actions [B, 6]."""
        mode = getattr(self, 'anatomy_mode', 'token') if self.anatomy is not None else None
        if mode == 'query':
            action = torch.cat([action, self.condition(attachment).to(action.dtype)], dim=1)
            return self.predictor(z, visible, action, extra=True)
        if mode == 'adaln':
            e = self.condition(attachment)[:, 0]                                 # [B, D]
            m = self.anatomy_modulation(e).view(len(e), -1, 4, 1, e.shape[-1])  # [B, depth, 4, 1, D]
            modulation = [m[:, i].unbind(1) for i in range(m.shape[1])]
            return self.predictor(z, visible, action, extra=True, modulation=modulation)
        return self.predictor(z, visible, action, extra=True, condition=self.condition(attachment))

    def predict_latent(self, source, action, anatomy=None):
        """Predicted pooled target feature of the image at the relative pose (full context visible).

        anatomy: (face [B], barycentric [B, 3]) of the source frame; required only by anatomy models.
        """
        # Inference path (evaluation, diffusion decoder): no masking, the encoder observes all
        # 196 patches. The output [B, 384] estimates the pooled target feature of I_b.
        features = self.encoder(self.encoder_input(source))                        # [B, 196, 384]
        visible = torch.arange(features.shape[1], device=source.device).expand(len(source), -1)
        return self.motion(features, visible, action / self.action_scale, anatomy).mean(1)  # [B, 384]

    @torch.no_grad()
    def target_latent(self, images):
        # Target representation: EMA encoder on the complete image followed by per-token
        # LayerNorm, as in I-JEPA, which fixes the scale of the regression targets. [B, 196, 384].
        h = self.target_encoder(self.encoder_input(images))
        return F.layer_norm(h, (h.shape[-1],))

    def forward(self, source, target, action, reverse_action, anatomy=None, reverse_anatomy=None):
        # One training step. Both images of the pair share the same sampled masks.
        # anatomy / reverse_anatomy: attachments of a and b (anatomy models only). Each direction
        # sees only the attachment of its own starting frame.
        enc, pred = self.masks(len(source), source.device)
        spatial, motion_predicted, motion_target, auxiliary = [], [], [], []
        # Symmetric objective, as in the released code: a predicts b, and b predicts a.
        for image, other, act, current in [(source, target, action, anatomy), (target, source, reverse_action, reverse_anatomy)]:
            # Spatial world modeling (I-JEPA) within a single image.
            h = self.target_latent(image)                                          # [B, 196, 384]
            # Target features of the hidden patches; the four blocks are stacked along the batch: [4B, M, 384].
            h_mask = repeat_interleave_batch(apply_masks(h, pred), len(image), repeat=len(enc))
            # The online encoder observes the context patches only.
            z = self.encoder(self.encoder_input(image), enc)                       # [B, K, 384]
            # Predictor input: context tokens plus mask tokens carrying the target positions -> [4B, M, 384].
            spatial.append(F.smooth_l1_loss(self.predictor(z, enc, pred).float(), h_mask.float()))
            if self.anatomy_target is not None:
                # Auxiliary target: where the probe of *this* image is on the canonical body. It is a
                # label, not an input: nothing of it reaches the predictor.
                if current is None:
                    raise ValueError('model.anatomy_target needs the attachments (MocapPairs(anatomy=True))')
                auxiliary.append(self.anatomy_target(z, current))
            # Motion world modeling on the same context representation z, with an action token.
            # forward_extra: [context tokens; policy_net(act) + mask_token_extra] -> transformer ->
            # output of the action token only, [B, 1, 384] -> mean(1) -> [B, 384]. Through
            # self-attention the action token aggregates the context, which is where the model
            # learns how the ultrasound appearance changes under the given probe motion.
            motion_predicted.append(self.projector(self.motion(z, enc, act / self.action_scale, current).mean(1)))
            with torch.no_grad():
                # Target: target features of the *other* image averaged over all 196 patches and
                # mapped by the EMA projector -> [B, 1024].
                motion_target.append(self.target_projector(self.target_latent(other).mean(1)))
        spatial_loss = sum(spatial) / 2
        motion_loss = info_nce(torch.cat(motion_predicted + motion_target))
        # Motion weight 0.1, as in the paper and the released script (--extra_weight 0.1).
        loss = spatial_loss + self.motion_weight * motion_loss
        values = dict(spatial=spatial_loss, motion=motion_loss)
        if auxiliary:
            anatomy_loss = sum(a for a, _ in auxiliary) / len(auxiliary)
            loss = loss + self.anatomy_target_weight * anatomy_loss
            values.update(anatomy=anatomy_loss, anatomy_mm=sum(e for _, e in auxiliary) / len(auxiliary))
        return loss, dict(loss=loss, **values)

    @torch.no_grad()
    def update_target(self, momentum=.996):
        # EMA update target <- m * target + (1 - m) * online, applied after each optimizer step.
        # The momentum m increases from 0.996 to 1.0 during training (train.py), so the target
        # networks progressively freeze.
        for online, target in [(self.encoder, self.target_encoder), (self.projector, self.target_projector)]:
            for p, q in zip(online.parameters(), target.parameters()):
                q.lerp_(p, 1 - momentum)
