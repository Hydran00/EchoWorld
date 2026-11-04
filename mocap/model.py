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


class MotionWorldModel(nn.Module):
    def __init__(self, image_size=224, embed_dim=384, encoder_depth=12, predictor_depth=6, num_heads=6,
                 action_scale=(30., 30., 30., .3, .3, .3), projector_dim=1024, motion_weight=.1, mask=None):
        super().__init__()
        mask = dict(DEFAULT_MASK, **(mask or {}))
        # The complete configuration is stored in checkpoints and suffices to rebuild the model.
        self.config = dict(image_size=image_size, embed_dim=embed_dim, encoder_depth=encoder_depth,
                           predictor_depth=predictor_depth, num_heads=num_heads, action_scale=list(action_scale),
                           projector_dim=projector_dim, motion_weight=motion_weight, mask=mask)
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
        self.predictor.policy_net = nn.Sequential(nn.Linear(6, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
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

    def predict_latent(self, source, action):
        """Predicted pooled target feature of the image at the relative pose (full context visible)."""
        # Inference path (evaluation, diffusion decoder): no masking, the encoder observes all
        # 196 patches. The output [B, 384] estimates the pooled target feature of I_b.
        features = self.encoder(self.encoder_input(source))                        # [B, 196, 384]
        visible = torch.arange(features.shape[1], device=source.device).expand(len(source), -1)
        return self.predictor(features, visible, action / self.action_scale, extra=True).mean(1)  # [B, 384]

    @torch.no_grad()
    def target_latent(self, images):
        # Target representation: EMA encoder on the complete image followed by per-token
        # LayerNorm, as in I-JEPA, which fixes the scale of the regression targets. [B, 196, 384].
        h = self.target_encoder(self.encoder_input(images))
        return F.layer_norm(h, (h.shape[-1],))

    def forward(self, source, target, action, reverse_action):
        # One training step. Both images of the pair share the same sampled masks.
        enc, pred = self.masks(len(source), source.device)
        spatial, motion_predicted, motion_target = [], [], []
        # Symmetric objective, as in the released code: a predicts b, and b predicts a.
        for image, other, act in [(source, target, action), (target, source, reverse_action)]:
            # Spatial world modeling (I-JEPA) within a single image.
            h = self.target_latent(image)                                          # [B, 196, 384]
            # Target features of the hidden patches; the four blocks are stacked along the batch: [4B, M, 384].
            h_mask = repeat_interleave_batch(apply_masks(h, pred), len(image), repeat=len(enc))
            # The online encoder observes the context patches only.
            z = self.encoder(self.encoder_input(image), enc)                       # [B, K, 384]
            # Predictor input: context tokens plus mask tokens carrying the target positions -> [4B, M, 384].
            spatial.append(F.smooth_l1_loss(self.predictor(z, enc, pred).float(), h_mask.float()))
            # Motion world modeling on the same context representation z, with an action token.
            # forward_extra: [context tokens; policy_net(act) + mask_token_extra] -> transformer ->
            # output of the action token only, [B, 1, 384] -> mean(1) -> [B, 384]. Through
            # self-attention the action token aggregates the context, which is where the model
            # learns how the ultrasound appearance changes under the given probe motion.
            motion_predicted.append(self.projector(self.predictor(z, enc, act / self.action_scale, extra=True).mean(1)))
            with torch.no_grad():
                # Target: target features of the *other* image averaged over all 196 patches and
                # mapped by the EMA projector -> [B, 1024].
                motion_target.append(self.target_projector(self.target_latent(other).mean(1)))
        spatial_loss = sum(spatial) / 2
        motion_loss = info_nce(torch.cat(motion_predicted + motion_target))
        # Motion weight 0.1, as in the paper and the released script (--extra_weight 0.1).
        loss = spatial_loss + self.motion_weight * motion_loss
        return loss, dict(loss=loss, spatial=spatial_loss, motion=motion_loss)

    @torch.no_grad()
    def update_target(self, momentum=.996):
        # EMA update target <- m * target + (1 - m) * online, applied after each optimizer step.
        # The momentum m increases from 0.996 to 1.0 during training (train.py), so the target
        # networks progressively freeze.
        for online, target in [(self.encoder, self.target_encoder), (self.projector, self.target_projector)]:
            for p, q in zip(online.parameters(), target.parameters()):
                q.lerp_(p, 1 - momentum)
