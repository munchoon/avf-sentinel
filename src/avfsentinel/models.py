"""Two-stream fusion model: paired arm photos + Mel-spectrogram audio."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CLASSES


class FocalLoss(nn.Module):
    """Multi-label focal BCE.

    ``pos_weight`` and ``alpha`` up-weight the positive (dangerous) class, which
    is deliberate: a missed thrombosis has a much higher cost than a false alarm.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.65, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(len(CLASSES)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------
def _convnext(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Identity()
    return model, 768


def _resnet(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model, 512


def _efficientnet(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Identity()
    return model, 1280


def _vit(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Identity()
    return model, 768


def _biomedclip(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Optional BiomedCLIP visual encoder (requires ``open_clip_torch``).

    Install with: pip install open_clip_torch
    The default demo intentionally uses ConvNeXt-Tiny so it runs offline after
    the torchvision weights are cached.
    """
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "BiomedCLIP requires open_clip_torch. Install it with:\n"
            "  pip install open_clip_torch\n"
            "then rerun with --arm-backbone biomedclip."
        ) from exc
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    visual = model.visual
    return visual, int(visual.output_dim)


_ARM_BACKBONES = {
    "convnext_tiny": _convnext,
    "resnet18": _resnet,
    "efficientnet_b0": _efficientnet,
    "vit_b_16": _vit,
    "biomedclip": _biomedclip,
}


def build_arm_backbone(name: str = "convnext_tiny", pretrained: bool = True) -> tuple[nn.Module, int]:
    if name not in _ARM_BACKBONES:
        raise ValueError(f"Unknown arm backbone {name!r}. Choose from {sorted(_ARM_BACKBONES)}")
    return _ARM_BACKBONES[name](name, pretrained)


def build_audio_backbone(name: str = "resnet18", pretrained: bool = True, in_channels: int = 2) -> tuple[nn.Module, int]:
    if name == "resnet18":
        model, dim = _resnet(name, pretrained)
        old = model.conv1
        new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        model.conv1 = new
        return model, dim
    if name == "efficientnet_b0":
        model, dim = _efficientnet(name, pretrained)
        old = model.features[0][0]
        new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        model.features[0][0] = new
        return model, dim
    if name == "convnext_tiny":
        model, dim = _convnext(name, pretrained)
        old = model.features[0][0]
        new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.features[0][0] = new
        return model, dim
    raise ValueError(f"Unknown audio backbone {name!r}")


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class ArmPairEncoder(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True, out_dim: int = 512) -> None:
        super().__init__()
        self.encoder, feat_dim = build_arm_backbone(backbone, pretrained)
        self.backbone_name = backbone
        self.feat_dim = feat_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(4 * feat_dim),
            nn.Linear(4 * feat_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

    def forward(self, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        f_d = self.encoder(down)
        f_u = self.encoder(up)
        z = torch.cat([f_d, f_u, f_u - f_d, f_d * f_u], dim=1)
        return self.proj(z)


class AudioEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, out_dim: int = 256) -> None:
        super().__init__()
        self.encoder, feat_dim = build_audio_backbone(backbone, pretrained, in_channels=2)
        self.backbone_name = backbone
        self.feat_dim = feat_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.acoustic = nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

    def forward(self, spec: torch.Tensor, acoustic: torch.Tensor) -> torch.Tensor:
        f = self.encoder(spec)
        z = self.proj(f)
        a = self.acoustic(acoustic)
        return torch.cat([z, a], dim=1)


class FusionModel(nn.Module):
    def __init__(
        self,
        arm_backbone: str = "convnext_tiny",
        audio_backbone: str = "resnet18",
        pretrained: bool = True,
        arm_dim: int = 512,
        audio_dim: int = 256,
        hidden: int = 384,
        dropout: float = 0.25,
        n_classes: int = len(CLASSES),
        modality_dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.arm = ArmPairEncoder(arm_backbone, pretrained, arm_dim)
        self.audio = AudioEncoder(audio_backbone, pretrained, audio_dim)
        self.pair_stats = nn.Sequential(
            nn.Linear(12, 48),
            nn.GELU(),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        self.arm_norm = nn.LayerNorm(arm_dim)
        self.audio_norm = nn.LayerNorm(audio_dim + 32)
        fused_dim = arm_dim + audio_dim + 32 + 32 + 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, n_classes),
        )
        self.modality_dropout = modality_dropout
        self.config = {
            "arm_backbone": arm_backbone,
            "audio_backbone": audio_backbone,
            "pretrained": pretrained,
            "arm_dim": arm_dim,
            "audio_dim": audio_dim,
            "hidden": hidden,
            "dropout": dropout,
            "n_classes": n_classes,
        }

    def set_encoders_trainable(self, trainable: bool) -> None:
        for module in (self.arm.encoder, self.audio.encoder):
            for p in module.parameters():
                p.requires_grad = trainable

    def forward(
        self,
        image_down: torch.Tensor,
        image_up: torch.Tensor,
        spectrogram: torch.Tensor | None,
        pair_features: torch.Tensor,
        acoustic_features: torch.Tensor | None,
        has_audio: bool = True,
    ) -> torch.Tensor:
        arm_emb = self.arm_norm(self.arm(image_down, image_up))
        pair_emb = self.pair_stats(pair_features)

        if spectrogram is None or acoustic_features is None:
            has_audio = False
        if has_audio:
            audio_emb = self.audio_norm(self.audio(spectrogram, acoustic_features))
        else:
            audio_emb = torch.zeros(
                image_down.shape[0], self.config["audio_dim"] + 32, device=image_down.device
            )

        arm_mask = torch.ones(image_down.shape[0], 1, device=image_down.device)
        audio_mask = torch.full_like(arm_mask, float(has_audio))

        if self.training and self.modality_dropout > 0:
            drop_audio = torch.rand(image_down.shape[0], 1, device=image_down.device) < self.modality_dropout
            drop_arm = torch.rand(image_down.shape[0], 1, device=image_down.device) < (self.modality_dropout * 0.5)
            audio_emb = audio_emb * (~drop_audio).float()
            audio_mask = audio_mask * (~drop_audio).float()
            arm_emb = arm_emb * (~drop_arm).float()
            pair_emb = pair_emb * (~drop_arm).float()
            arm_mask = arm_mask * (~drop_arm).float()

        z = torch.cat([arm_emb, audio_emb, pair_emb, arm_mask, audio_mask], dim=1)
        return self.fusion(z)


def build_model(
    arm_backbone: str = "convnext_tiny",
    audio_backbone: str = "resnet18",
    pretrained: bool = True,
    **kwargs: Any,
) -> FusionModel:
    return FusionModel(arm_backbone=arm_backbone, audio_backbone=audio_backbone, pretrained=pretrained, **kwargs)
