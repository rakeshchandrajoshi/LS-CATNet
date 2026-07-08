"""LS-CATNet model implementation.

This implementation follows the working LS-CATNet code used for the manuscript:
- shallow EfficientNet-B0 backbone from timm
- four multi-scale patch-token branches using patch sizes 5, 7, 9 and 14
- separable self-attention transformer blocks
- coordinate-attention cross-scale bridges
- adaptive branch-wise softmax-gated fusion

The fusion used here is branch-wise softmax gating, not classical SE channel
recalibration. The manuscript should therefore describe it as adaptive
branch-wise gated fusion.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The LS-CATNet model requires timm. Install it with: pip install timm"
    ) from exc


EMBED_DIM = 32
MID_DIM = 64
BACKBONE_OUT_CHANNELS = 24
GRID_A, GRID_B, GRID_C, GRID_D = 11, 8, 6, 4


class SeparableSelfAttention(nn.Module):
    """Separable self-attention using depthwise Conv1D Q/K/V projections."""

    def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.dwq = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
        self.dwk = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
        self.dwv = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        batch_size, n_tokens, channels = x.shape
        xt = x.transpose(1, 2)  # [B, C, N]

        q = self.dwq(xt).transpose(1, 2)
        k = self.dwk(xt).transpose(1, 2)
        v = self.dwv(xt).transpose(1, 2)

        q = q.reshape(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, channels)
        return self.norm(x + self.proj(out))


class SepTransformerBlock(nn.Module):
    """Separable self-attention followed by residual MLP and layer normalization."""

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        mlp_hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ssa = SeparableSelfAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ssa(x)
        return self.norm(x + self.ffn(x))


class CoordinateAttention(nn.Module):
    """Coordinate attention over height and width directions.

    This implementation avoids version-dependent AdaptiveAvgPool2d(None, 1)
    calls by directly averaging over the corresponding spatial dimensions.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(4, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        _, _, height, width = x.shape
        h = x.mean(dim=3, keepdim=True)  # [B, C, H, 1]
        w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B, C, W, 1]
        hw = self.act(self.bn1(self.conv1(torch.cat([h, w], dim=2))))
        ah = torch.sigmoid(self.conv_h(hw[:, :, :height, :]))
        aw = torch.sigmoid(self.conv_w(hw[:, :, height:, :])).permute(0, 1, 3, 2)
        return x * ah * aw


class CCABridge(nn.Module):
    """Cross-scale coordinate-attention bridge.

    The bridge transforms source-branch tokens to a spatial map, adapts the
    resolution to the target branch, refines it using coordinate attention and
    returns a token sequence at the target resolution.
    """

    def __init__(self, src_hw: int, tgt_hw: int, channels: int = 32, mid: int = 64):
        super().__init__()
        self.src_hw = src_hw
        self.tgt_hw = tgt_hw
        self.conv_up = nn.Conv2d(channels, mid, 3, padding=1, bias=False)
        self.bn_up = nn.BatchNorm2d(mid)
        self.ca = CoordinateAttention(mid, reduction=4)
        self.conv_dn = nn.Conv2d(mid, channels, 3, padding=1, bias=False)
        self.bn_dn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        batch_size, n_tokens, channels = x.shape
        expected_tokens = self.src_hw * self.src_hw
        if n_tokens != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} source tokens, got {n_tokens}")

        x = x.reshape(batch_size, self.src_hw, self.src_hw, channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        if self.src_hw != self.tgt_hw:
            x = F.interpolate(x, size=(self.tgt_hw, self.tgt_hw), mode="bilinear", align_corners=False)

        x = self.act(self.bn_up(self.conv_up(x)))
        x = self.ca(x)
        x = self.act(self.bn_dn(self.conv_dn(x)))
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.reshape(batch_size, self.tgt_hw * self.tgt_hw, channels)


class PatchEmbeddingLite(nn.Module):
    """Convolutional patch embedding for CNN feature maps."""

    def __init__(
        self,
        patch_size: int,
        in_channels: int = BACKBONE_OUT_CHANNELS,
        inter_dim: int = 8,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            inter_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.proj = nn.Linear(inter_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        return self.norm(self.proj(x))


class LSCATNet(nn.Module):
    """LS-CATNet: Lightweight Cross-Scale Attention Transformer Network."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        tmp = timm.create_model("efficientnet_b0", pretrained=pretrained_backbone)
        self.backbone = nn.Sequential(tmp.conv_stem, tmp.bn1, tmp.blocks[0], tmp.blocks[1])
        del tmp

        # Freeze stem, bn1 and blocks[0]; train blocks[1].
        for idx, child in enumerate(self.backbone.children()):
            for param in child.parameters():
                param.requires_grad = idx == 3

        self.pea = PatchEmbeddingLite(5, BACKBONE_OUT_CHANNELS, 8, EMBED_DIM)
        self.peb = PatchEmbeddingLite(7, BACKBONE_OUT_CHANNELS, 8, EMBED_DIM)
        self.pec = PatchEmbeddingLite(9, BACKBONE_OUT_CHANNELS, 8, EMBED_DIM)
        self.ped = PatchEmbeddingLite(14, BACKBONE_OUT_CHANNELS, 8, EMBED_DIM)

        self.vita = self._make_vit_branch(dropout)
        self.vitb = self._make_vit_branch(dropout)
        self.vitc = self._make_vit_branch(dropout)
        self.vitd = self._make_vit_branch(dropout)

        # Eight bridge connections: A->B, A->C, B->A, B->C, C->A, C->B, D->A, D->B.
        self.caa2b = CCABridge(GRID_A, GRID_B, EMBED_DIM, MID_DIM)
        self.caa2c = CCABridge(GRID_A, GRID_C, EMBED_DIM, MID_DIM)
        self.cab2a = CCABridge(GRID_B, GRID_A, EMBED_DIM, MID_DIM)
        self.cab2c = CCABridge(GRID_B, GRID_C, EMBED_DIM, MID_DIM)
        self.cac2a = CCABridge(GRID_C, GRID_A, EMBED_DIM, MID_DIM)
        self.cac2b = CCABridge(GRID_C, GRID_B, EMBED_DIM, MID_DIM)
        self.cad2a = CCABridge(GRID_D, GRID_A, EMBED_DIM, MID_DIM)
        self.cad2b = CCABridge(GRID_D, GRID_B, EMBED_DIM, MID_DIM)

        self.fca = self._make_projection_head(dropout=0.0)
        self.fcb = self._make_projection_head(dropout=0.0)
        self.fcc = self._make_projection_head(dropout=0.0)
        self.fcd = self._make_projection_head(dropout=0.0)

        # Adaptive branch-wise gated fusion.
        # It generates four normalized branch weights from concatenated branch features.
        self.fusion_gate = nn.Sequential(nn.Linear(64 * 4, 4), nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _make_vit_branch(dropout: float) -> nn.ModuleList:
        return nn.ModuleList([SepTransformerBlock(EMBED_DIM, 4, EMBED_DIM, dropout) for _ in range(5)])

    @staticmethod
    def _make_projection_head(dropout: float = 0.0) -> nn.Sequential:
        layers: List[nn.Module] = [nn.Linear(EMBED_DIM, 64), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(64, 64), nn.GELU()])
        return nn.Sequential(*layers)

    @staticmethod
    def _run_vit(tokens: torch.Tensor, blocks: Iterable[nn.Module]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for block in blocks:
            tokens = block(tokens)
            outputs.append(tokens)
        return outputs

    def _forward_branches(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ta = self.pea(feat)  # [B, 121, 32]
        tb = self.peb(feat)  # [B,  64, 32]
        tc = self.pec(feat)  # [B,  36, 32]
        td = self.ped(feat)  # [B,  16, 32]

        va = self._run_vit(ta, self.vita)
        vb = self._run_vit(tb, self.vitb)
        vc = self._run_vit(tc, self.vitc)
        vd = self._run_vit(td, self.vitd)

        fuseda = va[4] + self.cab2a(vb[1]) + self.cac2a(vc[1]) + self.cad2a(vd[1])
        fusedb = vb[4] + self.caa2b(va[1]) + self.cac2b(vc[1]) + self.cad2b(vd[1])
        fusedc = vc[4] + self.caa2c(va[1]) + self.cab2c(vb[1])
        fusedd = vd[4]

        fa = self.fca(fuseda.mean(dim=1))
        fb = self.fcb(fusedb.mean(dim=1))
        fc = self.fcc(fusedc.mean(dim=1))
        fd = self.fcd(fusedd.mean(dim=1))
        return fa, fb, fc, fd

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        fa, fb, fc, fd = self._forward_branches(feat)
        stacked = torch.stack([fa, fb, fc, fd], dim=1)  # [B, 4, 64]
        weights = self.fusion_gate(torch.cat([fa, fb, fc, fd], dim=1)).unsqueeze(-1)  # [B, 4, 1]
        fused = (stacked * weights).sum(dim=1)  # [B, 64]
        return fused, weights.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused, _ = self.forward_features(x)
        return self.classifier(fused)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total, trainable and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def module_wise_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        name: sum(p.numel() for p in module.parameters() if p.requires_grad)
        for name, module in model.named_children()
    }
