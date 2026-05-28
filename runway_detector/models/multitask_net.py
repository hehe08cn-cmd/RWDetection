"""HRNet-Offset corner detection model.

Loads frozen HRNet backbone + corner head from checkpoint.
Edge/centerline detection is handled by the standalone ScanlineEdgeNet.
"""

import torch
import torch.nn as nn

from .hrnet_detection import TimmHRNetBackbone, HRNetOutputHead


class MultiTaskNet(nn.Module):
    """Loads HRNet-Offset checkpoint (backbone + corner head, frozen)."""

    def __init__(self, checkpoint_path: str, crop_size: int = 256,
                 device: str = "cuda"):
        super().__init__()
        self.crop_size = crop_size

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_cfg = ckpt.get("config", {})
        in_channels = ckpt_cfg.get("model", {}).get("in_channels", 3)
        sd = ckpt["model_state_dict"]

        self.backbone = TimmHRNetBackbone(
            model_name="hrnet_w18_small_v2.ms_in1k",
            pretrained=False,
            in_chans=in_channels,
        )
        self.corner_head = HRNetOutputHead(1920, 4)

        # Load checkpoint, remapping old prefix "head." → "corner_head."
        model_sd = self.state_dict()
        loaded = 0
        for k, v in sd.items():
            target = k
            if k.startswith("head.") and "corner_head." + k[5:] in model_sd:
                target = "corner_head." + k[5:]
            if target in model_sd and model_sd[target].shape == v.shape:
                model_sd[target] = v.clone()
                loaded += 1
        missing = [k for k in sd if k not in model_sd]
        if missing:
            print(f"  Checkpoint keys not in model (ok): {len(missing)} keys")
        print(f"  Loaded {loaded}/{len(sd)} keys from checkpoint")
        self.load_state_dict(model_sd)

        # Freeze backbone + corner head
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.corner_head.parameters():
            p.requires_grad = False

    def forward(self, image: torch.Tensor) -> dict:
        feats = self.backbone(image)  # (B, 1920, H/4, W/4)
        corner_out = self.corner_head(feats)
        return {
            "heatmaps": corner_out["heatmaps"],
            "coords": corner_out["coords"],
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.corner_head.eval()
        return self
