import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        no_residual: bool = False,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(io_channels, hidden_channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels, eps=1e-8),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,  # depthwise
            ),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels, eps=1e-8),
        )

        self.res_out = None if no_residual else nn.Conv1d(hidden_channels, io_channels, 1)
        self.skip_out = nn.Conv1d(hidden_channels, io_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feat = self.conv_layers(x)
        residual = None if self.res_out is None else self.res_out(feat)
        skip = self.skip_out(feat)
        return residual, skip


class VisualModule(nn.Module):
    """
    Lightweight modified ConvTasNet mask generator for
    audio-visual fusion and refinement.

    Input:
        x   : audio feature      [B, T, F]
        x_v : lip feature        [B, T, F]  (already interpolated)

    Output:
        corr_logit : [B, T, 1]
        residual   : [B, T, F]
    """

    def __init__(
        self,
        audio_feat_dim: int = 512,       # dim for residual output
        visual_feat_dim: int = 512,       # dim for residual output
        num_feats: int = 128,
        num_hidden: int = 256,
        num_layers: int = 4,
        num_stacks: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.feat_dim = audio_feat_dim
        in_dim = audio_feat_dim + visual_feat_dim

        self.input_norm = nn.GroupNorm(1, in_dim, eps=1e-8)
        #self.input_norm = nn.LayerNorm(in_dim, eps=1e-8)
        self.input_conv = nn.Conv1d(in_dim, num_feats, 1)

        # TCN trunk
        self.tcn_blocks = nn.ModuleList()
        for s in range(num_stacks):
            for l in range(num_layers):
                dilation = 2 ** l
                padding = dilation
                is_last = (s == num_stacks - 1) and (l == num_layers - 1)

                self.tcn_blocks.append(
                    ConvBlock(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        no_residual=is_last,
                    )
                )

        self.output_prelu = nn.PReLU()

        self.res_head = nn.Conv1d(num_feats, self.feat_dim, 1)

    def forward(self, x, vis_emb, output_logit=False):
        """
        x, x_v: [B, T, F]
        """
        # concatenate AV features
        T = x.shape[1]
        if vis_emb.shape[1] != T:
            vis_emb = vis_emb.transpose(1,2)
            vis_emb = F.interpolate(vis_emb, size=T, mode="linear", align_corners=False)
            vis_emb = vis_emb.transpose(1,2)

        z = torch.cat([x, vis_emb], dim=-1)  # [B, T, 2F]

        z = z.transpose(1, 2)           # [B, 2F, T]
        feats = self.input_norm(z)
        #feats = feats.transpose(1, 2)           # [B, 2F, T]
        feats = self.input_conv(feats)

        skip_sum = 0.0
        for block in self.tcn_blocks:
            residual, skip = block(feats)
            if residual is not None:
                feats = feats + residual
            skip_sum = skip_sum + skip

        h = self.output_prelu(skip_sum)

        residual = self.res_head(h).transpose(1, 2)     # [B, T, F]
            

        return residual
