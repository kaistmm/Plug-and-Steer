import torch.nn.functional as F
from torch import nn
import torch
from typing import Optional, Tuple
from torchaudio.models import ConvTasNet
from src.model.visual_encoder.resnet18 import Visual_encoder
from src.model.visual_encoder.visual_module_residual import VisualModule

class ConvBlock(torch.nn.Module):
    """1D Convolutional block.

    Args:
        io_channels (int): The number of input/output channels, <B, Sc>
        hidden_channels (int): The number of channels in the internal layers, <H>.
        kernel_size (int): The convolution kernel size of the middle layer, <P>.
        padding (int): Padding value of the convolution in the middle layer.
        dilation (int, optional): Dilation value of the convolution in the middle layer.
        no_redisual (bool, optional): Disable residual block/output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

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

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
        )

        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)
        )
        self.skip_out = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=io_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out


class MaskGenerator(torch.nn.Module):
    """TCN (Temporal Convolution Network) Separation Module

    Generates masks for separation.

    Args:
        input_dim (int): Input feature dimension, <N>.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks, <P>.
        num_featrs (int): Input/output feature dimenstion of conv blocks, <B, Sc>.
        num_hidden (int): Intermediate feature dimention of conv blocks, <H>
        num_layers (int): The number of conv blocks in one stack, <X>.
        num_stacks (int): The number of conv block stacks, <R>.
        msk_activate (str): The activation function of the mask output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
        swap_layer: int=0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources

        self.input_norm = torch.nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=1e-8)
        self.input_conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=num_feats, kernel_size=1)

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2**l
                self.conv_layers.append(
                    ConvBlock(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        # The last ConvBlock does not need residual
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                    )
                )
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats,
            out_channels=input_dim * num_sources,
            kernel_size=1,
        )
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")
        
        self.swap_layer = swap_layer
        self.visual_module = VisualModule(num_feats)
        
    def forward(self, 
                input: torch.Tensor, 
                vis_emb=None,
                swap_matrix: torch.Tensor=None, 
                perturbation_cfg = None,
                output_gate=False
                ) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            Tensor: shape [batch, num_sources, features, frames]
        """
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        gate_logit = None
        for ii, layer in enumerate(self.conv_layers):
            residual, skip = layer(feats)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip

            if self.swap_layer==ii:
                # [B, 128, 4001] # _, F, T = output.shape
                if vis_emb is not None:
                    output_tr = output.transpose(1,2)
                    output_tr = output_tr + self.visual_module(output_tr, vis_emb)
                    output = output_tr.transpose(1,2)
                
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output.view(batch_size, self.num_sources, self.input_dim, -1), gate_logit


class ConvTasNet(torch.nn.Module):
    """Conv-TasNet architecture introduced in
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    :cite:`Luo_2019`.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.

    See Also:
        * :class:`torchaudio.pipelines.SourceSeparationBundle`: Source separation pipeline with pre-trained models.

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).
    """

    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        msk_activate: str = "sigmoid",
        ckpt_path: str = None, 
        freeze: bool = False, 
        swap_layer: int=0,
    ):
        super().__init__()

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=enc_num_feats,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.mask_generator = MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
            swap_layer=swap_layer,
        )
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        
        # below code is additional code
        self.ckpt_path = ckpt_path
        self.freeze = freeze

        if self.ckpt_path is not None:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            self.load_state_dict(checkpoint["model"], strict=False)
            print(f"Loaded ConvTasnet model from {ckpt_path}")

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
        
        if swap_layer is not None: 
            self.swap_layer = swap_layer
        swap_dim = msk_num_feats

        for param in self.mask_generator.visual_module.parameters():
            param.requires_grad = True
        
        

    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings

    def forward(self, input: torch.Tensor, vis_emb=None, perturbation_cfg=None, output_gate=False) -> torch.Tensor:
        """Perform source separation. Generate audio source waveforms.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, channel==1, frames]

        Returns:
            Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]
        """
        input = input.unsqueeze(1)

        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}")

        # B: batch size
        # L: input frame length
        # L': padded input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)  # B, F, M
        
        mask, gate = self.mask_generator(feats, vis_emb, None, perturbation_cfg, output_gate=output_gate)
        masked = mask * feats.unsqueeze(1)
        masked = masked.view(batch_size * self.num_sources, self.enc_num_feats, -1)  # B*S, F, M
        decoded = self.decoder(masked)  # B*S, 1, L'
        output = decoded.view(batch_size, self.num_sources, num_padded_frames)  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        out = [output[:,i,:].squeeze(1) for i in range(self.num_sources)]
        return out, gate


def conv_tasnet_base(num_sources: int = 2) -> ConvTasNet:
    r"""Builds non-causal version of :class:`~torchaudio.models.ConvTasNet`.

    The parameter settings follow the ones with the highest Si-SNR metirc score in the paper,
    except the mask activation function is changed from "sigmoid" to "relu" for performance improvement.

    Args:
        num_sources (int, optional): Number of sources in the output.
            (Default: 2)
    Returns:
        ConvTasNet:
            ConvTasNet model.
    """
    return ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=16,
        enc_num_feats=512,
        msk_kernel_size=3,
        msk_num_feats=128,
        msk_num_hidden_feats=512,
        msk_num_layers=8,
        msk_num_stacks=3,
        msk_activate="relu",
    )


class ConvTasNet_AV(torch.nn.Module):
    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        msk_activate: str = "sigmoid",
        ckpt_path: str = None, 
        freeze: bool = False, 
        swap_layer: int=0,
    ):
        super().__init__()
        self.audio_model = ConvTasNet(
            num_sources=num_sources,
            enc_kernel_size=enc_kernel_size,
            enc_num_feats=enc_num_feats,
            msk_kernel_size=msk_kernel_size,
            msk_num_feats=msk_num_feats,
            msk_num_hidden_feats=msk_num_hidden_feats,
            msk_num_layers=msk_num_layers,
            msk_num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
            ckpt_path=ckpt_path, 
            freeze=freeze, 
            swap_layer=swap_layer,
        )
        self.visual_model = Visual_encoder()

    def forward(self, audio_input, visual_input=None, perturbation_cfg=None, output_gate=False):
        if not visual_input is None:
            visual_input = visual_input.float()
            visual_emb = self.visual_model(visual_input).transpose(1,2)   # (B, T', 512)
        else:
            visual_emb = None
        audio_output, gate = self.audio_model(audio_input, vis_emb=visual_emb, perturbation_cfg=perturbation_cfg, output_gate=output_gate)

        if output_gate:
            return audio_output, gate
        else:
            return audio_output


def Main_Model(**kwargs):
    model = ConvTasNet_AV(**kwargs)
    return model
