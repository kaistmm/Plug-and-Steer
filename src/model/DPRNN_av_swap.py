import torch.nn.functional as F
from torch import nn
import torch

import warnings

warnings.filterwarnings('ignore')
from src.model.visual_encoder.resnet18 import Visual_encoder
from src.model.visual_encoder.visual_module_gate import VisualModule


def cgld_logit(
    logits: torch.Tensor,
    K: int = 7,
    eps: float = 1e-6,
):
    """
    Confidence-Guided Local–Global Decoding (CGLD), logit-input version.

    Args:
        logits: [B, T] raw model outputs (logits, before sigmoid)
        K: CWTD window size (odd recommended)
        eps: numerical stability constant

    Returns:
        logits_refined: [B, T] refined logits
        probs_refined:  [B, T] refined probabilities (sigmoid of refined logits)
        states:         [B, T] hard labels in {0,1}
    """
    assert logits.dim() == 2, "logits must be [B, T]"
    B, T = logits.shape

    # confidence weight from logit magnitude: w(0)=0, w->1 as |logit| grows
    w = torch.tanh(logits.abs())  # [B, T] in [0, 1)

    # --- CWTD local refinement (weighted moving average via conv1d) ---
    ker = torch.ones(1, 1, K, device=logits.device, dtype=logits.dtype)
    num = F.conv1d((w * logits).unsqueeze(1), ker, padding=K // 2).squeeze(1)  # [B,T]
    den = F.conv1d(w.unsqueeze(1), ker, padding=K // 2).squeeze(1)             # [B,T]
    l_loc = num / (den + eps)                                                  # [B,T]

    # --- global evidence (confidence-weighted average over time) ---
    L = (w * l_loc).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True) + eps)  # [B,1]

    # --- confidence-guided local–global fusion ---
    l_hat = w * l_loc + (1.0 - w) * L  # [B,T]

    # outputs
    #p_hat = torch.sigmoid(l_hat)
    #s_hat = (l_hat > 0).to(logits.dtype)  # logit threshold 0 <=> prob threshold 0.5

    return l_hat


class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)

class Encoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1, bias=False)

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RNN(nn.Module):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2,
                 swap_layer=0, compress_dim=4):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_rnn.append(Dual_RNN_Block(out_channels, hidden_channels,
                                     rnn_type=rnn_type, norm=norm, dropout=dropout,
                                     bidirectional=bidirectional))

        self.conv2d = nn.Conv2d(
            out_channels, out_channels*num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
         # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                         nn.Sigmoid()
                                         )
        self.swap_layer = swap_layer
        
        self.compress_dim = compress_dim
        self.emb_dim=emb_dim=64
        ### reduce C (64) -> 4
        self.compress = nn.Conv2d(self.emb_dim, self.compress_dim, 1)
        # self.decompress = nn.Conv2d(self.compress_dim, self.emb_dim, 1)
        self.vis_module = VisualModule(self.compress_dim*self.K)
        

    def forward(self, x, vis_emb=None, swap_matrix=None, perturbation_cfg=None, output_gate=False):
        '''
           x: [B, N, L]

        '''
        # [B, N, L]
        input_shape = x.shape
        # print(f"1. input_shape {input_shape}")
        T = input_shape[-1]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # print(f"2. after conv. shape {x.shape}")
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # print(f"3. after Segmentation shape: {x.shape}")
        # [B, N*spks, K, S]
        gate_logit = None
        
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
            if self.swap_layer == i and swap_matrix is not None:
                B, C, K, S = x.shape
                # swap_matrix = swap_matrix.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,K, S)  # (T,F,F)
                # print(f"swap matrix shape: {swap_matrix.shape}")
                x = x.detach()
                
                if vis_emb is not None:
                    ### x and vis_emb should be in [B,T,F] shape
                    # print(vis_emb.shape, x.shape) #torch.Size([2, 50, 512]) torch.Size([2, 64, 250, 258])
                    # import pdb; pdb.set_trace()
                    # print("1", x.shape, gap) # [2, 64, 250, 258], 126
                    # x_merged = self._over_add(x, gap)
                    # print("2", x_merged.shape) # [2, 64, 31999]
                    """v1
                    x_c = x.permute(0,1,3,2).contiguous() # [2, 64, 258, 250]
                    x_c = self.compress(x_c) # [2, 64, 258, 8]
                    x_c = x_c.permute(0,2,1,3).contiguous() # [2, 258, 64, 8]
                    x_c = x_c.reshape(B, S, self.compress_dim*self.emb_dim) # [2, 258, 512]
                    # print(x_c.shape)
                    gate, gate_logit = self.vis_module(x_c, vis_emb, output_logit=True)
                    """
                    
                    # v2. reduce C
                    x_c = self.compress(x)
                    x_c = x_c.permute(0,3,1,2).contiguous().reshape(B,S,self.compress_dim*self.K) # [2, 258, 4, 250]-> [2,258,1000]
                    # print(f"3.{x_c.shape}")
                    gate, gate_logit = self.vis_module(x_c, vis_emb, output_logit=True)
                    gate = gate.squeeze(-1).unsqueeze(1).unsqueeze(1)
                    gate_logit = gate_logit.squeeze(-1).unsqueeze(1).unsqueeze(1)
                    
                    # print(x.shape, gate.shape, swap_matrix.shape)
                    # swap_res = torch.einsum('btf,tfo->bto', x, swap_matrix)
                    swap_res = torch.einsum('bnks,nf->bfks', x, swap_matrix)
                    # print(swap_res.shape)
                    if self.training:
                        #x = x + swap_res * gate_label.clone().detach().unsqueeze(-1)
                        #x = x + torch.einsum('bfks,bs->bfks', swap_res, gate.squeeze(-1))
                        x = x + swap_res * gate
                    else:
                        gate = torch.nn.functional.sigmoid(gate_logit.mean(dim=-1)).unsqueeze(-1)
                        gate_value = (gate.detach()>=0.5).float()
                        x = x + swap_res * gate_value
                
                
                # if swap_mask is not None:
                #     if swap_mask.shape[1] != T:
                #         m = swap_mask.float().unsqueeze(1)   # (B,1,T')
                #         m = F.interpolate(m, size=T, mode="nearest")
                #         swap_mask_T = m.squeeze(1)            # (B,T)
                #         swap_mask = self._Segmentation(swap_mask_T.unsqueeze(1), self.K)
                #     else:
                #         swap_mask = self._Segmentation(swap_mask.unsqueeze(1), self.K)
                #         swap_mask_T = swap_mask.float()
                #     print(swap_mask_T.shape)
                #     x = x + torch.einsum('bnks,nf,bks->bfks', x, swap_matrix, swap_mask_T)
                #     # x = x + torch.einsum('btf,tfo,bt->bto', x, swap_matrix, swap_mask_T)
                # else:
                #     x = x + torch.einsum('bnks,nf->bfks', x, swap_matrix)

                

        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B*self.num_spks,-1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x)*self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)
        # print(x.shape)
        return x, gate_logit

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type()).to(input.device)
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

class Dual_RNN_model(nn.Module):
    '''
       model of Dual Path RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            hidden_channels: The hidden size of RNN
            kernel_size: Encoder and Decoder Kernel size
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2,
                 ckpt_path=None, freeze=False, swap_layer=0, compress_dim=4,
                 ):
        super(Dual_RNN_model,self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size,out_channels=in_channels)
        self.separation = Dual_Path_RNN(in_channels, out_channels, hidden_channels,
                 rnn_type=rnn_type, norm=norm, dropout=dropout,
                 bidirectional=bidirectional, num_layers=num_layers, K=K, num_spks=num_spks,
                 swap_layer=swap_layer, compress_dim=compress_dim)
        self.decoder = Decoder(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=kernel_size//2, bias=False)
        self.num_spks = num_spks
        
        # below code is additional code
        self.ckpt_path = ckpt_path
        self.freeze = freeze

        self.W = nn.Parameter(torch.zeros(out_channels, out_channels), requires_grad=True)
        self.I = nn.Parameter(torch.eye(out_channels), requires_grad=False)
        
        if self.ckpt_path is not None:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            m, u = self.load_state_dict(checkpoint["model"], strict=False)
            if len(u) != 0:
                raise ValueError("Unexpected key in swap ckpt")
            if len(m) > 0:
                for k in m:
                    if k.startswith('separation.vis_module.') or k.startswith('separation.compress.'):
                        continue
                    else:
                        raise ValueError(f"{k} is missing from swap ckpt")
            print(f"Loaded DPRNN_swap model from {ckpt_path}")
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
                
        for param in self.separation.vis_module.parameters():
            param.requires_grad = True
        self.swap_layer = swap_layer
        
    
    def forward(self, x, vis_emb=None, perturbation_cfg=None, output_gate=False):
        '''
           x: [B, L]
        '''
        # [B, N, L]
        e = self.encoder(x)
        # [spks, B, N, L]
        s, gate = self.separation(e, vis_emb, self.W, perturbation_cfg, output_gate=output_gate)
        
        # [B, N, L] -> [B, L]
        out = [s[i]*e for i in range(self.num_spks)]
        audio = [self.decoder(out[i]).squeeze(1) for i in range(self.num_spks)]
        return audio, gate


class Dual_RNN_model_AV(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.audio_model = Dual_RNN_model(**kwargs)
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
    model = Dual_RNN_model_AV(**kwargs)
    return model
