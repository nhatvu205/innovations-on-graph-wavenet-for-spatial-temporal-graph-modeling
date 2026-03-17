"""
Graph WaveNet - Gated TCN Module
==================================
WaveNet-style Temporal Convolutional Network với gating mechanism.

Specs (theo pipeline & paper):
  - Kernel size        : 2
  - Blocks             : 4
  - Layers per block   : 2
  - Dilation sequence  : (1, 2) x 4  = [1,2, 1,2, 1,2, 1,2]  (reset mỗi block)
  - Receptive field    : 1 + (1+2) x 4 = 13 time steps  (>= 12 required)
  - Gating             : tanh(TCN-a) * sigmoid(TCN-b)
  - residual_channels  : 32
  - skip_channels      : 256
  - Input left-pad     : 1 (để giữ temporal alignment)

Kiến trúc mỗi ST-layer:
    Input (B, C, T, N)
        │
        ├── TCN-a  (dilated conv) → tanh    ┐
        │                                    ├─ element-wise * → gated output
        └── TCN-b  (dilated conv) → sigmoid ┘
                │
              GCN (xử lý ở module khác)
                │
              + residual (skip connection)
                │
             Output (B, C, T', N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Dilated Causal Conv2d 
# =============================================================================

class DilatedCausalConv(nn.Module):
    """
    2D dilated convolution theo chiều thời gian (causal).

    Input shape : (B, C_in, T, N)
    Output shape: (B, C_out, T - dilation*(kernel-1), N)

    Causal = chỉ nhìn về quá khứ, không nhìn tương lai.
    Dilation mở rộng receptive field mà không tăng params.

    Args:
        c_in     : channels đầu vào
        c_out    : channels đầu ra
        kernel   : kernel size (mặc định 2 theo paper)
        dilation : dilation factor (1 hoặc 2 theo pipeline)
    """

    def __init__(self, c_in: int, c_out: int, kernel: int = 2, dilation: int = 1):
        super().__init__()
        # Conv2d: kernel (dilation*(kernel-1)+1, 1) trên chiều (T, N)
        self.conv = nn.Conv2d(
            in_channels  = c_in,
            out_channels = c_out,
            kernel_size  = (kernel, 1),
            dilation     = (dilation, 1),
            padding      = 0              # padding thủ công để causal
        )
        self.dilation = dilation
        self.kernel   = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C_in, T, N)
        Returns:
            out : (B, C_out, T - dilation*(kernel-1), N)
        """
        # Causal padding: thêm (dilation*(kernel-1)) zeros phía trái chiều T
        pad = self.dilation * (self.kernel - 1)
        x = F.pad(x, (0, 0, pad, 0))   # pad (left=pad, right=0) trên dim T
        return self.conv(x)


# =============================================================================
# Gated TCN Layer (một layer đơn lẻ)
# =============================================================================

class GatedTCNLayer(nn.Module):
    """
    Một layer Gated TCN gồm:
      - Hai nhánh dilated conv song song: TCN-a và TCN-b
      - Gating: output = tanh(TCN-a(x)) * sigmoid(TCN-b(x))
      - Residual connection: cộng input vào output (nếu cùng channel)

    Args:
        c_in       : channels đầu vào
        c_out      : channels đầu ra (residual_channels = 32)
        kernel     : kernel size (mặc định 2)
        dilation   : dilation factor
    """

    def __init__(self, c_in: int, c_out: int, kernel: int = 2, dilation: int = 1):
        super().__init__()

        # Hai nhánh conv song song (mỗi nhánh output c_out channels)
        self.tcn_a = DilatedCausalConv(c_in, c_out, kernel, dilation)   # → tanh
        self.tcn_b = DilatedCausalConv(c_in, c_out, kernel, dilation)   # → sigmoid

        # Residual: nếu c_in != c_out thì cần 1x1 conv để match shape
        self.residual_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) \
                             if c_in != c_out else nn.Identity()

        # Skip connection: project lên skip_channels (256)
        # (sẽ được dùng ở tầng assembly, để ở đây cho đầy đủ)
        self.skip_conv = nn.Conv2d(c_out, 256, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, C_in, T, N)

        Returns:
            residual : (B, C_out, T', N)   — tiếp tục đi vào layer kế tiếp
            skip     : (B, 256,  T', N)   — đi thẳng ra output head
        """
        # Tính receptive field bị thu hẹp sau conv
        gate  = torch.tanh(self.tcn_a(x))       # (B, C_out, T', N)
        gate *= torch.sigmoid(self.tcn_b(x))     # element-wise gating

        # Cắt residual cho khớp chiều T'
        T_out = gate.shape[2]
        x_res = x[:, :, -T_out:, :]             # lấy T' timestep cuối
        residual = gate + self.residual_conv(x_res)

        # Skip projection
        skip = self.skip_conv(gate)              # (B, 256, T', N)

        return residual, skip


# =============================================================================
# Gated TCN Block (nhiều layers với dilation tăng dần, reset mỗi block)
# =============================================================================

class GatedTCNBlock(nn.Module):
    """
    Một block gồm nhiều GatedTCNLayer với dilation sequence tăng dần.

    Theo pipeline:
        layers_per_block = 2
        dilations = [1, 2]    ← reset về 1 ở block tiếp theo

    Args:
        c_in             : channels vào block đầu tiên
        c_residual       : residual channels (32)
        kernel           : kernel size (2)
        dilations        : list dilation cho từng layer, vd [1, 2]
    """

    def __init__(
        self,
        c_in        : int,
        c_residual  : int = 32,
        kernel      : int = 2,
        dilations   : list = None
    ):
        super().__init__()
        dilations = dilations or [1, 2]

        layers = []
        in_ch  = c_in
        for d in dilations:
            layers.append(GatedTCNLayer(in_ch, c_residual, kernel, dilation=d))
            in_ch = c_residual      # các layer sau dùng c_residual làm input

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, C_in, T, N)

        Returns:
            x    : (B, C_residual, T'', N)   — output cuối block
            skips : list of skip tensors từ mỗi layer
        """
        skips = []
        for layer in self.layers:
            x, skip = layer(x)
            skips.append(skip)
        return x, skips


# =============================================================================
# Full Gated TCN Stack (4 blocks × 2 layers = 8 ST-layers)
# =============================================================================

class GatedTCNStack(nn.Module):
    """
    Stack đầy đủ của Gated TCN theo spec Graph WaveNet:
        - 4 blocks, mỗi block 2 layers
        - Dilation per block: [1, 2]  (reset mỗi block)
        - Tổng 8 ST-layers
        - Receptive field: 1 + sum([1,2]*4) = 13 >= 12 ✓

    Input được left-pad 1 bước trước khi vào stack
    (để giữ alignment sau khi qua các conv).

    Args:
        c_in       : input channels (in_dim = 2 sau start conv → 32)
        c_residual : residual channels (32)
        kernel     : kernel size (2)
        num_blocks : số block (4)
        layers_per_block : số layer mỗi block (2)
    """

    def __init__(
        self,
        c_in             : int   = 32,
        c_residual       : int   = 32,
        kernel           : int   = 2,
        num_blocks       : int   = 4,
        layers_per_block : int   = 2
    ):
        super().__init__()

        # Dilation sequence trong mỗi block: [1, 2, 4, 8, ...]
        # Theo pipeline chỉ dùng 2 layers nên: [1, 2]
        dilations = [2**i for i in range(layers_per_block)]   # [1, 2]

        self.blocks = nn.ModuleList([
            GatedTCNBlock(
                c_in       = c_in if i == 0 else c_residual,
                c_residual = c_residual,
                kernel     = kernel,
                dilations  = dilations
            )
            for i in range(num_blocks)
        ])

        self.num_blocks       = num_blocks
        self.layers_per_block = layers_per_block

        # Tính receptive field để verify
        rf = 1
        for _ in range(num_blocks):
            for d in dilations:
                rf += d * (kernel - 1)
        self.receptive_field = rf

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, C_in, T, N)   — T thường = 13 sau left-pad

        Returns:
            skips_sum : (B, 256, 1, N)   — tổng tất cả skip connections
            x         : (B, C_residual, T'', N)
        """
        all_skips = []

        for block in self.blocks:
            x, skips = block(x)
            all_skips.extend(skips)

        # Align tất cả skips về cùng chiều T (lấy T nhỏ nhất)
        T_min = min(s.shape[2] for s in all_skips)
        skips_aligned = [s[:, :, -T_min:, :] for s in all_skips]

        # Cộng tất cả skip connections lại
        skips_sum = sum(skips_aligned)   # (B, 256, T_min, N)

        return skips_sum, x
