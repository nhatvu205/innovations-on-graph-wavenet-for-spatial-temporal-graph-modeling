import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    """Batched graph neighbourhood convolution: x_{v} = sum_{u} A_{vu} x_{u}."""

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    """1x1 Conv2d wrapper used as a channel-mixing MLP."""

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    """
    K-step diffusion graph convolution.

    Concatenates 0th-order through K-th-order diffused features for every
    support matrix, then mixes with a 1x1 conv.

    Args:
        c_in:        input channel size
        c_out:       output channel size
        dropout:     dropout probability
        support_len: number of support (transition) matrices
        order:       diffusion order K (default 2)
    """

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    """
    Graph WaveNet model (Wu et al., IJCAI 2019).

    Architecture:
        start 1x1 Conv (in_dim -> residual_channels)
        -> blocks x layers ST-layers, each:
               Gated TCN (dilated causal conv, tanh * sigmoid)
               Diffusion GCN
               Residual add + BatchNorm
               Skip accumulation
        -> ReLU -> 1x1 Conv (skip -> end) -> 1x1 Conv (end -> out_dim)

    Args:
        device:             torch.device
        num_nodes:          number of graph nodes N
        dropout:            dropout probability (default 0.3)
        supports:           list of pre-built adjacency matrices (torch.Tensor)
        gcn_bool:           whether to use graph convolution
        addaptadj:          whether to learn a self-adaptive adjacency matrix
        aptinit:            optional initial matrix to seed node embeddings via SVD
        in_dim:             input feature dimension (default 2)
        out_dim:            number of prediction steps (default 12)
        residual_channels:  channel width for residual stream (default 32)
        dilation_channels:  channel width inside gated TCN (default 32)
        skip_channels:      channel width for skip accumulation (default 256)
        end_channels:       intermediate channel width before output (default 512)
        kernel_size:        TCN kernel size (default 2)
        blocks:             number of blocks (default 4)
        layers:             number of ST-layers per block (default 2)
    """

    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
    ):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.supports = supports

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            if aptinit is None:
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10).to(device), requires_grad=True
                ).to(device)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes).to(device), requires_grad=True
                ).to(device)
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
            self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
                    )

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True
        )
        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            # Gated activation: tanh(filter) * sigmoid(gate)
            filter_ = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_ * gate

            # Skip connection
            s = self.skip_convs[i](x)
            if not isinstance(skip, int):
                skip = skip[:, :, :, -s.size(3):]
            skip = s + skip

            # Graph convolution or fallback 1x1 conv
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            # Residual connection
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
