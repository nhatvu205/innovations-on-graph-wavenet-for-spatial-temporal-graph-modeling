import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAdaptiveAdj(nn.Module):
    def __init__(self, num_nodes, emb_dim=10, in_channels=32):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=(1, 1), bias=True)

    def forward(self, x, nodevec1, nodevec2):
        z = x.mean(dim=-1, keepdim=True)
        z = self.proj(z).squeeze(-1)
        z = z.permute(0, 2, 1)
        nv1 = nodevec1.unsqueeze(0) + z
        nv2 = nodevec2.t().unsqueeze(0) + z
        logits = torch.bmm(nv1, nv2.permute(0, 2, 1))
        return F.softmax(F.relu(logits), dim=-1)


class nconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, adjacency):
        return torch.einsum('ncvl,vw->ncwl', (x, adjacency)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for adjacency in support:
            x1 = self.nconv(x, adjacency)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, adjacency)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gcn_patched(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def _nconv_dynamic(self, x, adjacency):
        return torch.einsum('bcnl,bnm->bcml', x, adjacency).contiguous()

    def forward(self, x, support):
        out = [x]
        for adjacency in support:
            conv_fn = self._nconv_dynamic if adjacency.dim() == 3 else self.nconv
            x1 = conv_fn(x, adjacency)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = conv_fn(x1, adjacency)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
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
        dynamic_adj=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.dynamic_adj = dynamic_adj and addaptadj
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True)
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1.to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2.to(device), requires_grad=True)
            self.supports_len += 1

        if self.dynamic_adj:
            self.dyn_adj = DynamicAdaptiveAdj(num_nodes=num_nodes, emb_dim=10, in_channels=dilation_channels)
        else:
            self.dyn_adj = None

        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
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
                if self.gcn_bool:
                    gconv_cls = gcn_patched if self.dynamic_adj else gcn
                    self.gconv.append(
                        gconv_cls(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
                    )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
        static_supports = self.supports if self.supports is not None else []
        new_supports = None
        if self.gcn_bool and self.supports is not None and self.addaptadj and not self.dynamic_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = static_supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_out * gate

            s = self.skip_convs[i](x)
            if not isinstance(skip, int):
                skip = skip[:, :, :, -s.size(3):]
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj and self.dynamic_adj:
                    adp_dyn = self.dyn_adj(x, self.nodevec1, self.nodevec2)
                    x = self.gconv[i](x, static_supports + [adp_dyn])
                elif self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
