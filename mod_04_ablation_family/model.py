import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCausalSkipAttention(nn.Module):
    def __init__(self, channels, num_heads=4, window_size=16, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        batch_size, channels, num_nodes, seq_len = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, seq_len, channels)

        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        num_heads, head_dim = self.num_heads, self.head_dim
        q = q.view(batch_size * num_nodes, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size * num_nodes, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size * num_nodes, seq_len, num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        for i in range(seq_len):
            lo = max(0, i - self.window_size + 1)
            mask[i, lo : i + 1] = False
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size * num_nodes, num_heads, -1, -1)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, channels)
        out = self.out_proj(out)
        out = self.norm(out)
        return out.view(batch_size, num_nodes, seq_len, channels).permute(0, 3, 1, 2)


def _causal_window_mask(seq_len, window, device):
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        lo = max(0, i - window + 1)
        mask[i, lo : i + 1] = False
    return mask


class RelativePositionalEncoding(nn.Module):
    def __init__(self, num_heads, max_len=64):
        super().__init__()
        self.rel_bias = nn.Embedding(max_len, num_heads)
        nn.init.zeros_(self.rel_bias.weight)

    def forward(self, seq_len):
        device = self.rel_bias.weight.device
        idx = torch.arange(seq_len, device=device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0)
        dist = dist.clamp(max=self.rel_bias.num_embeddings - 1)
        bias = self.rel_bias(dist)
        return bias.permute(2, 0, 1)


class CausalWindowAttnTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.window_size = max(2, 2 * kernel_size)
        self.num_heads = num_heads
        self.head_dim = max(out_channels // num_heads, 1)
        self.scale = math.sqrt(self.head_dim)

        proj_dim = num_heads * self.head_dim
        self.q_proj = nn.Linear(in_channels, proj_dim, bias=False)
        self.k_proj = nn.Linear(in_channels, proj_dim, bias=False)
        self.v_proj = nn.Linear(in_channels, proj_dim, bias=False)
        self.out_proj = nn.Linear(proj_dim, out_channels)
        self.rel_pe = RelativePositionalEncoding(num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)
        self.gate = nn.Linear(out_channels, out_channels)
        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        batch_size, channels, num_nodes, seq_len = x.shape
        x_bn = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, seq_len, channels)

        q = self.q_proj(x_bn)
        k = self.k_proj(x_bn)
        v = self.v_proj(x_bn)

        num_heads, head_dim = self.num_heads, self.head_dim
        q = q.view(batch_size * num_nodes, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size * num_nodes, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size * num_nodes, seq_len, num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = attn + self.rel_pe(seq_len).unsqueeze(0)

        mask = _causal_window_mask(seq_len, self.window_size, x.device)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, num_heads * head_dim)
        out = self.out_proj(out)
        out = out * torch.sigmoid(self.gate(out))
        out = self.norm(out)

        out = out.view(batch_size, num_nodes, seq_len, -1).permute(0, 3, 1, 2)
        residual = self.residual_proj(x)
        return out[:, :, :, 1:] + residual[:, :, :, 1:]


class SkipAggregationAttn(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        while channels % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, skip_list):
        processed = []
        for s in skip_list:
            if s.dim() == 4 and s.size(-1) != 1:
                s = s.mean(dim=-1, keepdim=True)
            processed.append(s)

        batch_size, channels, num_nodes, _ = processed[0].shape
        elems = [p.squeeze(-1).permute(0, 2, 1) for p in processed]
        stacked = torch.stack(elems, dim=2)
        num_layers = stacked.shape[2]
        stacked_2d = stacked.view(batch_size * num_nodes, num_layers, channels)

        attended, _ = self.attn(stacked_2d, stacked_2d, stacked_2d)
        attended = self.norm(attended + stacked_2d)
        out = attended.mean(dim=1)
        out = out.view(batch_size, num_nodes, channels).permute(0, 2, 1)
        return out.unsqueeze(-1)


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
        return torch.einsum("ncvl,vw->ncwl", (x, adjacency)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn_patched(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self._nconv_static = nconv()
        self.mlp = linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def _nconv_dynamic(self, x, adjacency):
        return torch.einsum("bcnl,bnm->bcml", x, adjacency).contiguous()

    def forward(self, x, support):
        out = [x]
        for adjacency in support:
            conv_fn = self._nconv_dynamic if adjacency.dim() == 3 else self._nconv_static
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
        emb_dim=4,
        topk=10,
        use_dynamic_adaptive_adj=True,
        use_skip_attention=True,
        use_skip_aggregation_attention=False,
        use_static_adaptive_optimizations=False,
        adj_update_freq=5,
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.topk = topk
        self.use_dynamic_adaptive_adj = use_dynamic_adaptive_adj and gcn_bool and addaptadj
        self.use_skip_attention = use_skip_attention
        self.use_skip_aggregation_attention = use_skip_aggregation_attention
        self.use_static_adaptive_optimizations = use_static_adaptive_optimizations and gcn_bool and addaptadj and not self.use_dynamic_adaptive_adj
        self.adj_update_freq = adj_update_freq
        self._adj_step_counter = 0
        self._cached_adp = None
        self._cached_new_supports = None
        self.supports = supports

        self.supports_len = 0 if supports is None else len(supports)
        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, emb_dim).to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(emb_dim, num_nodes).to(device), requires_grad=True)
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :emb_dim], torch.diag(p[:emb_dim] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:emb_dim] ** 0.5), n[:, :emb_dim].t())
                self.nodevec1 = nn.Parameter(initemb1.to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2.to(device), requires_grad=True)
            if self.use_dynamic_adaptive_adj or not self.use_dynamic_adaptive_adj:
                self.supports_len += 1

        self.dyn_adj = (
            DynamicAdaptiveAdj(num_nodes, emb_dim, dilation_channels)
            if self.use_dynamic_adaptive_adj
            else None
        )

        self.skip_attentions = nn.ModuleList()
        if self.use_skip_attention:
            self.skip_attentions.extend(
                [
                    LocalCausalSkipAttention(skip_channels, num_heads=4, window_size=16, dropout=dropout)
                    for _ in range(blocks * layers)
                ]
            )
        self.skip_aggregation_attention = (
            SkipAggregationAttn(skip_channels, num_heads=4, dropout=dropout)
            if self.use_skip_aggregation_attention
            else None
        )

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.tcn_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        receptive_field = 1
        for _ in range(blocks):
            new_dilation = 1
            for _ in range(layers):
                self.tcn_layers.append(
                    CausalWindowAttnTCN(
                        residual_channels,
                        dilation_channels,
                        new_dilation,
                        4,
                        dropout,
                    )
                )
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                if gcn_bool:
                    self.gcn_layers.append(
                        gcn_patched(dilation_channels, residual_channels, dropout, self.supports_len)
                    )
                else:
                    self.residual_convs.append(
                        nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
                    )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                receptive_field += new_dilation
                new_dilation *= 2

        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1))

    def _topk_sparse(self, adaptive_adj):
        k = min(self.topk, adaptive_adj.size(1))
        _, topk_idx = torch.topk(adaptive_adj, k, dim=1)
        mask = torch.zeros_like(adaptive_adj)
        mask.scatter_(1, topk_idx, 1.0)
        return adaptive_adj * mask

    def _get_static_adaptive_supports(self, static_supports):
        if (
            self._cached_new_supports is None
            or self._adj_step_counter % self.adj_update_freq == 0
        ):
            adaptive_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            if self.use_static_adaptive_optimizations:
                adaptive_adj = self._topk_sparse(adaptive_adj)
            self._cached_adp = adaptive_adj
            self._cached_new_supports = static_supports + [adaptive_adj]
        self._adj_step_counter += 1
        return self._cached_new_supports

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            input = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))

        x = self.start_conv(input)
        static_supports = self.supports if self.supports is not None else []
        skip = 0
        skip_list = []
        gcn_idx = 0

        for layer_idx in range(self.blocks * self.layers):
            residual = x
            x_tcn = self.tcn_layers[layer_idx](x)

            s = self.skip_convs[layer_idx](x_tcn)
            if self.use_skip_attention:
                s = self.skip_attentions[layer_idx](s)

            if self.use_skip_aggregation_attention:
                if s.size(-1) != 1:
                    s = s.mean(dim=-1, keepdim=True)
                skip_list.append(s)
            else:
                if not isinstance(skip, int):
                    skip = skip[:, :, :, -s.size(3) :]
                skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    if self.use_dynamic_adaptive_adj and self.dyn_adj is not None:
                        adaptive_adj = self.dyn_adj(x_tcn, self.nodevec1, self.nodevec2)
                        current_supports = static_supports + [adaptive_adj]
                    else:
                        current_supports = self._get_static_adaptive_supports(static_supports)
                    x = self.gcn_layers[gcn_idx](x_tcn, current_supports)
                else:
                    x = self.gcn_layers[gcn_idx](x_tcn, static_supports)
                gcn_idx += 1
            else:
                x = self.residual_convs[layer_idx](x_tcn)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[layer_idx](x)

        if self.use_skip_aggregation_attention:
            skip = self.skip_aggregation_attention(skip_list)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
