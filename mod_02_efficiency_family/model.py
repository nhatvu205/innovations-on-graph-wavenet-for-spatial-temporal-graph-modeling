import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_EMB_DIM = 4
DEFAULT_TOPK = 10
DEFAULT_ADJ_UPDATE_FREQ = 5


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
            if s.dim() == 4:
                s = s.mean(dim=-1, keepdim=True)
            processed.append(s)

        batch_size, channels, num_nodes, _ = processed[0].shape
        elems = [p.squeeze(-1).permute(0, 2, 1) for p in processed]
        stacked = torch.stack(elems, dim=2)
        stacked_2d = stacked.view(batch_size * num_nodes, len(processed), channels)

        attended, _ = self.attn(stacked_2d, stacked_2d, stacked_2d)
        attended = self.norm(attended + stacked_2d)
        out = attended.mean(dim=1)
        out = out.view(batch_size, num_nodes, channels).permute(0, 2, 1)
        return out.unsqueeze(-1)


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
        use_static_adaptive_optimizations=False,
        use_causal_window_tcn=False,
        use_skip_aggregation_attention=False,
        emb_dim=DEFAULT_EMB_DIM,
        topk=DEFAULT_TOPK,
        adj_update_freq=DEFAULT_ADJ_UPDATE_FREQ,
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.use_static_adaptive_optimizations = use_static_adaptive_optimizations and gcn_bool and addaptadj
        self.use_causal_window_tcn = use_causal_window_tcn
        self.use_skip_aggregation_attention = use_skip_aggregation_attention
        self.topk = topk
        self.adj_update_freq = adj_update_freq
        self._adj_step_counter = 0
        self._cached_adp = None
        self._cached_new_supports = None

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.tcn_layers = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            node_emb_dim = emb_dim if self.use_static_adaptive_optimizations else 10
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, node_emb_dim).to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(node_emb_dim, num_nodes).to(device), requires_grad=True)
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :node_emb_dim], torch.diag(p[:node_emb_dim] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:node_emb_dim] ** 0.5), n[:, :node_emb_dim].t())
                self.nodevec1 = nn.Parameter(initemb1.to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2.to(device), requires_grad=True)
            self.supports_len += 1

        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                if self.use_causal_window_tcn:
                    self.tcn_layers.append(
                        CausalWindowAttnTCN(
                            in_channels=residual_channels,
                            out_channels=dilation_channels,
                            kernel_size=new_dilation,
                            num_heads=4,
                            dropout=dropout,
                        )
                    )
                else:
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
                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
                self.residual_convs.append(
                    nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.skip_agg_attn = (
            SkipAggregationAttn(channels=skip_channels, num_heads=8, dropout=dropout)
            if self.use_skip_aggregation_attention
            else None
        )
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def _topk_sparse(self, adaptive_adj):
        k = min(self.topk, adaptive_adj.size(1))
        _, topk_idx = torch.topk(adaptive_adj, k, dim=1)
        mask = torch.zeros_like(adaptive_adj)
        mask.scatter_(1, topk_idx, 1.0)
        return adaptive_adj * mask

    def _compute_and_cache_adj(self):
        adaptive_adj_full = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adaptive_adj_sparse = self._topk_sparse(adaptive_adj_full)
        adaptive_cached = adaptive_adj_sparse.detach()
        self._cached_adp = adaptive_cached
        self._cached_new_supports = self.supports + [adaptive_cached]

    def _get_supports(self):
        if not (self.gcn_bool and self.addaptadj and self.supports is not None):
            return self.supports
        if self.use_static_adaptive_optimizations:
            if self.training:
                should_update = (
                    self._cached_new_supports is None
                    or self._adj_step_counter % self.adj_update_freq == 0
                )
                if should_update:
                    adaptive_adj_full = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                    adaptive_adj_live = self._topk_sparse(adaptive_adj_full)
                    self._compute_and_cache_adj()
                    supports = self.supports + [adaptive_adj_live]
                else:
                    supports = self._cached_new_supports
                self._adj_step_counter += 1
                return supports
            adaptive_adj_full = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adaptive_adj_sparse = self._topk_sparse(adaptive_adj_full)
            return self.supports + [adaptive_adj_sparse.detach()]

        adaptive_adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return self.supports + [adaptive_adj]

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0
        skip_list = []
        current_supports = self._get_supports()

        for layer_idx in range(self.blocks * self.layers):
            residual = x
            if self.use_causal_window_tcn:
                x = self.tcn_layers[layer_idx](x)
            else:
                filter_out = torch.tanh(self.filter_convs[layer_idx](residual))
                gate = torch.sigmoid(self.gate_convs[layer_idx](residual))
                x = filter_out * gate

            s = self.skip_convs[layer_idx](x)
            if self.use_skip_aggregation_attention:
                skip_list.append(s)
            else:
                if not isinstance(skip, int):
                    skip = skip[:, :, :, -s.size(3):]
                skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[layer_idx](x, current_supports)
                else:
                    x = self.gconv[layer_idx](x, self.supports)
            else:
                x = self.residual_convs[layer_idx](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[layer_idx](x)

        if self.use_skip_aggregation_attention:
            x = self.skip_agg_attn(skip_list)
        else:
            x = skip

        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


def build_optimizer(model, base_lr=0.001, attn_lr_multiplier=4.0, adj_lr_multiplier=0.5, weight_decay=1e-4):
    adj_params, attn_params, base_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'nodevec' in name:
            adj_params.append(param)
        elif any(k in name for k in ['tcn_layers', 'skip_agg_attn', 'rel_pe', 'q_proj', 'k_proj', 'v_proj', 'out_proj', 'rel_bias', 'gate']):
            attn_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {'params': base_params, 'lr': base_lr, 'name': 'base'},
        {'params': attn_params, 'lr': base_lr * attn_lr_multiplier, 'name': 'attention'},
        {'params': adj_params, 'lr': base_lr * adj_lr_multiplier, 'name': 'adaptive_adj'},
    ]
    return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def print_optimizer_lr(optimizer):
    for group in optimizer.param_groups:
        name = group.get('name', '?')
        n_params = len(group['params'])
        print(f'  [{name:20s}]  lr={group["lr"]:.2e}  n_params={n_params}')


def register_grad_monitor(model):
    hooks = []

    def make_hook(name):
        def hook(grad):
            if grad is not None:
                norm = grad.norm().item()
                if norm > 0:
                    print(f'  grad_norm [{name:40s}] = {norm:.4f}')
        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            hooks.append(param.register_hook(make_hook(name)))
    return hooks
