import torch
import torch.nn as nn

from shared import util
try:
    from .model import build_optimizer, print_optimizer_lr, gwnet
except ImportError:
    from mod_02_efficiency_family.model import build_optimizer, print_optimizer_lr, gwnet


class trainer:
    def __init__(
        self,
        scaler,
        in_dim,
        seq_length,
        num_nodes,
        nhid,
        dropout,
        lrate,
        wdecay,
        device,
        supports,
        gcn_bool,
        addaptadj,
        aptinit,
        use_static_adaptive_optimizations=False,
        use_causal_window_tcn=False,
        use_skip_aggregation_attention=False,
        use_per_module_lr=False,
        emb_dim=4,
        topk=10,
        adj_update_freq=5,
        attn_lr_multiplier=4.0,
        adj_lr_multiplier=0.5,
    ):
        self.model = gwnet(
            device,
            num_nodes,
            dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=aptinit,
            in_dim=in_dim,
            out_dim=seq_length,
            residual_channels=nhid,
            dilation_channels=nhid,
            skip_channels=nhid * 8,
            end_channels=nhid * 16,
            use_static_adaptive_optimizations=use_static_adaptive_optimizations,
            use_causal_window_tcn=use_causal_window_tcn,
            use_skip_aggregation_attention=use_skip_aggregation_attention,
            emb_dim=emb_dim,
            topk=topk,
            adj_update_freq=adj_update_freq,
        )
        self.model.to(device)
        if use_per_module_lr:
            self.optimizer = build_optimizer(
                self.model,
                base_lr=lrate,
                attn_lr_multiplier=attn_lr_multiplier,
                adj_lr_multiplier=adj_lr_multiplier,
                weight_decay=wdecay,
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.use_per_module_lr = use_per_module_lr

    def print_optimizer_info(self):
        if self.use_per_module_lr:
            print_optimizer_lr(self.optimizer)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        with torch.no_grad():
            output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
