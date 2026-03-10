import torch
import torch.nn as nn
import torch.optim as optim

from src.model import gwnet
from src import util


class trainer:
    """
    Wraps gwnet for training and evaluation.

    The input tensor is left-padded by 1 zero column before being passed to the
    model, following the original implementation.

    Args:
        scaler:      StandardScaler fitted on the training set
        in_dim:      input feature dimension (typically 2)
        seq_length:  number of prediction steps (output dim)
        num_nodes:   number of graph nodes
        nhid:        hidden channel width (residual_channels = dilation_channels = nhid)
        dropout:     dropout probability
        lrate:       Adam learning rate
        wdecay:      Adam weight decay
        device:      torch.device
        supports:    list of pre-built adj matrices (torch.Tensor) or None
        gcn_bool:    enable graph convolution
        addaptadj:   enable self-adaptive adjacency matrix
        aptinit:     optional matrix to seed node embeddings via SVD
    """

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
        )
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        """
        One gradient step.

        Args:
            input:    (B, C, N, T) — already transposed by caller
            real_val: (B, N, T)    — already transposed by caller
        Returns:
            (mae, mape, rmse) as Python floats
        """
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
        """
        Forward pass without gradient computation.

        Returns:
            (mae, mape, rmse) as Python floats
        """
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
