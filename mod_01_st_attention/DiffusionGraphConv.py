import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Diffusion Graph Convolution
# =============================================================================
 
class DiffusionGraphConv(nn.Module):
    """
    Graph Convolution layer dùng K-step diffusion.
 
    Hỗ trợ 3 loại transition matrix:
        - Forward  : D_out^{-1} @ A        (traffic xuôi chiều)
        - Backward : D_in^{-1}  @ A^T      (traffic ngược chiều)
        - Adaptive : A_adp (từ SelfAdaptiveAdjacency)
 
    Với diffusion order K=2 và S supports, input feature
    được nhân lần lượt qua:
        [I, T^1, T^2]  với mỗi transition T
    rồi concat lại → (K*S + 1) lần channel → 1x1 conv → output
 
    Công thức:
        Z = sum_s  sum_k  T_s^k @ X @ W_{s,k}
        H = 1x1_conv( concat([X, T1^1@X, T1^2@X,  T2^1@X, T2^2@X, ...]) )
 
    Args:
        c_in         : số channel đầu vào
        c_out        : số channel đầu ra (= residual_channels = 32)
        order        : diffusion order K (mặc định 2)
        num_supports : số lượng transition matrix (không tính identity)
    """
 
    def __init__(self, c_in: int, c_out: int, order: int = 2, num_supports: int = 2):
        super().__init__()
        self.order = order
        self.num_supports = num_supports
 
        # Tổng số "view" được concat:
        # identity (x1) + K bước x num_supports
        total_features = c_in * (1 + order * num_supports)
 
        # 1x1 conv để mix tất cả diffusion features
        self.linear = nn.Linear(total_features, c_out)
 
    def _series_diffusion(self, T: torch.Tensor, X: torch.Tensor) -> list:
        """
        Tính [T^1 @ X, T^2 @ X, ..., T^K @ X]
 
        Args:
            T : (N, N) transition matrix (đã normalize)
            X : (B, N, C) node features
 
        Returns:
            list of K tensors, mỗi tensor shape (B, N, C)
        """
        result = []
        X_k = X
        for _ in range(self.order):
            # (N, N) @ (B, N, C) -> dùng einsum cho batch
            X_k = torch.einsum('nm,bmc->bnc', T, X_k)
            result.append(X_k)
        return result
 
    def forward(
        self,
        X: torch.Tensor,
        supports: list,
        A_adp: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            X        : (B, N, C_in) node features tại một timestep
            supports : list of pre-built transition matrices (N, N)
                       thường là [T_forward, T_backward]
            A_adp    : (N, N) adaptive adjacency matrix (optional)
                       nếu None thì không dùng adaptive
 
        Returns:
            out : (B, N, C_out)
        """
        # Bắt đầu với identity (0th order) — bản thân X
        diffused = [X]
 
        # Tất cả transition matrices cần xét
        all_supports = list(supports)
        if A_adp is not None:
            all_supports.append(A_adp)
 
        # K-step diffusion cho mỗi transition matrix
        for T in all_supports:
            diffused.extend(self._series_diffusion(T, X))
 
        # Concat theo chiều channel: (B, N, C_in * (1 + K*S))
        out = torch.cat(diffused, dim=-1)
 
        # 1x1 conv (linear trên chiều feature): (B, N, C_out)
        out = self.linear(out)
        return out
 
 