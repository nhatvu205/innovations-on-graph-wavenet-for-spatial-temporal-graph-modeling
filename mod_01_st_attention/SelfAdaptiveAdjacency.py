import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# =============================================================================
# Self-Adaptive Adjacency Matrix
# =============================================================================
 
class SelfAdaptiveAdjacency(nn.Module):
    """
    Học ma trận kề ẩn (adaptive adjacency matrix) thông qua
    hai bảng embedding E1 (source) và E2 (destination).
 
    Công thức:
        A_adp = SoftMax( ReLU( E1 @ E2 ) )
 
    Args:
        num_nodes   : số lượng node (cảm biến), N
        embed_dim   : chiều embedding (mặc định 10 theo paper)
        random_init : True  -> khởi tạo ngẫu nhiên (--randomadj)
                      False -> có thể seed từ SVD của adj có sẵn
    """
 
    def __init__(self, num_nodes: int, embed_dim: int = 10, random_init: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
 
        # E1: (N, 10) — "tôi ảnh hưởng đến ai?"
        # E2: (N, 10) — "tôi bị ai ảnh hưởng?"  (dùng E2.T nên shape (N,10))
        self.E1 = nn.Embedding(num_nodes, embed_dim)
        self.E2 = nn.Embedding(num_nodes, embed_dim)
 
        if random_init:
            nn.init.xavier_uniform_(self.E1.weight)
            nn.init.xavier_uniform_(self.E2.weight)
 
    def forward(self) -> torch.Tensor:
        """
        Returns:
            A_adp : Tensor shape (N, N), row-normalized adaptive adjacency matrix
        """
        node_idx = torch.arange(self.num_nodes, device=self.E1.weight.device)
 
        e1 = self.E1(node_idx)   # (N, 10)
        e2 = self.E2(node_idx)   # (N, 10)
 
        # (N, 10) @ (10, N) -> (N, N)
        logits = torch.mm(e1, e2.T)
        A_adp = F.softmax(F.relu(logits), dim=-1)   # row-softmax sau ReLU
        return A_adp
 
    def seed_from_svd(self, adj: torch.Tensor):
        """
        (Tùy chọn) Khởi tạo E1, E2 từ SVD của ma trận kề có sẵn
        thay vì random. Dùng khi random_init=False.
 
        Args:
            adj : Tensor shape (N, N), pre-built adjacency matrix
        """
        # SVD: adj ≈ U @ diag(S) @ Vt
        U, S, Vt = torch.linalg.svd(adj, full_matrices=False)
        # Lấy embed_dim chiều đầu
        sqrt_S = torch.sqrt(S[:self.embed_dim])
        with torch.no_grad():
            self.E1.weight.copy_(U[:, :self.embed_dim] * sqrt_S)
            self.E2.weight.copy_(Vt[:self.embed_dim, :].T * sqrt_S)
 