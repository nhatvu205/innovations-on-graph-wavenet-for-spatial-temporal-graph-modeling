# =============================================================================
# HELPER: Tạo transition matrices từ adjacency matrix
# =============================================================================
 
def build_transition_matrices(adj: torch.Tensor, adj_type: str = 'doubletransition'):
    """
    Tạo danh sách transition matrices từ pre-built adjacency matrix.
 
    Args:
        adj      : (N, N) weighted adjacency matrix (W_{ij})
        adj_type : 'doubletransition' -> [D_out^{-1}A, D_in^{-1}A^T]
                   'transition'       -> [D_out^{-1}A]
                   'identity'         -> [I]
 
    Returns:
        supports : list of (N, N) Tensors
    """
    def normalize(A):
        """Row-normalize: D^{-1} @ A"""
        D = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return A / D
 
    if adj_type == 'doubletransition':
        T_fwd = normalize(adj)          # D_out^{-1} A
        T_bwd = normalize(adj.T)        # D_in^{-1} A^T
        return [T_fwd, T_bwd]
 
    elif adj_type == 'transition':
        return [normalize(adj)]
 
    elif adj_type == 'identity':
        N = adj.shape[0]
        return [torch.eye(N, device=adj.device)]
 
    else:
        raise ValueError(f"adj_type '{adj_type}' không hợp lệ. "
                         "Chọn: 'doubletransition', 'transition', 'identity'")