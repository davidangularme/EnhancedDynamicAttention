import numpy as np
from scipy import sparse
from typing import List, Tuple, Callable
import multiprocessing as mp
from functools import partial

class EnhancedDynamicAttention:
    def __init__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                 initial_threshold: int = 10,
                 sparsity_threshold: float = 0.1,
                 adaptive_threshold_func: Callable[[int, int], int] = None):
        """
        Initialize with matrices Q (query), K (key), and V (value).
        
        Args:
            Q (np.ndarray): Query matrix (n x d)
            K (np.ndarray): Key matrix (n x d)
            V (np.ndarray): Value matrix (n x d)
            initial_threshold (int): Initial number of updates to accumulate before recalculation
            sparsity_threshold (float): Threshold for considering a matrix sparse
            adaptive_threshold_func (Callable): Function to dynamically adjust the lazy threshold
        """
        self.Q = Q
        self.K = sparse.csr_matrix(K) if self._is_sparse(K, sparsity_threshold) else K
        self.V = sparse.csr_matrix(V) if self._is_sparse(V, sparsity_threshold) else V
        self.n, self.d = Q.shape
        self.lazy_threshold = initial_threshold
        self.adaptive_threshold_func = adaptive_threshold_func or (lambda x, y: initial_threshold)
        self.pending_updates_K: List[Tuple[int, int, float]] = []
        self.pending_updates_V: List[Tuple[int, int, float]] = []
        self.num_updates = 0
        self.update_history: List[int] = []
        
        self._initialize_matrices()

    @staticmethod
    def _is_sparse(matrix: np.ndarray, threshold: float) -> bool:
        """Determine if a matrix should be treated as sparse."""
        return np.count_nonzero(matrix) / matrix.size < threshold

    def _initialize_matrices(self):
        """Initialize the attention-related matrices."""
        K_dense = self.K.toarray() if sparse.issparse(self.K) else self.K
        self.A = sparse.csr_matrix(np.exp(self.Q @ K_dense.T))
        D_diag = self.A.sum(axis=1).A.flatten()
        self.D = sparse.csr_matrix((D_diag, (range(self.n), range(self.n))))
        self.D_inv = sparse.csr_matrix((1 / D_diag, (range(self.n), range(self.n))))
        V_dense = self.V.toarray() if sparse.issparse(self.V) else self.V
        self.att_matrix = self.D_inv @ self.A @ V_dense

    def apply_lazy_updates(self):
        """Apply all the lazy updates stored for both K and V matrices."""
        if not self.pending_updates_K and not self.pending_updates_V:
            return

        with mp.Pool() as pool:
            if self.pending_updates_K:
                K_update_func = partial(self._update_K_row)
                updated_K_rows = pool.starmap(K_update_func, self.pending_updates_K)
                for i, row in updated_K_rows:
                    if sparse.issparse(self.K):
                        self.K[i] = row
                    else:
                        self.K[i] = row.flatten()
                    K_dense = self.K.toarray() if sparse.issparse(self.K) else self.K
                    self.A[i] = np.exp(self.Q[i] @ K_dense.T)
                    new_D_value = self.A[i].sum()
                    self.D[i, i] = new_D_value
                    self.D_inv[i, i] = 1 / new_D_value

            if self.pending_updates_V:
                for i, j, delta in self.pending_updates_V:
                    if sparse.issparse(self.V):
                        self.V[i, j] += delta
                    else:
                        self.V[i, j] += delta

        V_dense = self.V.toarray() if sparse.issparse(self.V) else self.V
        self.att_matrix = self.D_inv @ self.A @ V_dense

        self.pending_updates_K.clear()
        self.pending_updates_V.clear()
        self.num_updates = 0
        self._adjust_threshold()

    def _update_K_row(self, i: int, j: int, delta: float) -> Tuple[int, np.ndarray]:
        """Update a row of K matrix."""
        if sparse.issparse(self.K):
            row = self.K[i].toarray().flatten()
        else:
            row = self.K[i].copy()
        row[j] += delta
        return i, sparse.csr_matrix(row) if sparse.issparse(self.K) else row

    def _adjust_threshold(self):
        """Dynamically adjust the lazy threshold."""
        self.update_history.append(self.num_updates)
        if len(self.update_history) > 10:  # We keep last 10 update counts
            self.update_history.pop(0)
        avg_updates = sum(self.update_history) / len(self.update_history)
        self.lazy_threshold = self.adaptive_threshold_func(int(avg_updates), self.n * self.d)

    def _lazy_update(self, updates: List[Tuple[int, int, float]], matrix: str):
        """Generic method for lazy updates to K or V matrices."""
        pending_updates = self.pending_updates_K if matrix == 'K' else self.pending_updates_V
        pending_updates.extend(updates)
        self.num_updates += len(updates)
        
        if self.num_updates >= self.lazy_threshold:
            self.apply_lazy_updates()

    def lazy_update_K(self, updates: List[Tuple[int, int, float]]):
        """Update entries of K lazily."""
        self._lazy_update(updates, 'K')

    def lazy_update_V(self, updates: List[Tuple[int, int, float]]):
        """Update entries of V lazily."""
        self._lazy_update(updates, 'V')

    def query(self, i: int, j: int) -> float:
        """
        Query the attention matrix for the value at position (i, j).
        If there are pending lazy updates, apply them first.
        """
        self.apply_lazy_updates()
        return self.att_matrix[i, j]

    def get_attention_matrix(self) -> np.ndarray:
        """Return the current attention matrix, applying any pending updates."""
        self.apply_lazy_updates()
        return self.att_matrix.toarray() if sparse.issparse(self.att_matrix) else self.att_matrix

    def get_approximation_error(self) -> float:
        """
        Calculate the approximation error between the lazy update mechanism
        and a full recalculation.
        """
        # Perform a full recalculation
        K_dense = self.K.toarray() if sparse.issparse(self.K) else self.K
        V_dense = self.V.toarray() if sparse.issparse(self.V) else self.V
        A_full = np.exp(self.Q @ K_dense.T)
        D_full = np.diag(A_full.sum(axis=1))
        D_inv_full = np.linalg.inv(D_full)
        att_matrix_full = D_inv_full @ A_full @ V_dense

        # Compare with the current lazy-updated matrix
        current_matrix = self.get_attention_matrix()
        error = np.linalg.norm(att_matrix_full - current_matrix) / np.linalg.norm(att_matrix_full)
        return error

def adaptive_threshold(avg_updates: int, matrix_size: int) -> int:
    """
    Adaptive threshold function.
    Adjust the threshold based on the average number of updates and matrix size.
    """
    return max(10, min(100, int(avg_updates * np.log10(matrix_size))))

def main():
    np.random.seed(42)  # For reproducibility
    n, d = 1000, 50  # Larger dimensions for a more realistic scenario
    Q = np.random.rand(n, d)
    K = np.random.rand(n, d)
    V = np.random.rand(n, d)

    print("Initializing EnhancedDynamicAttention...")
    dyn_att = EnhancedDynamicAttention(Q, K, V, adaptive_threshold_func=adaptive_threshold)

    print("\nPerforming lazy updates...")
    for _ in range(100):
        k_updates = [(np.random.randint(0, n), np.random.randint(0, d), np.random.randn()*0.1) for _ in range(10)]
        v_updates = [(np.random.randint(0, n), np.random.randint(0, d), np.random.randn()*0.1) for _ in range(10)]
        dyn_att.lazy_update_K(k_updates)
        dyn_att.lazy_update_V(v_updates)

    print("Querying specific positions:")
    print("Position (0, 0):", dyn_att.query(0, 0))
    print("Position (500, 25):", dyn_att.query(500, 25))

    print("\nCalculating approximation error...")
    error = dyn_att.get_approximation_error()
    print(f"Approximation error: {error}")

    print("\nFinal lazy threshold:", dyn_att.lazy_threshold)

if __name__ == "__main__":
    main()
