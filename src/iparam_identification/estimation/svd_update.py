"""Incremental SVD update for online estimation.

This module implements incremental SVD updating algorithms that allow
efficient computation of SVD as new data rows arrive, without
recomputing the full SVD from scratch.

Reference:
    Brand, M. (2002). Incremental singular value decomposition of
    uncertain data with missing values. ECCV.

    Kubus et al. 2008, Section IV-D (application to RTLS).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SVDState:
    """State of an incremental SVD computation.

    Represents the thin SVD: A ≈ U @ S @ V^T

    where:
        - U: (m, k) left singular vectors
        - S: (k,) singular values (stored as 1D array)
        - Vt: (k, n) right singular vectors (V transposed)
        - k: rank of the approximation (k <= min(m, n))

    Attributes:
        U: Left singular vectors (m, k).
        S: Singular values as 1D array (k,).
        Vt: Right singular vectors transposed (k, n).
        n_rows: Number of rows processed.
    """

    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    n_rows: int = 0

    def __post_init__(self):
        """Validate shapes."""
        self.S = np.asarray(self.S).ravel()
        k = len(self.S)

        if self.U.shape[1] != k:
            raise ValueError(
                f"U columns ({self.U.shape[1]}) must match S length ({k})"
            )
        if self.Vt.shape[0] != k:
            raise ValueError(
                f"Vt rows ({self.Vt.shape[0]}) must match S length ({k})"
            )

    @property
    def rank(self) -> int:
        """Current rank of the SVD approximation."""
        return len(self.S)

    @property
    def n_cols(self) -> int:
        """Number of columns in the original matrix."""
        return self.Vt.shape[1]

    def get_matrix(self) -> np.ndarray:
        """Reconstruct the approximated matrix U @ S @ V^T."""
        return self.U @ np.diag(self.S) @ self.Vt

    def get_V(self) -> np.ndarray:
        """Get V (not transposed)."""
        return self.Vt.T


def initialize_svd(
    A: np.ndarray,
    max_rank: Optional[int] = None,
    eps: float = 0.0,
    preserve_minimum: bool = True,
) -> SVDState:
    """Initialize SVD state from initial data matrix.

    Args:
        A: Initial data matrix (m, n).
        max_rank: Maximum rank to retain. If None, uses min(m, n).
        eps: Threshold for singular value truncation.
            Use eps=0 to keep all singular values (recommended for TLS).
        preserve_minimum: If True, always keep at least one singular
            value even if below eps (needed for TLS solution extraction).

    Returns:
        SVDState containing initial SVD.
    """
    A = np.asarray(A)
    m, n = A.shape

    # Compute full SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Truncate small singular values
    if eps > 0:
        significant = S > eps
        k = np.sum(significant)
        # For TLS, we need at least the smallest singular vector
        if preserve_minimum and k < len(S):
            k = len(S)  # Keep all for TLS
    else:
        k = len(S)  # Keep all singular values

    if max_rank is not None:
        k = min(k, max_rank)

    U = U[:, :k]
    S = S[:k]
    Vt = Vt[:k, :]

    return SVDState(U=U, S=S, Vt=Vt, n_rows=m)


def incremental_svd_update(
    state: SVDState,
    new_row: np.ndarray,
    max_rank: Optional[int] = None,
    eps: float = 1e-12,
) -> SVDState:
    """Update SVD incrementally with a new row.

    Implements Brand's algorithm for rank-one update when adding
    a new row to the matrix.

    Given existing SVD: A = U @ S @ V^T
    After adding row c: A' = [A; c^T] = U' @ S' @ V'^T

    Args:
        state: Current SVD state.
        new_row: New row to append (n,).
        max_rank: Maximum rank to retain.
        eps: Threshold for orthogonality/truncation.

    Returns:
        Updated SVDState.

    Algorithm (Brand 2002):
        1. Project new row onto existing V space: p = V^T @ c
        2. Compute residual: r = c - V @ p
        3. Form augmented matrices and compute small SVD
        4. Update U, S, V
    """
    new_row = np.asarray(new_row).ravel()

    if new_row.shape[0] != state.n_cols:
        raise ValueError(
            f"New row length ({new_row.shape[0]}) must match "
            f"matrix columns ({state.n_cols})"
        )

    U = state.U
    S = state.S
    Vt = state.Vt
    V = Vt.T

    k = state.rank
    m = state.n_rows
    n = state.n_cols

    # Step 1: Project new row onto V space
    # p = V^T @ c (projection coefficients)
    p = V.T @ new_row  # (k,)

    # Step 2: Compute residual
    # r = c - V @ p (component orthogonal to V)
    r = new_row - V @ p  # (n,)
    r_norm = np.linalg.norm(r)

    if r_norm > eps:
        # New row has component outside V span
        # Need to extend V

        # Normalize residual
        r_hat = r / r_norm  # (n,)

        # Form augmented matrix for small SVD:
        # [S @ P; 0 p_perp] where P accounts for new V column
        # But for row addition, the structure is:
        # [ diag(S)  0 ]   becomes  [U 0]  @ K @ [V r_hat]^T
        # [   p^T   r_norm]         [0 1]

        # Form K matrix (k+1) x (k+1)
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = np.diag(S)
        K[k, :k] = p
        K[k, k] = r_norm

        # Small SVD of K
        Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

        # Update U: U' = [U 0; 0 1] @ Uk
        U_extended = np.zeros((m + 1, k + 1))
        U_extended[:m, :k] = U
        U_extended[m, k] = 1.0
        U_new = U_extended @ Uk

        # Update V: V' = [V r_hat] @ Vk
        V_extended = np.hstack([V, r_hat.reshape(-1, 1)])  # (n, k+1)
        V_new = V_extended @ Vkt.T

        S_new = Sk
        Vt_new = V_new.T

    else:
        # New row lies in span of V
        # No rank increase

        # Form K matrix (k+1) x k
        K = np.zeros((k + 1, k))
        K[:k, :k] = np.diag(S)
        K[k, :] = p

        # Small SVD of K
        Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

        # Update U: U' = [U; 0] @ Uk[:, :k]
        U_extended = np.zeros((m + 1, k))
        U_extended[:m, :] = U
        U_new = np.hstack([U_extended, np.zeros((m + 1, 1))])
        U_new = U_new[:, :Uk.shape[1]]
        U_new = np.vstack([U, np.zeros((1, k))]) @ Uk[:k, :]

        # Actually, simpler formulation:
        U_aug = np.vstack([U, np.zeros((1, k))])  # (m+1, k)
        # K = [diag(S); p^T], SVD gives rotation
        # U_new = [U; 0] @ Uk doesn't quite work...

        # Let me use the correct formulation:
        # For this case, we extend U with a zero row and apply rotation
        U_new_temp = np.zeros((m + 1, k + 1))
        U_new_temp[:m, :k] = U
        U_new_temp[m, k] = 1.0
        U_new = U_new_temp @ Uk[:, :len(Sk)]

        # V doesn't change much
        V_new = V @ Vkt.T[:k, :]
        S_new = Sk
        Vt_new = V_new.T

    # Truncate small singular values
    significant = S_new > eps
    n_significant = np.sum(significant)

    if max_rank is not None:
        n_keep = min(n_significant, max_rank)
    else:
        n_keep = n_significant

    if n_keep < len(S_new):
        U_new = U_new[:, :n_keep]
        S_new = S_new[:n_keep]
        Vt_new = Vt_new[:n_keep, :]

    return SVDState(
        U=U_new,
        S=S_new,
        Vt=Vt_new,
        n_rows=m + 1,
    )


def incremental_svd_update_block(
    state: SVDState,
    new_rows: np.ndarray,
    max_rank: Optional[int] = None,
    eps: float = 1e-12,
) -> SVDState:
    """Update SVD with multiple new rows at once.

    More efficient than calling incremental_svd_update repeatedly
    when adding multiple rows.

    Args:
        state: Current SVD state.
        new_rows: New rows to append (p, n).
        max_rank: Maximum rank to retain.
        eps: Threshold for truncation.

    Returns:
        Updated SVDState.
    """
    new_rows = np.asarray(new_rows)

    if new_rows.ndim == 1:
        new_rows = new_rows.reshape(1, -1)

    if new_rows.shape[1] != state.n_cols:
        raise ValueError(
            f"New rows columns ({new_rows.shape[1]}) must match "
            f"matrix columns ({state.n_cols})"
        )

    U = state.U
    S = state.S
    Vt = state.Vt
    V = Vt.T

    k = state.rank
    m = state.n_rows
    n = state.n_cols
    p = new_rows.shape[0]  # number of new rows

    # Project new rows onto V space
    P = new_rows @ V  # (p, k) - projection coefficients

    # Compute residuals
    R = new_rows - P @ Vt  # (p, n) - orthogonal components

    # QR factorization of residuals
    Q_r, R_r = np.linalg.qr(R.T, mode='reduced')  # Q_r: (n, p), R_r: (p, p)

    # Effective rank of residuals
    r_rank = np.sum(np.abs(np.diag(R_r)) > eps)

    if r_rank > 0:
        Q_r = Q_r[:, :r_rank]
        R_r = R_r[:r_rank, :]

        # Form augmented K matrix
        # K = [ diag(S)   0      ]
        #     [   P      R_r^T   ]
        K = np.zeros((k + p, k + r_rank))
        K[:k, :k] = np.diag(S)
        K[k:k + p, :k] = P
        K[k:k + p, k:k + r_rank] = R_r.T[:p, :r_rank]

        # SVD of K
        Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

        # Update U
        U_extended = np.zeros((m + p, k + p))
        U_extended[:m, :k] = U
        U_extended[m:m + p, k:k + p] = np.eye(p)
        U_new = U_extended @ Uk

        # Update V
        V_extended = np.hstack([V, Q_r])  # (n, k + r_rank)
        V_new = V_extended @ Vkt.T

        S_new = Sk
        Vt_new = V_new.T
    else:
        # New rows lie entirely in span of V
        K = np.zeros((k + p, k))
        K[:k, :k] = np.diag(S)
        K[k:k + p, :] = P

        Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

        U_extended = np.zeros((m + p, k + p))
        U_extended[:m, :k] = U
        U_extended[m:m + p, k:k + p] = np.eye(p)
        U_new = U_extended[:, :Uk.shape[0]] @ Uk

        V_new = V @ Vkt.T
        S_new = Sk
        Vt_new = V_new.T

    # Truncate
    significant = S_new > eps
    n_significant = np.sum(significant)

    if max_rank is not None:
        n_keep = min(n_significant, max_rank)
    else:
        n_keep = n_significant

    if n_keep < len(S_new):
        U_new = U_new[:, :n_keep]
        S_new = S_new[:n_keep]
        Vt_new = Vt_new[:n_keep, :]

    return SVDState(
        U=U_new,
        S=S_new,
        Vt=Vt_new,
        n_rows=m + p,
    )


def downdate_svd(
    state: SVDState,
    row_to_remove: np.ndarray,
    eps: float = 1e-12,
) -> SVDState:
    """Remove a row from the SVD (downdate).

    This is the inverse operation of update - useful for
    sliding window estimation.

    Args:
        state: Current SVD state.
        row_to_remove: Row to remove from the decomposition (n,).
        eps: Numerical threshold.

    Returns:
        Updated SVDState with row removed.

    Note:
        This operation can be numerically unstable if the row
        to remove significantly contributed to singular vectors.
    """
    row = np.asarray(row_to_remove).ravel()

    if row.shape[0] != state.n_cols:
        raise ValueError(
            f"Row length ({row.shape[0]}) must match columns ({state.n_cols})"
        )

    U = state.U
    S = state.S
    Vt = state.Vt
    V = Vt.T

    k = state.rank
    m = state.n_rows

    # Find coefficients of row in U @ S @ V^T representation
    # row ≈ sum_i s_i * (u_i)_j * v_i^T for row j
    # We need to identify which row(s) correspond to this row

    # Project row onto V space
    c = V.T @ row  # (k,) - representation in right singular vector space

    # For downdate, we use the formula from Brand's paper
    # This is essentially the inverse of the update

    # Simplified approach: reconstruct without this row contribution
    # This is approximate but often sufficient

    # Form downdate matrix
    # K = diag(S) - u * c^T where u is the relevant U row
    # But we don't know which U row corresponds to this data row...

    # Alternative: use the general rank-1 downdate formula
    # (A - row @ row^T / ||row||^2) has SVD related to original

    # For now, use a simple approach: reconstruct matrix, remove row, re-SVD
    # This is O(mn) but correct

    # This is a placeholder - full downdate is complex
    # Reconstruct full matrix
    full_matrix = U @ np.diag(S) @ Vt

    # Find and remove the closest matching row
    distances = np.linalg.norm(full_matrix - row.reshape(1, -1), axis=1)
    row_idx = np.argmin(distances)

    # Remove row
    reduced_matrix = np.delete(full_matrix, row_idx, axis=0)

    # Recompute SVD
    return initialize_svd(reduced_matrix, max_rank=k, eps=eps)


def get_tls_solution_from_svd(state: SVDState) -> np.ndarray:
    """Extract TLS solution from SVD state.

    For augmented matrix [A | y], the TLS solution is:
        φ = -v[:-1] / v[-1]

    where v is the right singular vector corresponding to the
    smallest singular value.

    Args:
        state: SVD state of augmented matrix [A | y].

    Returns:
        TLS parameter estimate (n-1,).

    Raises:
        ValueError: If TLS solution is ill-defined.

    Note:
        This function requires that the SVD state contains the
        full rank (all singular values up to n_cols). If the SVD
        has been truncated, the solution may be inaccurate.
    """
    # Find index of smallest singular value
    min_idx = np.argmin(state.S)

    # Get corresponding right singular vector (row of V^T)
    v = state.Vt[min_idx, :]

    # Check solvability
    if np.abs(v[-1]) < 1e-15:
        raise ValueError(
            "TLS solution is ill-defined: last component of smallest "
            "singular vector is nearly zero."
        )

    # TLS solution
    phi = -v[:-1] / v[-1]

    return phi
