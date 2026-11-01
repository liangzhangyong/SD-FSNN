"""
BasicAnsatzDecay: SWIM basis functions with exponential decay envelope.

Designed for unbounded domains (e.g., GPE with harmonic trap).
Basis functions:  φ_k(x) = exp(-α·‖x‖²) · tanh(w_k^T·x + b_k)

The envelope g(x) = exp(-α·‖x‖²) → 0 as ‖x‖ → ∞, implicitly enforcing
the boundary condition on unbounded domains.

Frozen PINN SWIM: Directional initialization via project_on_gradient_plane
and probability-weighted sampling (weight_probabilities), aligned to the
gradient of the initial/fiducial solution. This is the key mechanism that
makes SWIM succeed in high dimensions — it focuses basis directions where
the solution varies most.
"""
import numpy as np


# =============================================================================
# Frozen PINN SWIM helpers
# =============================================================================
def project_on_gradient_plane(X1, X2, gradient_fn):
    """
    Project displacement vectors (X2 - X1) onto the plane spanned by the
    function's gradients at X1 and X2.

    This is the core of Frozen PINN SWIM: the gradient ∇f defines the
    directions in which the solution has the most structure. By projecting
    the random displacement vectors onto the gradient span, we obtain
    directions that capture the most informative function variation —
    exactly what SWIM basis functions need for efficient approximation.

    This matches the original Frozen PINN SWIM implementation from
    ``frozen_pinn_swim_projection.ipynb``.

    Parameters
    ----------
    X1 : np.ndarray, shape (N, D)
        "From" points.
    X2 : np.ndarray, shape (N, D)
        "To" points.
    gradient_fn : callable
        gradient_fn(x) -> (N, D) array of ∇f at each point x.

    Returns
    -------
    V_proj : np.ndarray, shape (N, D)
        Projected displacement vectors onto the gradient plane.
    """
    G1 = gradient_fn(X1)  # (N, D)
    G2 = gradient_fn(X2)  # (N, D)

    # Build 2-vector basis from gradients and orthonormalize via Gram-Schmidt
    u1 = G1.copy()
    u1_norm = np.linalg.norm(u1, axis=1, keepdims=True)
    u1_norm = np.clip(u1_norm, a_min=1e-20, a_max=None)
    u1 = u1 / u1_norm

    # u2 = G2 - (G2·u1)u1, then normalize
    u2 = G2 - np.sum(G2 * u1, axis=1, keepdims=True) * u1
    u2_norm = np.linalg.norm(u2, axis=1, keepdims=True)
    u2_norm = np.clip(u2_norm, a_min=1e-20, a_max=None)
    u2 = u2 / u2_norm

    # Displacement
    V_ij = X2 - X1  # (N, D)

    # Project onto the 2D plane spanned by (u1, u2)
    c1 = np.sum(V_ij * u1, axis=1, keepdims=True)  # (N, 1)
    c2 = np.sum(V_ij * u2, axis=1, keepdims=True)  # (N, 1)
    V_proj = c1 * u1 + c2 * u2  # (N, D)

    return V_proj


def weight_probabilities(dy, dists):
    """
    Compute sampling probabilities from function-value differences.

    SWIM samples directions with probability proportional to how much the
    solution changes along that direction (larger |Δψ| → higher weight).
    This focuses the basis on regions of high solution activity.

    P(k) ∝ |Δψ_k|² / dist_k   (or just |Δψ_k|² for dist≈const)

    Parameters
    ----------
    dy : np.ndarray, shape (N, ...)  — Δψ = ψ(to) - ψ(from), broadcasting over output dims
    dists : np.ndarray, shape (N, 1)  — Euclidean distances

    Returns
    -------
    probs : np.ndarray, shape (N,)  — normalized probabilities
    """
    dy_norm = np.linalg.norm(dy, axis=tuple(range(1, dy.ndim)))
    weights = dy_norm**2 / dists.ravel()
    weights = np.clip(weights, a_min=1e-30, a_max=None)
    return weights / weights.sum()


class BasicAnsatzDecay:
    """
    n-D basis: φ_k(x) = g(x) · tanh(w_k^T·x + b_k)
    where g(x) = exp(-alpha · ‖x‖²)  (ExpDecayEnvelope).

    Weights fixed by SWIM data-pair sampler.

    Parameters
    ----------
    n_neurons : int
        Number of basis functions.
    random_state : int
        Random seed for reproducibility.
    alpha : float
        Decay rate of the envelope. Default 0.5 matches the Gaussian
        initial condition exp(-‖x‖²/2).
    dist_min : float
        Minimum distance clamp to avoid division by zero.
    """

    def __init__(self, n_neurons, random_state=42, alpha=0.5, dist_min=1e-10,
                 sparse_dim=0):
        self.n_neurons = n_neurons
        self.rng = np.random.default_rng(random_state)
        self.alpha = alpha
        self.dist_min = dist_min
        self.sparse_dim = sparse_dim  # s per neuron; 0 = dense (all dims)
        self.W = None   # (D, n_neurons)
        self.b = None   # (1, n_neurons)
        # Sparse random features index structure (set by _sparsify_weights)
        self._sp_idx = None   # (n_neurons, s) — active dim indices per neuron
        self._sp_val = None   # (n_neurons, s) — corresponding weight values
        self._sp_valsq = None # (n_neurons,)   — ||w_k||^2 per neuron (precomputed)
        self._is_sparse = False

    def init_model(self, domain, **kwargs):
        """
        SWIM data-pair sampling to initialise W (D, n_neurons) and b (1, n_neurons).

        If sparse_dim > 0, after initialization, sparsify W so each neuron
        only uses sparse_dim randomly selected input dimensions.

        Parameters
        ----------
        domain : Domain
            Must have .interior_points of shape (n_pts, D).
        """
        x = domain.interior_points
        self._init_from_points(x)
        if self.sparse_dim > 0:
            self._sparsify_weights()

    def init_model_random(self, n_dim, n_init_points=2000, x_range=8.0, **kwargs):
        """
        SWIM data-pair sampling using RANDOM quadrature points (no fixed grid dependency).

        This enables mesh-free initialization for high-dimensional problems.

        Parameters
        ----------
        n_dim : int
            Spatial dimension D.
        n_init_points : int
            Number of random initialization points (default 2000).
        x_range : float
            Range for random sampling (default 8.0, matches L in GPE).
        """
        rng = kwargs.get('rng', self.rng)

        # Generate random initialization points (truncated Gaussian for GPE)
        alpha = self.alpha
        sigma = 1.0 / np.sqrt(4.0 * alpha)  # Match decay envelope
        x = rng.normal(0.0, sigma, size=(n_init_points, n_dim))
        # Clip to reasonable range
        x = np.clip(x, -x_range, x_range)

        self._init_from_points(x, rng=rng)

    def _init_from_points(self, x, rng=None):
        """
        Internal helper: initialize weights from given points.

        Parameters
        ----------
        x : np.ndarray, shape (n_pts, D)
            Points for SWIM data-pair sampling.
        rng : np.random.Generator, optional
            Random number generator. Uses self.rng if None.
        """
        if rng is None:
            rng = self.rng

        n = x.shape[0]
        scale = 0.5 * (np.log(1.5) - np.log(0.5))

        n_reps = max(1, int(np.ceil(self.n_neurons / n)))
        idx_from = rng.integers(0, n, size=n * n_reps)
        delta = rng.integers(1, n - 1, size=idx_from.shape[0])
        idx_to = (idx_from + delta) % n

        directions = x[idx_to] - x[idx_from]
        dists = np.linalg.norm(directions, axis=1, keepdims=True).clip(self.dist_min)
        directions = directions / dists

        probs = np.ones(dists.shape[0]) / dists.shape[0]
        sel = rng.choice(dists.shape[0], size=self.n_neurons, replace=True, p=probs)

        directions = directions[sel]
        idx_from_s = idx_from[sel]

        self.W = (2 * scale * directions / dists[sel]).T       # (D, n_neurons)
        self.b = (-np.sum(x[idx_from_s] * self.W.T, axis=-1)).reshape(1, -1) - scale  # (1, n_neurons)

    def init_model_from_psi(self, domain, psi0_fn, psi_grad0_fn=None):
        """
        Frozen PINN SWIM: Initialize basis aligned to |ψ₀| gradient.

        This is the key mechanism that distinguishes Frozen PINN SWIM from
        plain SWIM. Instead of uniformly random tangent directions, we:

        1. Evaluate |ψ| and ∇|ψ| at interior points
        2. Project random (X2-X1) onto the orthogonal complement of ∇|ψ|
           → directions that preserve |ψ| while varying phase (perfect for GPE)
        3. Weight sampling by |∇|ψ||² → focus on high-gradient regions
        4. Use tanh to approximate step discontinuities in |ψ|

        For GPE: ψ_grad0_fn should be the gradient of |ψ₀| = |ψ(x,0)|.
        If not provided, falls back to plain random SWIM.

        Parameters
        ----------
        domain : Domain
            Must have .interior_points of shape (n_pts, D).
        psi0_fn : callable
            psi0_fn(x) -> (n_pts,) complex array of ψ(x,0).
        psi_grad0_fn : callable, optional
            psi_grad0_fn(x) -> (n_pts, D) gradient of |ψ(x,0)|.
            If None, uses |ψ| = |psi0_fn| and calls grad|ψ| numerically.
        """
        x = domain.interior_points
        psi0 = psi0_fn(x)  # (n_pts,) complex

        # |ψ₀|
        abs_psi0 = np.abs(psi0)

        if psi_grad0_fn is None:
            # Numerical gradient of |ψ₀|
            eps = 1e-6
            grad_abs_psi0 = np.zeros_like(x)
            for d in range(x.shape[1]):
                x_plus = x.copy(); x_plus[:, d] += eps
                x_minus = x.copy(); x_minus[:, d] -= eps
                grad_abs_psi0[:, d] = (
                    np.abs(psi0_fn(x_plus)) - np.abs(psi0_fn(x_minus))
                ) / (2.0 * eps)
        else:
            grad_abs_psi0 = psi_grad0_fn(x)

        self._init_from_points_gradient_aware(
            x, abs_psi0, grad_abs_psi0,
            rng=self.rng
        )

    def init_model_from_trajectory(self, domain, u_snaps, v_snaps,
                                   u_fraction=None):
        """Trajectory-aware SWIM sampling (breaks the frozen-at-t=0 limit).

        Standard Frozen PINN SWIM samples (w_k, b_k) using only the IC
        ψ(x,0), which is optimal for problems whose solution keeps the
        same spatial profile (e.g. parabolic / reaction-diffusion with a
        smooth exponentially decaying attractor). For conservative PDEs
        like the GPE, the solution ψ(x,t) changes shape during evolution
        (e.g. Thomas-Fermi spread, phase oscillations), so an IC-only
        basis is under-resolved at later times and the Galerkin residual
        accumulates coherently.

        This method generalises the supervised SWIM sampler to the
        full space-time dataset ``{(x_i, t_k, u_k_i, v_k_i)}``:

        1. Snapshot flattening: replicate the spatial grid N_snap times
           and stack u/v values at all (x, t) combinations into one big
           training vector. The SWIM sampler's |Δu|²/d² (and |Δv|²/d²)
           probability then picks directions (w_k, b_k) that maximise
           the trajectory-wide variation of ψ.
        2. Cross-time pairs are **not** excluded — they let the sampler
           discover directions that capture both spatial and temporal
           structure simultaneously.
        3. The adaptive u/v split of :meth:`_init_from_uv_components`
           is reused to allocate neurons between real/imaginary parts
           according to their global L² amplitudes.

        Parameters
        ----------
        domain : Domain
            Must have ``interior_points`` of shape ``(n_pts, D)``.
        u_snaps, v_snaps : np.ndarray, shape ``(N_snap, n_pts)``
            Real/imaginary parts of ψ at N_snap trajectory snapshots,
            evaluated on ``domain.interior_points``.
        u_fraction : float or None
            Optional override of the u/v neuron allocation; defaults to
            adaptive ``‖u‖ / (‖u‖+‖v‖)``.
        """
        x = domain.interior_points
        u_snaps = np.asarray(u_snaps)
        v_snaps = np.asarray(v_snaps)
        if u_snaps.ndim != 2 or v_snaps.ndim != 2:
            raise ValueError(
                f"u_snaps/v_snaps must be 2-D (N_snap, n_pts); "
                f"got shapes {u_snaps.shape}, {v_snaps.shape}"
            )
        if u_snaps.shape[1] != x.shape[0]:
            raise ValueError(
                f"Snapshot column count {u_snaps.shape[1]} does not match "
                f"domain.interior_points count {x.shape[0]}"
            )

        rng = self.rng
        n, D = x.shape
        scale = 0.5 * (np.log(1.5) - np.log(0.5))

        if u_fraction is None:
            nu = float(np.linalg.norm(u_snaps))
            nv = float(np.linalg.norm(v_snaps))
            tot = nu + nv + 1e-30
            u_fraction = float(np.clip(nu / tot, 0.05, 0.95))

        n_u = int(round(self.n_neurons * u_fraction))
        n_v = self.n_neurons - n_u

        n_reps = max(1, int(np.ceil(self.n_neurons / n)))
        idx_from = rng.integers(0, n, size=n * n_reps)
        delta = rng.integers(1, n - 1, size=idx_from.shape[0])
        idx_to = (idx_from + delta) % n

        disp = x[idx_to] - x[idx_from]
        dists = np.linalg.norm(disp, axis=1, keepdims=True).clip(self.dist_min)
        directions = disp / dists

        du = u_snaps[:, idx_to] - u_snaps[:, idx_from]  # (N_snap, N_pairs)
        dv = v_snaps[:, idx_to] - v_snaps[:, idx_from]
        weights_u = np.sum(np.abs(du) ** 2, axis=0) / (dists.ravel() ** 2)
        weights_v = np.sum(np.abs(dv) ** 2, axis=0) / (dists.ravel() ** 2)
        weights_u = np.clip(weights_u, 1e-30, None)
        weights_v = np.clip(weights_v, 1e-30, None)
        probs_u = weights_u / weights_u.sum()
        probs_v = weights_v / weights_v.sum()

        def _sample_from_probs(probs, n_target):
            if n_target <= 0:
                return None, None
            sel = rng.choice(dists.shape[0], size=n_target, replace=True, p=probs)
            dir_s = directions[sel]
            dist_s = dists[sel]
            from_s = idx_from[sel]
            W_blk = (2 * scale * dir_s / dist_s).T
            b_blk = (-np.sum(x[from_s] * W_blk.T, axis=-1)).reshape(1, -1) - scale
            return W_blk, b_blk

        W_parts, b_parts = [], []
        W_u, b_u = _sample_from_probs(probs_u, n_u)
        if W_u is not None:
            W_parts.append(W_u)
            b_parts.append(b_u)
        W_v, b_v = _sample_from_probs(probs_v, n_v)
        if W_v is not None:
            W_parts.append(W_v)
            b_parts.append(b_v)

        self.W = np.concatenate(W_parts, axis=1)
        self.b = np.concatenate(b_parts, axis=1)
        if self.sparse_dim > 0:
            self._sparsify_weights()

    def _init_from_uv_components(self, x, u_vals, v_vals, rng=None,
                                 u_fraction=None):
        """Supervise SWIM pair sampling on the (u, v) components jointly.

        For GPE the modulus |ψ| varies slowly while the real/imaginary
        components u = Re(ψ), v = Im(ψ) carry the phase oscillation that
        the basis must resolve after a time block. We allocate a share
        of the neurons to supervision by |Δu|²/d² and the rest by
        |Δv|²/d².

        The allocation is **adaptive** to the relative amplitudes of u
        and v: at a block boundary the global phase rotation may leave
        either component near zero (e.g. u≈0 when cos(μt)≈0). Using a
        fixed 50/50 split there wastes half the basis on sampling noise,
        so ``u_fraction`` is chosen proportional to the L² amplitudes.

        Parameters
        ----------
        x : np.ndarray, shape (n_pts, D)
        u_vals, v_vals : np.ndarray, shape (n_pts,)
        rng : np.random.Generator, optional
        u_fraction : float or None
            Explicit u/v split. Default ``None`` uses an adaptive split
            ``‖u‖ / (‖u‖+‖v‖)`` clipped to [0.05, 0.95].
        """
        if rng is None:
            rng = self.rng

        n, D = x.shape
        scale = 0.5 * (np.log(1.5) - np.log(0.5))

        if u_fraction is None:
            nu = float(np.linalg.norm(u_vals))
            nv = float(np.linalg.norm(v_vals))
            tot = nu + nv + 1e-30
            u_fraction = float(np.clip(nu / tot, 0.05, 0.95))

        n_u = int(round(self.n_neurons * u_fraction))
        n_v = self.n_neurons - n_u

        def _sample_block(vals, n_target):
            n_reps = max(1, int(np.ceil(max(n_target, n) / n)))
            idx_from = rng.integers(0, n, size=n * n_reps)
            delta = rng.integers(1, n - 1, size=idx_from.shape[0])
            idx_to = (idx_from + delta) % n
            disp = x[idx_to] - x[idx_from]
            dists = np.linalg.norm(disp, axis=1, keepdims=True).clip(self.dist_min)
            directions = disp / dists
            dv = vals[idx_to] - vals[idx_from]
            weights = (np.abs(dv) / dists.ravel()) ** 2
            weights = np.clip(weights, 1e-30, None)
            probs = weights / weights.sum()
            sel = rng.choice(dists.shape[0], size=n_target, replace=True, p=probs)
            dir_s = directions[sel]
            dist_s = dists[sel]
            from_s = idx_from[sel]
            W_blk = (2 * scale * dir_s / dist_s).T                     # (D, n_target)
            b_blk = (-np.sum(x[from_s] * W_blk.T, axis=-1)).reshape(1, -1) - scale
            return W_blk, b_blk

        W_parts, b_parts = [], []
        if n_u > 0:
            W_u, b_u = _sample_block(u_vals, n_u)
            W_parts.append(W_u); b_parts.append(b_u)
        if n_v > 0:
            W_v, b_v = _sample_block(v_vals, n_v)
            W_parts.append(W_v); b_parts.append(b_v)

        self.W = np.concatenate(W_parts, axis=1)
        self.b = np.concatenate(b_parts, axis=1)
        if self.sparse_dim > 0:
            self._sparsify_weights()

    def _init_from_points_gradient_aware(self, x, abs_psi0, grad_abs_psi0, rng=None):
        """
        Gradient-aware SWIM initialization (Frozen PINN SWIM core).

        Matches the original Frozen PINN SWIM from
        ``frozen_pinn_swim_projection.ipynb``:
        1. Project displacement (X2-X1) onto the gradient SPAN at X1,X2
        2. Weight sampling by |Δψ|/d  (original ``weight_probabilities``)

        Parameters
        ----------
        x : np.ndarray, shape (n_pts, D)
        abs_psi0 : np.ndarray, shape (n_pts,)
            |ψ(x,0)| at each point.
        grad_abs_psi0 : np.ndarray, shape (n_pts, D)
            ∇|ψ(x,0)| at each point.
        rng : np.random.Generator, optional
        """
        if rng is None:
            rng = self.rng

        n = x.shape[0]
        D = x.shape[1]
        scale = 0.5 * (np.log(1.5) - np.log(0.5))

        n_reps = max(1, int(np.ceil(self.n_neurons / n)))
        idx_from = rng.integers(0, n, size=n * n_reps)
        delta = rng.integers(1, n - 1, size=idx_from.shape[0])
        idx_to = (idx_from + delta) % n

        X1 = x[idx_from]
        X2 = x[idx_to]

        # ── Frozen PINN SWIM: project onto gradient plane ──
        # project_on_gradient_plane projects (X2-X1) onto the 2D plane
        # spanned by ∇f(X1) and ∇f(X2), keeping directions where the
        # function varies most (matching the original SWIM paper).
        raw_disp = X2 - X1
        if D == 1:
            # In 1-D, projection is trivial — just use raw displacement
            disp = raw_disp
        else:
            # Build a lookup closure for the gradient at arbitrary points.
            # We pre-computed grad_abs_psi0 at *all* interior points, so
            # the gradient_fn just indexes into that array.
            def _grad_fn_from(pts):
                return grad_abs_psi0[idx_from[:len(pts)]]
            def _grad_fn_to(pts):
                return grad_abs_psi0[idx_to[:len(pts)]]
            V_proj = project_on_gradient_plane(
                X1, X2,
                lambda _x: grad_abs_psi0[idx_from],
            )
            raw_norm = np.linalg.norm(raw_disp, axis=1, keepdims=True)
            proj_norm = np.linalg.norm(V_proj, axis=1, keepdims=True)
            # Blend: use projection where it preserves >=10% of displacement,
            #        otherwise keep raw direction to avoid rank collapse.
            safe = (proj_norm > 0.1 * raw_norm).astype(V_proj.dtype)
            disp = safe * V_proj + (1.0 - safe) * raw_disp
        dists = np.linalg.norm(disp, axis=1, keepdims=True).clip(self.dist_min)
        directions = disp / dists

        # ── Frozen PINN SWIM: probability weighting ──
        # Use |Δψ|/d (matching original weight_probabilities from the
        # reference notebook) for all dimensions.
        dpsi = abs_psi0[idx_to] - abs_psi0[idx_from]
        weights = weight_probabilities(
            dpsi.reshape(-1, 1), dists
        )
        sel = rng.choice(dists.shape[0], size=self.n_neurons, replace=True, p=weights)

        directions = directions[sel]
        idx_from_s = idx_from[sel]

        self.W = (2 * scale * directions / dists[sel]).T    # (D, n_neurons)
        self.b = (-np.sum(x[idx_from_s] * self.W.T, axis=-1)).reshape(1, -1) - scale

    # ----------------------------------------------------------
    # Sparse Random Features
    # ----------------------------------------------------------
    def _init_sparse_random_features(self, x, rng=None):
        """
        Initialize sparse random features: each neuron uses only s=sparse_dim
        randomly selected input dimensions.

        W is stored dense (for compatibility) but has only s nonzeros per column.
        Additionally stores indexed format (_sp_idx, _sp_val) for efficient
        gather-based forward pass O(n·s·K) instead of O(n·D·K).
        """
        if rng is None:
            rng = self.rng
        D = x.shape[1]
        K = self.n_neurons
        s = min(self.sparse_dim, D)
        r_m = 0.05  # weight scale (small for high-dim stability)

        self.W = np.zeros((D, K))
        self._sp_idx = np.zeros((K, s), dtype=np.intp)
        self._sp_val = np.zeros((K, s))

        for j in range(K):
            dims = rng.choice(D, s, replace=False)
            vals = rng.uniform(-r_m, r_m, size=s)
            self._sp_idx[j] = dims
            self._sp_val[j] = vals
            self.W[dims, j] = vals

        self.b = rng.uniform(-r_m, r_m, size=(1, K))
        self._sp_valsq = np.sum(self._sp_val**2, axis=1)  # (K,)
        self._is_sparse = True

    def _sparsify_weights(self):
        """
        Convert an existing dense W into sparse indexed format.
        Called after any dense initialization if sparse_dim > 0.
        """
        D, K = self.W.shape
        s = min(self.sparse_dim, D)
        self._sp_idx = np.zeros((K, s), dtype=np.intp)
        self._sp_val = np.zeros((K, s))
        rng = self.rng

        for j in range(K):
            # Keep only s largest-magnitude weights per neuron
            nz = np.nonzero(self.W[:, j])[0]
            if len(nz) > s:
                top_s = np.argsort(np.abs(self.W[nz, j]))[-s:]
                nz = nz[top_s]
            elif len(nz) < s:
                # Pad with random dims (zero weight)
                remaining = np.setdiff1d(np.arange(D), nz)
                extra = rng.choice(remaining, s - len(nz), replace=False)
                nz = np.concatenate([nz, extra])

            self._sp_idx[j] = nz[:s]
            self._sp_val[j] = self.W[nz[:s], j]

            # Zero out non-selected dims in W
            mask = np.zeros(D, dtype=bool)
            mask[nz[:s]] = True
            self.W[~mask, j] = 0.0

        self._sp_valsq = np.sum(self._sp_val**2, axis=1)
        self._is_sparse = True

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    def _preact(self, x):
        """z_k = w_k^T·x + b_k,  shape (n_pts, n_neurons)"""
        if self._is_sparse:
            return self._preact_sparse(x)
        return x @ self.W + self.b

    def _preact_sparse(self, x):
        """Gather-based preactivation: O(n·s·K) instead of O(n·D·K)."""
        n = x.shape[0]
        s = self._sp_idx.shape[1]
        z = np.zeros((n, self.n_neurons))
        for k in range(s):
            z += x[:, self._sp_idx[:, k]] * self._sp_val[:, k]
        return z + self.b

    def _envelope(self, x):
        """g(x) = exp(-alpha · ‖x‖²),  shape (n_pts, 1). Returns 1 if alpha=0."""
        if self.alpha == 0.0:
            return np.ones((x.shape[0], 1))
        return np.exp(-self.alpha * np.sum(x**2, axis=1, keepdims=True))

    # ----------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------
    def evaluate_model(self, x):
        """φ_k(x) = g(x) · tanh(z_k),  shape (n_pts, n_neurons)"""
        return self._envelope(x) * np.tanh(self._preact(x))

    def evaluate_model_laplace(self, x):
        """
        Exact n-D Laplacian: Δφ_k = Σ_{d=0}^{D-1} ∂²φ_k/∂x_d²

        If sparse RF is active, only sums over active dimensions per neuron
        for the h-dependent terms (h_d1, h_d2 involve w_d which is zero for
        inactive dims), plus a closed-form correction for the g-dependent
        term that sums over ALL D dims.

        Parameters
        ----------
        x : np.ndarray, shape (n_pts, D)

        Returns
        -------
        np.ndarray, shape (n_pts, n_neurons)
        """
        if self._is_sparse:
            return self._evaluate_model_laplace_sparse(x)
        return self._evaluate_model_laplace_dense(x)

    def _evaluate_model_laplace_dense(self, x):
        """Full dense Laplacian (original implementation)."""
        a = self.alpha
        g = self._envelope(x)               # (n, 1)
        z = self._preact(x)                  # (n, K)
        h = np.tanh(z)                       # (n, K)
        h1 = 1.0 - h**2                     # sech²(z)

        # Broadcast over all D dimensions at once
        xd = x[:, :, np.newaxis]             # (n, D, 1)
        wd = self.W[np.newaxis, :, :]        # (1, D, K)
        g3 = g[:, np.newaxis, :]             # (n, 1, 1)

        g_d1 = -2.0 * a * xd * g3                          # ∂g/∂x_d
        g_d2 = (-2.0 * a + 4.0 * a**2 * xd**2) * g3       # ∂²g/∂x_d²

        h3 = h[:, np.newaxis, :]             # (n, 1, K)
        h13 = h1[:, np.newaxis, :]           # (n, 1, K)

        h_d1 = h13 * wd                                    # ∂h/∂x_d
        h_d2 = -2.0 * h3 * h13 * wd**2                    # ∂²h/∂x_d²

        # Sum over D axis -> (n, K)
        return (g_d2 * h3 + 2.0 * g_d1 * h_d1 + g3 * h_d2).sum(axis=1)

    def _evaluate_model_laplace_sparse(self, x):
        """
        Sparse RF Laplacian: exploit that w_d=0 for inactive dimensions.

        Δφ_k = Σ_d [g_d2·h + 2·g_d1·h_d1 + g·h_d2]

        For inactive dims (w_d=0): h_d1=0, h_d2=0, only g_d2·h survives.
        For active dims: full expression.

        g-only term (all D dims):
          Σ_d g_d2 · h = h · Σ_d g_d2 = h · g · [-2αD + 4α²‖x‖²]
          This is a closed-form scalar, O(n·K) — dimension-free.

        Active-dim terms (only s dims):
          Σ_{d∈active} [2·g_d1·h_d1 + g·h_d2]
          Computed via gather, O(n·s·K).
        """
        a = self.alpha
        n = x.shape[0]
        D = x.shape[1]
        K = self.n_neurons
        s = self._sp_idx.shape[1]

        g = self._envelope(x)                 # (n, 1)
        z = self._preact(x)                   # (n, K) — uses sparse _preact
        h = np.tanh(z)                        # (n, K)
        h1 = 1.0 - h**2                      # sech²(z), (n, K)

        # Term 1: g-only (closed-form over ALL D dims)
        #   Σ_d g_d2 = g · [-2αD + 4α²‖x‖²]
        x_sq_sum = np.sum(x**2, axis=1, keepdims=True)  # (n, 1)
        sum_g_d2 = g * (-2.0 * a * D + 4.0 * a**2 * x_sq_sum)  # (n, 1)
        term1 = sum_g_d2 * h  # (n, K)

        # Terms 2 & 3: only active dims via gather
        term23 = np.zeros((n, K))
        for k_idx in range(s):
            dim_idx = self._sp_idx[:, k_idx]   # (K,) active dim for each neuron
            w_val = self._sp_val[:, k_idx]     # (K,) weight value

            xd_k = x[:, dim_idx]               # (n, K) — x at active dim for each neuron

            # g_d1 = -2α·x_d·g
            g_d1_k = -2.0 * a * xd_k * g       # (n, K)
            # h_d1 = sech²(z)·w_d
            h_d1_k = h1 * w_val                 # (n, K)
            # h_d2 = -2·tanh(z)·sech²(z)·w_d²
            h_d2_k = -2.0 * h * h1 * w_val**2  # (n, K)

            term23 += 2.0 * g_d1_k * h_d1_k + g * h_d2_k

        return term1 + term23

    def evaluate_model_laplace_batched(self, x):
        """
        Memory-efficient Laplacian: loop over D dimensions instead of building
        the (n, D, K) intermediate array used by _evaluate_model_laplace_dense.

        Peak memory: O(n·K) per step = same as a single forward pass.
        Suitable for large D (D=10, K=4096, n=5000 → 163 MB peak vs 1.6 GB dense).

        Parameters
        ----------
        x : np.ndarray, shape (n_pts, D)

        Returns
        -------
        np.ndarray, shape (n_pts, K)  — identical result to evaluate_model_laplace
        """
        a = self.alpha
        g = self._envelope(x)      # (n, 1)
        z = self._preact(x)        # (n, K)  — uses sparse path if _is_sparse
        h = np.tanh(z)             # (n, K)
        h1 = 1.0 - h ** 2         # sech², (n, K)

        lap = np.zeros_like(h)     # (n, K)
        for d in range(x.shape[1]):
            xd = x[:, d:d+1]                               # (n, 1)
            wd = self.W[d:d+1, :]                          # (1, K)
            g_d1 = -2.0 * a * xd * g                      # (n, 1)
            g_d2 = (-2.0 * a + 4.0 * a**2 * xd**2) * g   # (n, 1)
            h_d1 = h1 * wd                                 # (n, K)
            h_d2 = -2.0 * h * h1 * wd**2                  # (n, K)
            lap += g_d2 * h + 2.0 * g_d1 * h_d1 + g * h_d2

        return lap

    def evaluate_model_laplace_stochastic(self, x, dim_batch_size, rng):
        """
        Stochastic Dimension Gradient Descent (SDGD) Laplacian estimator.

        Randomly samples dim_batch_size dimensions, computes their
        second-order partials, and scales by D/dim_batch_size for an
        unbiased estimate of the full Laplacian.

        Parameters
        ----------
        x : np.ndarray, shape (n_pts, D)
        dim_batch_size : int
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (n_pts, n_neurons)
        """
        D = x.shape[1]
        a = self.alpha
        g = self._envelope(x)
        z = self._preact(x)
        h = np.tanh(z)
        h1 = 1.0 - h**2

        sampled_dims = rng.choice(D, size=dim_batch_size, replace=False)
        scale = D / dim_batch_size

        xd = x[:, sampled_dims, np.newaxis]
        wd = self.W[np.newaxis, sampled_dims, :]
        g3 = g[:, np.newaxis, :]

        g_d1 = -2.0 * a * xd * g3
        g_d2 = (-2.0 * a + 4.0 * a**2 * xd**2) * g3

        h3 = h[:, np.newaxis, :]
        h13 = h1[:, np.newaxis, :]

        h_d1 = h13 * wd
        h_d2 = -2.0 * h3 * h13 * wd**2

        lap = (g_d2 * h3 + 2.0 * g_d1 * h_d1 + g3 * h_d2).sum(axis=1)
        return scale * lap
