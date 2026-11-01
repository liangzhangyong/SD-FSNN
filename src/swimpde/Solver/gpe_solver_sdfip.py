"""
SDIFP-style GPE Solver with random sampling collocation.

Key idea: replace the implicit uniform-weight trapezoidal quadrature
used in Nonlinear_GPE_Solver with explicit Monte-Carlo / importance-sampling
quadrature, so that the augmented ODE constraint (mass-penalty term) and the
SDRE dynamics are computed with the SAME quadrature rule.

This is the mesh-free analogue of Nonlinear_GPE_Solver.  It achieves
high accuracy under random sampling by:

  1. Sampling quadrature nodes from the Gaussian envelope N(0, 1/(2α)),
     weighted so that  Σ w_q·f(x_q)  ≈  ∫ f(x)·exp(-2α|x|²) dx.
  2. Building all matrices (_C, _M_V, _G_quad) with those MC weights,
     guaranteeing consistency between the augmented ODE and the final
     mass projection.
  3. Using the same nodes as both basis-evaluation nodes (A_inv) and
     quadrature nodes, so the "integral = volume_x * mean" simplification
     is exact (up to MC variance).

Usage (D=1 is the primary target):
    from swimpde.Solver.gpe_solver_sdfip import GPE_SDIFP_Solver
    solver = GPE_SDIFP_Solver(
        domain=domain,        # any Domain; interior_points are used as seed only
        ansatz=ansatz,        # BasicAnsatzDecay
        u0=initial_u,
        v0=initial_v,
        beta=10.0,
        gamma=2.0,
        svd_cutoff=1e-14,
        n_quad=4096,          # MC quadrature nodes (per call)
        seed_quad=42,         # RNG seed for reproducible sampling
        mass_projection=True,
        mass_penalty=1e0,
    )
    solver, status = solver.fit((0, T_final), t_eval=t_eval)
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Callable
from scipy.integrate import solve_ivp
import copy


@dataclass
class GPE_SDIFP_Solver:
    """
    SDIFP-style SDRE solver for the GPE with Monte-Carlo quadrature.

    All spatial integrals are computed with importance-sampling quadrature
    using the Gaussian envelope N(0, σ²) where σ = 1/√(4α).  This guarantees
    consistency between the augmented ODE (penalty term) and the final
    mass projection.

    Parameters
    ----------
    domain : Domain
        Domain object (interior_points are used as seed for MC sampling).
    ansatz : BasicAnsatzDecay
        Basis functions with exponential-decay envelope.
    u0, v0 : Callable
        Initial conditions for real and imaginary parts.
    beta, gamma : float
        GPE parameters.
    svd_cutoff : float
        Relative SVD truncation threshold.
    n_quad : int
        Number of Monte-Carlo quadrature nodes per call.
    seed_quad : int
        RNG seed for reproducible quadrature sampling.
    ode_solver : str
        ODE method (default 'DOP853').
    mass_projection : bool
        If True, compute Gram matrix via MC quadrature and apply exact
        mass projection at end of fit().
    mass_penalty : float
        Penalty weight for augmented ODE mass-conservation row.

    Attributes
    ----------
    n_basis : int
        Number of SVD-truncated basis functions.
    sol : solve_ivp solution object
    """

    domain: object
    ansatz: object
    u0: Callable
    v0: Callable
    beta: float = 10.0
    gamma: float = 2.0
    svd_cutoff: float = 1e-10
    n_quad: int = 4096
    seed_quad: int = 42
    ode_solver: str = 'DOP853'
    mass_projection: bool = False
    mass_penalty: float = 1e0

    # ── Internal state ──────────────────────────────────────────────────────────
    _A_r: np.ndarray = field(default=None, repr=False)
    _A_inv: np.ndarray = field(default=None, repr=False)
    _C: np.ndarray = field(default=None, repr=False)
    _M_V: np.ndarray = field(default=None, repr=False)
    _Vt_s: np.ndarray = field(default=None, repr=False)
    _n_basis: int = field(default=0, repr=False)
    _sol: object = field(default=None, repr=False)
    _G_quad: np.ndarray = field(default=None, repr=False)
    _N0: float = field(default=1.0, repr=False)
    _quad_cache: dict = field(default_factory=dict, repr=False)

    # ─────────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────────

    @property
    def n_basis(self):
        return self._n_basis

    @property
    def sol(self):
        return self._sol

    def _generate_quadrature_nodes(self, n_q=None, seed=None):
        """
        Generate MC quadrature nodes from the Gaussian importance distribution.

        The envelope φ(x) ∝ exp(-α|x|²) has variance 1/(2α) per dimension.
        We sample x ~ N(0, σ²) with σ = 1/√(4α) so that:

            E_{x~N}[f(x)·w(x)]  =  ∫ f(x)·exp(-2α|x|²) dx

        where  w(x) = (2α/π)^{D/2} · exp(-2α|x|²).

        Parameters
        ----------
        n_q : int, optional
            Number of nodes (defaults to self.n_quad).
        seed : int, optional
            RNG seed (defaults to self.seed_quad).

        Returns
        -------
        x_q : np.ndarray, shape (n_q, D)
            Quadrature nodes.
        w_q : np.ndarray, shape (n_q,)
            Importance-sampling weights (sum w_q = 1).
        """
        if n_q is None:
            n_q = self.n_quad
        if seed is None:
            seed = self.seed_quad

        cache_key = (n_q, seed)
        if cache_key in self._quad_cache:
            return self._quad_cache[cache_key]

        rng = np.random.default_rng(seed)
        D = self.domain.interior_points.shape[1]
        alpha = self.ansatz.alpha

        sigma = 1.0 / np.sqrt(4.0 * alpha)
        x_q = rng.normal(0.0, sigma, size=(n_q, D))

        # Importance-sampling weight:
        #   w_q ∝ exp(-2α|x_q|²) / NormalPDF(x_q; 0, σ²)
        #   = exp(-2α|x|²) / (1/(√(2π)σ))^D · exp(-|x|²/(2σ²))
        #   ∝ exp(-2α|x|²) · exp(-2α|x|²)
        #   = exp(-4α|x|²)
        # So the weight is proportional to exp(-4α|x|²).
        # To get proper weights that sum to 1, we use the un-normalized form
        # and divide by the sum:
        log_q = (0.5 * D * np.log(2.0 * alpha / np.pi)
                 - 2.0 * alpha * np.sum(x_q**2, axis=1))
        w_raw = np.exp(-log_q)
        w_q = w_raw / w_raw.sum()

        self._quad_cache[cache_key] = (x_q, w_q)
        return x_q, w_q

    def _build_matrices(self):
        """
        Build SVD-truncated basis matrices and SDRE operators using
        MC quadrature.

        For MC quadrature we use the SAME nodes for:
          (a) basis evaluation (A_inv, A_r)
          (b) quadrature (C, M_V, G_quad)

        This is the key difference from Nonlinear_GPE_Solver, which uses
        fixed interior points for (a) and Gauss-Hermite nodes for (b).
        """
        x_q, w_q = self._generate_quadrature_nodes()
        n_q = x_q.shape[0]
        alpha = self.ansatz.alpha

        # ── Basis evaluation on MC nodes ───────────────────────────────────────
        A_q = self.ansatz.evaluate_model(x_q)
        A_lap_q = self.ansatz.evaluate_model_laplace(x_q)

        # SVD on MC evaluation matrix (consistent with MC quadrature)
        U_s, S_s, Vt_s = np.linalg.svd(A_q, full_matrices=False)
        mask = S_s / S_s[0] > self.svd_cutoff
        self._Vt_s = Vt_s[mask, :]
        self._n_basis = int(mask.sum())

        if self._n_basis == 0:
            raise ValueError(
                f"SVD cutoff={self.svd_cutoff} removed all basis functions. "
                "Lower svd_cutoff."
            )

        print(f"SVD truncation: {self.ansatz.n_neurons} -> {self._n_basis} basis "
              f"functions (on {n_q} MC nodes)")

        A_r = A_q @ self._Vt_s.T
        A_lap_r = A_lap_q @ self._Vt_s.T

        self._A_r = A_r
        self._A_inv = np.linalg.pinv(A_r, rcond=self.svd_cutoff)

        # ── Laplacian matrix (MC quadrature) ───────────────────────────────────
        #   C_{ki} = ∫ φ_k(x)·Δφ_i(x)·exp(-2α|x|²) dx
        #            ≈ Σ_j w_j · A_r[j,k] · A_lap_r[j,i]
        # C[k,i] = Σ_j w_j · A_r[j,k] · A_lap_r[j,i] = (A_r.T diag(w_q) A_lap_r)[k,i]
        # A_r.T (n_basis, n_q) * w_q (n_q,) → (n_basis, n_q) [broadcast adds dim-1]
        self._C = (A_r.T * w_q) @ A_lap_r   # (n_basis, n_basis)

        # ── Potential matrix (MC quadrature) ───────────────────────────────────
        #   M_V_{ki} = ∫ V(x)·φ_k(x)·φ_i(x)·exp(-2α|x|²) dx
        #            ≈ Σ_j w_j · V_q[j] · A_r[j,k] · A_r[j,i]
        # M_V[k,i] = Σ_j (w_j·V_q[j]) · A_r[j,k] · A_r[j,i]
        # A_r.T diag(w_q·V_q) A_r: need element-wise (w_q[:,None] * V_q)
        V_q = self._V_potential(x_q)  # shape (n_q,)
        self._M_V = (A_r.T * (w_q * V_q)) @ A_r   # (n_basis, n_basis)

        # ── Gram matrix for mass (MC quadrature) ───────────────────────────────
        if self.mass_projection:
            self._build_quad_gram_mc(x_q, w_q, A_r)

    def _build_quad_gram_mc(self, x_q, w_q, A_r):
        """
        Build the mass Gram matrix via MC quadrature on the same nodes.

        Mass:  N(t) = c_u^T G c_u + c_v^T G c_v
        where  G_{ki} = ∫ φ_k(x)·φ_i(x)·exp(-2α|x|²) dx
                     ≈ Σ_j w_j · A_r[j,k] · A_r[j,i]
        """
        D = self.domain.interior_points.shape[1]
        alpha = self.ansatz.alpha
        print(f"  Mass projection (SDIFP MC): n_quad={x_q.shape[0]}, "
              f"alpha={alpha}, D={D}")

        # w_q · A_r: (n_q, n_basis)
        wA = A_r * np.sqrt(w_q[:, None])
        self._G_quad = wA.T @ wA   # (n_basis, n_basis)

    def _project_initial_conditions(self):
        """Project u0, v0 onto the truncated basis using MC quadrature."""
        x_q, w_q = self._generate_quadrature_nodes()
        u0_vals = self.u0(x_q)
        v0_vals = self.v0(x_q)

        c_u0 = self._A_inv @ u0_vals
        c_v0 = self._A_inv @ v0_vals

        # Reconstruction quality check
        u0_rec = self._A_r @ c_u0
        rel_err = np.linalg.norm(u0_rec - u0_vals) / (np.linalg.norm(u0_vals) + 1e-30)
        print(f"IC reconstruction (MC quadrature): L2 rel err = {rel_err:.2e}")

        if self.mass_projection and self._G_quad is not None:
            self._N0 = float(c_u0 @ self._G_quad @ c_u0 + c_v0 @ self._G_quad @ c_v0)
            print(f"  Exact mass (MC): N(0) = {self._N0:.16f}")

        return c_u0, c_v0

    def _make_ode_rhs(self):
        """
        Build the ODE RHS with MC-quadrature matrices.

        The augmented mass-conservation row uses the same G_quad that
        the final projection uses, ensuring consistency.
        """
        A_r = self._A_r
        A_inv = self._A_inv
        C = self._C
        M_V = self._M_V
        n = self._n_basis
        beta = self.beta
        do_aug = self.mass_projection and self._G_quad is not None
        G = self._G_quad if do_aug else None
        N0 = self._N0 if do_aug else 1.0
        penalty = self.mass_penalty

        if do_aug:
            G2 = G @ G

        def ode_rhs(t, y):
            c_u, c_v = y[:n], y[n:]

            # Reconstruct at MC quadrature nodes
            u_vals = A_r @ c_u
            v_vals = A_r @ c_v
            psi2 = u_vals**2 + v_vals**2

            # ── Standard SDRE dynamics (MC quadrature) ──────────────────────────
            #   du/dt = -½Δv + V·v + β|ψ|²·v
            #   dv/dt =  ½Δu - V·u - β|ψ|²·u
            # In coefficient space (with MC quadrature for all integrals):
            #   dc_u = -½·C·c_v + M_V·c_v + β·A_inv·(ψ²·v)
            #   dc_v =  ½·C·c_u - M_V·c_u - β·A_inv·(ψ²·u)
            psi2_v = psi2 * v_vals
            psi2_u = psi2 * u_vals
            dc_u = -0.5 * (C @ c_v) + M_V @ c_v + beta * (A_inv @ psi2_v)
            dc_v =  0.5 * (C @ c_u) - M_V @ c_u - beta * (A_inv @ psi2_u)

            if do_aug:
                # ── Augmented mass-conservation row ─────────────────────────────
                # Mass: N_t = c_u^T G c_u + c_v^T G c_v
                N_t = c_u @ G @ c_u + c_v @ G @ c_v
                dN = N_t - N0

                # Mass rate from free dynamics
                Gc_u = G @ c_u
                Gc_v = G @ c_v
                dN_free = 2.0 * (c_u @ (G @ dc_u) + c_v @ (G @ dc_v))

                # Target mass rate
                dN_target = -penalty * dN

                # Lagrange multiplier correction
                denom = 2.0 * (c_u @ (G2 @ c_u) + c_v @ (G2 @ c_v))
                if abs(denom) > 1e-30:
                    lam = (dN_target - dN_free) / denom
                    dc_u = dc_u + lam * Gc_u
                    dc_v = dc_v + lam * Gc_v

            return np.concatenate([dc_u, dc_v])

        return ode_rhs

    def fit(self, t_span, t_eval=None, rtol=1e-6, atol=1e-8):
        """
        Solve the GPE with SDIFP MC quadrature.

        Parameters
        ----------
        t_span : tuple (t0, tf)
            Integration interval.
        t_eval : array-like, optional
            Times at which to store the solution.
        rtol, atol : float
            ODE solver tolerances.

        Returns
        -------
        self : GPE_SDIFP_Solver
        status : int
            0 if successful.
        """
        self.ansatz.init_model(self.domain)
        self._build_matrices()
        c_u0, c_v0 = self._project_initial_conditions()
        y0 = np.concatenate([c_u0, c_v0])

        ode_rhs = self._make_ode_rhs()
        sol = solve_ivp(
            ode_rhs, t_span, y0,
            method=self.ode_solver,
            t_eval=t_eval,
            dense_output=True,
            rtol=rtol, atol=atol,
        )
        self._sol = sol

        if sol.status != 0:
            print(f"Warning: ODE solver returned status={sol.status}: {sol.message}")

        # ── Exact mass projection (direct SDIFP mapping) ─────────────────────────
        #   ψ(t) → ψ(t) · √(N₀ / N(t))
        # The projection uses the SAME G_quad as the augmented ODE,
        # guaranteeing N(t) = N₀ exactly at every time.
        if self.mass_projection and self._G_quad is not None and sol.sol is not None:
            n = self._n_basis
            G = self._G_quad
            N0 = self._N0
            _dense = sol.sol

            def projected_dense(t, _d=_dense, _G=G, _N0=N0, _n=n):
                y = _d(t)
                cu, cv = y[:_n], y[_n:]
                Nt = float(cu @ _G @ cu + cv @ _G @ cv)
                scale = np.sqrt(_N0 / Nt) if Nt > 0 else 1.0
                return np.concatenate([scale * cu, scale * cv])

            sol.sol = projected_dense

            # Project stored snapshots
            for j in range(sol.y.shape[1]):
                cu = sol.y[:n, j]
                cv = sol.y[n:, j]
                Nt = float(cu @ G @ cu + cv @ G @ cv)
                if Nt > 0:
                    scale = np.sqrt(N0 / Nt)
                    sol.y[:n, j] = scale * cu
                    sol.y[n:, j] = scale * cv

        self._coefficients_c = sol.sol if sol.sol is not None else None
        return self, sol.status

    def evaluate(self, x_eval, t_eval):
        """
        Evaluate |ψ(x, t)| = √(u² + v²) at given space-time points.

        Parameters
        ----------
        x_eval : np.ndarray, shape (n_eval, D)
        t_eval : np.ndarray, shape (n_t,)

        Returns
        -------
        psi_abs : np.ndarray, shape (n_eval, n_t)
        """
        if self._sol is None:
            raise RuntimeError("Call fit() before evaluate().")

        A_test = self.ansatz.evaluate_model(x_eval) @ self._Vt_s.T
        n = self._n_basis

        t_eval = np.atleast_1d(t_eval)
        psi_abs = np.zeros((x_eval.shape[0], len(t_eval)))

        if self._sol.sol is not None:
            for i, t in enumerate(t_eval):
                y = self._sol.sol(t)
                u_p = A_test @ y[:n]
                v_p = A_test @ y[n:]
                psi_abs[:, i] = np.sqrt(u_p**2 + v_p**2)
        else:
            for i, t in enumerate(t_eval):
                idx = np.argmin(np.abs(self._sol.t - t))
                u_p = A_test @ self._sol.y[:n, idx]
                v_p = A_test @ self._sol.y[n:, idx]
                psi_abs[:, i] = np.sqrt(u_p**2 + v_p**2)

        return psi_abs

    def evaluate_uv(self, x_eval, t_eval):
        """
        Evaluate u(x, t) and v(x, t) separately.

        Returns
        -------
        u, v : np.ndarray, each shape (n_eval, n_t)
        """
        if self._sol is None:
            raise RuntimeError("Call fit() before evaluate_uv().")

        A_test = self.ansatz.evaluate_model(x_eval) @ self._Vt_s.T
        n = self._n_basis

        t_eval = np.atleast_1d(t_eval)
        u_out = np.zeros((x_eval.shape[0], len(t_eval)))
        v_out = np.zeros((x_eval.shape[0], len(t_eval)))

        if self._sol.sol is not None:
            for i, t in enumerate(t_eval):
                y = self._sol.sol(t)
                u_out[:, i] = A_test @ y[:n]
                v_out[:, i] = A_test @ y[n:]
        else:
            for i, t in enumerate(t_eval):
                idx = np.argmin(np.abs(self._sol.t - t))
                u_out[:, i] = A_test @ self._sol.y[:n, idx]
                v_out[:, i] = A_test @ self._sol.y[n:, idx]

        return u_out, v_out

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _V_potential(self, x):
        """Harmonic trap: V(x) = 0.5·γ²·Σ x_d²"""
        return 0.5 * self.gamma**2 * np.sum(x**2, axis=1)
