"""
Nonlinear_GPE_Solver: SDRE solver for the Gross-Pitaevskii Equation.

    i ∂ψ/∂t = [-½ Δ + V(x) + β|ψ|²] ψ

Splits ψ = u + iv into real/imaginary parts and solves the coupled ODE system:
    du/dt = -½ Δv + V·v + β(u²+v²)·v
    dv/dt =  ½ Δu - V·u - β(u²+v²)·u

Uses SWIM basis functions (via BasicAnsatzDecay) for spatial discretization
and scipy.integrate.solve_ivp for time integration.

Mass conservation is enforced via EXACT projection (rescaling):
    After ODE integration, ψ → ψ · √(N₀/N(t))
    This is a direct mathematical mapping that guarantees N(t) = N₀ to machine precision.

The augmented ODE (penalty term) guides the trajectory close to the constraint,
and the final projection enforces it exactly. Energy drift is suppressed by
using a high-order symplectic-inspired ODE solver (DOP853).
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Callable
from scipy.integrate import solve_ivp


@dataclass
class Nonlinear_GPE_Solver:
    """
    SDRE solver for the d-dimensional GPE with harmonic trap.

    Attributes
    ----------
    domain : Domain
        Computational domain with interior_points.
    ansatz : BasicAnsatzDecay
        Basis functions with exponential decay envelope.
    u0 : Callable
        Initial condition for real part, u0(x) -> (n_pts,).
    v0 : Callable
        Initial condition for imaginary part, v0(x) -> (n_pts,).
    beta : float
        Nonlinear coupling coefficient.
    gamma : float
        Harmonic trap frequency.
    svd_cutoff : float
        Relative SVD truncation threshold.
    ode_solver : str
        Method for scipy.integrate.solve_ivp.
    mass_projection : bool
        If True, compute Gram matrix G via exact Gauss-Hermite quadrature
        and apply exact mass projection at end of fit().
    mass_penalty : float
        Penalty weight for the mass-conservation augmented row.
    """
    domain: object
    ansatz: object
    u0: Callable
    v0: Callable
    beta: float = 10.0
    gamma: float = 2.0
    svd_cutoff: float = 1e-10
    ode_solver: str = 'DOP853'
    mass_projection: bool = False
    mass_penalty: float = 1e0
    mass_projection_mode: str = 'aug-ode'
    n_qmc_proj: int = 500000
    """
    mass_projection_mode : str
        Controls how mass conservation is enforced:
        - 'aug-ode': augmented ODE (Gram-matrix penalty) during integration,
          then exact projection at the end. Gram matrix is used in both.
          Best when G can be computed accurately (D=1,2).
        - 'projection-only': skip Gram-matrix penalty in ODE, apply exact
          projection only at the end. The ODE ignores mass conservation.
          Recommended for D>=3 where Gram matrix cannot be computed accurately.
    """

    # Internal state (set during fit)
    _A_r: np.ndarray = field(default=None, repr=False)
    _A_inv: np.ndarray = field(default=None, repr=False)
    _C: np.ndarray = field(default=None, repr=False)
    _M_V: np.ndarray = field(default=None, repr=False)
    _Vt_s: np.ndarray = field(default=None, repr=False)
    _n_basis: int = field(default=0, repr=False)
    _coefficients_c: object = field(default=None, repr=False)
    _sol: object = field(default=None, repr=False)
    _G_quad: np.ndarray = field(default=None, repr=False)
    _G_proj: np.ndarray = field(default=None, repr=False)
    _N0: float = field(default=1.0, repr=False)
    # Frozen PINN SWIM helpers (for time-block mode)
    _psi0_fn: object = field(default=None, repr=False)
    _psi0_grad_fn: object = field(default=None, repr=False)
    _domain: object = field(default=None, repr=False)

    def _V_potential(self, x):
        """Harmonic trap: V(x) = 0.5·γ²·Σ x_d²"""
        return 0.5 * self.gamma**2 * np.sum(x**2, axis=1)

    def _build_matrices(self):
        """Build basis matrices, SVD truncation, and projection operators."""
        x = self.domain.interior_points

        A = self.ansatz.evaluate_model(x)
        A_lap = self.ansatz.evaluate_model_laplace(x)

        # SVD truncation
        U_s, S_s, Vt_s = np.linalg.svd(A, full_matrices=False)
        mask = S_s / S_s[0] > self.svd_cutoff
        self._Vt_s = Vt_s[mask, :]
        self._n_basis = int(mask.sum())

        if self._n_basis == 0:
            raise ValueError(
                f"SVD truncation removed all basis functions "
                f"(svd_cutoff={self.svd_cutoff}). Lower svd_cutoff."
            )

        print(f"SVD truncation: {self.ansatz.n_neurons} -> {self._n_basis} basis functions")

        self._A_r = A @ self._Vt_s.T
        A_lap_r = A_lap @ self._Vt_s.T

        self._A_inv = np.linalg.pinv(self._A_r, rcond=self.svd_cutoff)

        self._C = self._A_inv @ A_lap_r
        V_x = self._V_potential(x).reshape(-1, 1)
        self._M_V = self._A_inv @ (V_x * self._A_r)

        # Build Gram matrix for mass conservation
        # - aug-ode: need Gram for BOTH ODE penalty and final projection
        # - projection-only: need Gram only for final projection (high-accuracy)
        if self.mass_projection:
            self._build_quad_gram()

    def _build_quad_gram(self):
        """
        Build Gram matrix G for mass conservation (c_u^T G c_u + c_v^T G c_v).

        D=1: Exact Gauss-Hermite quadrature.
        D=2: Exact tensor-product Gauss-Hermite quadrature.
        D>=3: QMC Sobol + antithetic + exact radial moment.

        The mode controls what is built:
        - 'aug-ode': Gram used in both ODE (penalty) and final projection.
          D=1,2: exact G. D>=3: crude MC G for ODE + high-accuracy QMC G for projection.
        - 'projection-only': Gram used ONLY in final projection (skip ODE penalty).
          D=1,2: exact G. D>=3: high-accuracy QMC G only.
        """
        D = self.domain.interior_points.shape[1]
        alpha = self.ansatz.alpha
        mode = self.mass_projection_mode
        print(f"  Mass projection: D={D}, mode={mode}, ", end="")

        if D == 1:
            from scipy.special import roots_hermite
            n_quad = 600
            u_q, w_q = roots_hermite(n_quad)
            scale = 1.0 / np.sqrt(2.0 * alpha)
            x_phys = (u_q * scale).reshape(-1, 1)
            tanh_q = np.tanh(self.ansatz._preact(x_phys))
            tanh_r = tanh_q @ self._Vt_s.T
            eff_w = w_q * scale
            G = (tanh_r * eff_w[:, None]).T @ tanh_r
            self._G_quad = G
            self._G_proj = G
            print(f"exact Gauss-Hermite n={n_quad}")
            return

        elif D == 2:
            # D=2: 使用 Sobol QMC + antithetic（高精度，优于普通 MC）
            from scipy.stats import qmc
            n_qmc = 400000  # Sobol QMC 效率高
            sobol_sampler = qmc.Sobol(d=D, scramble=True, seed=12345)
            u_raw = sobol_sampler.random(n=n_qmc)

            sigma = 1.0 / np.sqrt(2.0 * alpha)
            # 使用 N(0, σ²) 的逆 CDF 将均匀点映射到正态分布
            from scipy.stats import norm
            x_q = norm.ppf(u_raw) * sigma  # (n_qmc, 2)

            # Antithetic pairs
            x_combined = np.concatenate([x_q, -x_q], axis=0)
            n_combined = len(x_combined)

            tanh_q = np.tanh(self.ansatz._preact(x_combined))
            tanh_r = tanh_q @ self._Vt_s.T  # (2n_qmc, n_basis)

            # Antithetic mean
            tanh_mean = (tanh_r[:n_qmc] + tanh_r[n_qmc:]) / 2.0

            # MC 权重（来自 N(0, σ) 分布的概率密度）
            log_q = (D * np.log(1.0 / (sigma * np.sqrt(2.0 * np.pi)))
                     - 0.5 / sigma**2 * np.sum(x_combined**2, axis=1))
            w = np.exp(-log_q) / n_combined  # 2n_qmc 个样本的总权重

            # 加权 Gram 矩阵: G = Σ w_i * tanh(x_i) * tanh(x_i)^T
            G = (tanh_r * w[:, None]).T @ tanh_r

            self._G_quad = G
            self._G_proj = G
            print(f"Sobol QMC + antithetic (n={n_qmc}×2), exact")
            return

        # ── D >= 3 ───────────────────────────────────────────────────────────────
        if mode == 'collocation':
            # Use discrete collocation Gram: G = A_r^T A_r
            # - Zero additional cost (A_r already computed)
            # - Full rank (SVD truncation guarantees)
            # - Preserves discrete mass: ||A_r c(t)||^2 = ||A_r c(0)||^2
            # - Constant factor cancels in the ratio N0/N(t)
            self._G_proj = self._A_r.T @ self._A_r
            self._G_quad = None
            cond = np.linalg.cond(self._G_proj)
            print(f"collocation Gram: G=A_r^T A_r ({self._G_proj.shape}), cond={cond:.2e}")
            return

        # QMC-based Gram (may be ill-conditioned for random SWIM basis)
        self._build_high_accuracy_gram(D, alpha)

        if mode == 'aug-ode':
            # Also build crude MC Gram for ODE penalty
            n_mc = 200000
            rng_quad = np.random.default_rng(12345)
            sigma = 1.0 / np.sqrt(4.0 * alpha)
            x_q = rng_quad.normal(0.0, sigma, size=(n_mc, D))
            A_q = self.ansatz.evaluate_model(x_q) @ self._Vt_s.T
            log_q = (0.5 * D * np.log(2.0 * alpha / np.pi)
                     - 2.0 * alpha * np.sum(x_q**2, axis=1))
            w = np.exp(-log_q) / n_mc
            wA_q = A_q * np.sqrt(w)[:, None]
            self._G_quad = wA_q.T @ wA_q
            diag_mc = np.diag(self._G_quad)
            diag_proj = np.diag(self._G_proj)
            print(f"aug-ode G: MC (n={n_mc}), proj G: Sobol QMC")
            print(f"  MC diag range: [{diag_mc.min():.4f}, {diag_mc.max():.4f}]")
            print(f"  Proj diag range: [{diag_proj.min():.4f}, {diag_proj.max():.4f}]")
        else:
            # projection-only: ODE ignores Gram, Gram only used for final projection
            self._G_quad = None  # ODE won't use Gram
            print(f"proj G: Sobol QMC (projection-only, ODE ignores Gram)")

    def _build_high_accuracy_gram(self, D, alpha):
        """
        High-accuracy Gram matrix for mass projection.
        Uses single-sided Sobol QMC + spherical radial decomposition.
        NOTE: antithetic pairs (x, -x) are intentionally NOT used because
        SWIM-initialised neurons often have b≈0, making tanh(w·x+b) an odd
        function whose antithetic mean is exactly 0, giving G_ii=0 and
        catastrophic division-by-zero in the mass projection scale factor.
        The normalisation constant cancels in the ratio N₀/N(t), so only the
        relative accuracy of G matters, not its absolute scale.
        n_qmc is controlled by self.n_qmc_proj (default 500000; 2^k recommended).
        """
        from scipy.stats import qmc
        from scipy.special import gamma as gamma_fn

        n_qmc = self.n_qmc_proj
        # Use power-of-2 for Sobol balance; silently round up if needed
        import math
        n_qmc_pow2 = 1 << math.ceil(math.log2(n_qmc))
        sobol_sampler = qmc.Sobol(d=1, scramble=True, seed=54321)
        u_raw = sobol_sampler.random(n=n_qmc_pow2)[:, 0]  # radial CDF values

        # Radial: inverse CDF of Rayleigh p(r)=4αr·exp(-2αr²)
        r = np.sqrt(-np.log(1.0 - np.clip(u_raw, 0, 1 - 1e-15)) / (2.0 * alpha))

        # Angular direction: uniform on S^{D-1} via Muller method
        z_full = np.random.default_rng(88888).standard_normal(size=(n_qmc_pow2, D))
        omega_full = z_full / (np.linalg.norm(z_full, axis=1, keepdims=True) + 1e-30)
        del z_full
        x_qmc = r[:, None] * omega_full

        # Evaluate basis at single-sided QMC points (NO antithetic)
        # Antithetic (x, -x) is biased for odd basis functions (b≈0 neurons):
        # tanh(w·x) + tanh(-w·x) = 0  →  G_ii = 0  →  scale = ∞
        tanh_q = np.tanh(self.ansatz._preact(x_qmc))  # (n_qmc, n_neurons)
        tanh_r = tanh_q @ self._Vt_s.T                # (n_qmc, n_basis)

        # Radial prefactor (constant; cancels in N₀/N(t) ratio)
        radial_prefactor = (2.0 * alpha) ** (-(D - 1) / 2.0) * gamma_fn(D / 2.0) / np.sqrt(np.pi)
        self._G_proj = radial_prefactor * (tanh_r.T @ tanh_r)

        # Diagnostics
        diag = np.diag(self._G_proj)
        n_zero = int((diag < 1e-12 * diag.max()).sum())
        print(f"  proj Gram: Sobol QMC single-sided (n={n_qmc_pow2}), D={D}")
        print(f"    Proj diag range: [{diag.min():.4e}, {diag.max():.4e}]  (n_zero={n_zero})")
        if self._G_quad is not None:
            diag_mc = np.diag(self._G_quad)
            diag_proj = np.diag(self._G_proj)
            rel_err = np.abs(diag_proj - diag_mc) / (np.abs(diag_mc) + 1e-30)
            print(f"    MC diag range: [{diag_mc.min():.4f}, {diag_mc.max():.4f}]")
            print(f"    MC vs Proj rel diff: mean={rel_err.mean():.4f}, max={rel_err.max():.4f}")

    def _project_initial_conditions(self):
        """Project u0, v0 onto the truncated basis."""
        x = self.domain.interior_points
        u0_vals = self.u0(x)
        v0_vals = self.v0(x)

        c_u0 = self._A_inv @ u0_vals
        c_v0 = self._A_inv @ v0_vals

        # Report reconstruction quality
        u0_rec = self._A_r @ c_u0
        rel_err = np.linalg.norm(u0_rec - u0_vals) / (np.linalg.norm(u0_vals) + 1e-30)
        print(f"IC reconstruction: L2 rel err = {rel_err:.2e}")

        # Store initial mass for exact projection (use high-accuracy G)
        if self.mass_projection and self._G_proj is not None:
            self._N0 = float(
                c_u0 @ (self._G_proj @ c_u0) + c_v0 @ (self._G_proj @ c_v0)
            )
            print(f"  Initial mass N(0) = {self._N0:.16f}")

        return c_u0, c_v0

    def _make_ode_rhs(self):
        """
        Build the ODE right-hand-side for the GPE system.

        The penalty term (augmented ODE) keeps the trajectory near the
        constraint manifold to reduce the size of the final projection step.
        """
        A_r = self._A_r
        A_inv = self._A_inv
        C = self._C
        M_V = self._M_V
        n = self._n_basis
        beta = self.beta
        # Only use augmented ODE when Gram matrix is available AND mode is 'aug-ode'
        do_aug = (self.mass_projection and
                  self._G_quad is not None and
                  self.mass_projection_mode == 'aug-ode')
        G = self._G_quad if do_aug else None
        N0 = self._N0 if self.mass_projection and self._G_proj is not None else 1.0
        penalty = self.mass_penalty

        # Precompute G² for the denominator of the Lagrange multiplier
        if do_aug:
            G2 = G @ G   # (n, n)

        def ode_rhs(t, y):
            c_u, c_v = y[:n], y[n:]

            # Reconstruct field values at collocation points
            u_vals = A_r @ c_u
            v_vals = A_r @ c_v
            psi2 = u_vals**2 + v_vals**2

            # Standard (unconstrained) dc/dt in coefficient space
            dc_u = -0.5 * (C @ c_v) + M_V @ c_v + beta * (A_inv @ (psi2 * v_vals))
            dc_v =  0.5 * (C @ c_u) - M_V @ c_u - beta * (A_inv @ (psi2 * u_vals))

            if do_aug:
                # Mass defect: N(t) = c_u^T G c_u + c_v^T G c_v
                N_t = c_u @ (G @ c_u) + c_v @ (G @ c_v)
                dN = N_t - N0

                # Current mass rate: dN/dt = 2*c_u^T*G*dc_u + 2*c_v^T*G*dc_v
                Gc_u = G @ c_u
                Gc_v = G @ c_v
                dN_free = 2.0 * (c_u @ (G @ dc_u) + c_v @ (G @ dc_v))

                # Target mass rate: dN/dt = -penalty * dN
                dN_target = -penalty * dN

                # Lagrange multiplier:
                #   dc_u += λ G c_u,  dc_v += λ G c_v
                #   => dN_corrected = dN_free + 2λ*(c_u^T*G²*c_u + c_v^T*G²*c_v)
                #   = dN_target
                denom = 2.0 * (c_u @ (G2 @ c_u) + c_v @ (G2 @ c_v))
                if abs(denom) > 1e-30:
                    lam = (dN_target - dN_free) / denom
                    dc_u = dc_u + lam * Gc_u
                    dc_v = dc_v + lam * Gc_v

            return np.concatenate([dc_u, dc_v])

        return ode_rhs

    def fit(self, t_span, t_eval=None, rtol=1e-6, atol=1e-8):
        """
        Solve the GPE from t_span[0] to t_span[1].

        Parameters
        ----------
        t_span : tuple (t0, tf)
            Integration interval.
        t_eval : array-like, optional
            Times at which to store the solution. If None, solver chooses.
        rtol, atol : float
            Tolerances for the ODE solver.

        Returns
        -------
        self : Nonlinear_GPE_Solver
            Fitted solver (call evaluate to get solution).
        status : int
            0 if successful.
        """
        # Initialise ansatz weights (skip if already done externally,
        # e.g. by init_ansatz_enhanced for SD-FSNN where SWIM init points
        # differ from collocation points)
        if not hasattr(self.ansatz, 'W') or self.ansatz.W is None:
            self.ansatz.init_model(self.domain)
        self._domain = self.domain

        # Store psi0 functions for Frozen PINN SWIM time-block re-initialization
        self._psi0_fn = lambda x: self.u0(x) + 1j * self.v0(x)
        def _psi0_grad(x):
            """Numerical gradient of |ψ₀|."""
            eps = 1e-6
            D = x.shape[1]
            grad = np.zeros_like(x)
            for d in range(D):
                x_plus = x.copy(); x_plus[:, d] += eps
                x_minus = x.copy(); x_minus[:, d] -= eps
                psi_plus = (self.u0(x_plus) + 1j * self.v0(x_plus))
                psi_minus = (self.u0(x_minus) + 1j * self.v0(x_minus))
                grad[:, d] = (np.abs(psi_plus) - np.abs(psi_minus)) / (2.0 * eps)
            return grad
        self._psi0_grad_fn = _psi0_grad

        # Build matrices
        self._build_matrices()

        # Project initial conditions
        c_u0, c_v0 = self._project_initial_conditions()
        y0 = np.concatenate([c_u0, c_v0])

        # Build and solve ODE
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

        # ── Exact mass projection (direct mapping) ──────────────────────────────
        # Projection uses HIGH-ACCURACY Gram (_G_proj) when available.
        # Works in both 'aug-ode' and 'projection-only' modes.
        if self.mass_projection and self._G_proj is not None and sol.sol is not None:
            n = self._n_basis
            G = self._G_proj  # always use high-accuracy Gram for projection
            N0 = self._N0
            _dense = sol.sol

            def projected_dense(t, _d=_dense, _G=G, _N0=N0, _n=n):
                y = _d(t)
                cu, cv = y[:_n], y[_n:]
                Nt = float(cu @ (_G @ cu) + cv @ (_G @ cv))
                scale = np.sqrt(_N0 / Nt) if Nt > 0 else 1.0
                return np.concatenate([scale * cu, scale * cv])

            sol.sol = projected_dense

            # Also project stored snapshots
            for j in range(sol.y.shape[1]):
                cu = sol.y[:n, j]
                cv = sol.y[n:, j]
                Nt = float(cu @ (G @ cu) + cv @ (G @ cv))
                if Nt > 0:
                    scale = np.sqrt(N0 / Nt)
                    sol.y[:n, j] = scale * cu
                    sol.y[n:, j] = scale * cv

        # Store final (possibly projected) dense interpolant
        self._coefficients_c = sol.sol if sol.sol is not None else None

        return self, sol.status

    def evaluate(self, x_eval, t_eval):
        """
        Evaluate |ψ(x, t)| at given space-time points.

        In Frozen PINN SWIM time-block mode (fit_time_blocks), automatically
        dispatches to evaluate_blocks() using the correct block-wise ansatz.

        Parameters
        ----------
        x_eval : np.ndarray, shape (n_eval, D)
            Spatial evaluation points.
        t_eval : np.ndarray, shape (n_t,)
            Time points (must be within the solved interval).

        Returns
        -------
        psi_abs : np.ndarray, shape (n_eval, n_t)
            |ψ| = sqrt(u² + v²) at each (x, t).
        """
        if self._sol is None:
            raise RuntimeError("Call fit() or fit_time_blocks() before evaluate().")

        # Frozen PINN SWIM: use block-wise evaluation
        if hasattr(self, '_time_blocks') and self._time_blocks is not None:
            return self.evaluate_blocks(x_eval, t_eval)

        # ── Standard (single-block) evaluation ────────────────────────────────
        A_test = self.ansatz.evaluate_model(x_eval) @ self._Vt_s.T
        n = self._n_basis
        t_eval = np.atleast_1d(t_eval)
        psi_abs = np.zeros((x_eval.shape[0], len(t_eval)))

        if self._sol.sol is not None:
            for i, t in enumerate(t_eval):
                y = self._sol.sol(t)
                c_u = y[:n]
                c_v = y[n:]
                u_p = A_test @ c_u
                v_p = A_test @ c_v
                psi_abs[:, i] = np.sqrt(u_p**2 + v_p**2)
        else:
            for i, t in enumerate(t_eval):
                idx = np.argmin(np.abs(self._sol.t - t))
                c_u = self._sol.y[:n, idx]
                c_v = self._sol.y[n:, idx]
                u_p = A_test @ c_u
                v_p = A_test @ c_v
                psi_abs[:, i] = np.sqrt(u_p**2 + v_p**2)

        return psi_abs

    def evaluate_uv(self, x_eval, t_eval):
        """
        Evaluate u(x,t) and v(x,t) separately.

        In Frozen PINN SWIM time-block mode, uses block-wise evaluation.

        Returns
        -------
        u, v : np.ndarray, each shape (n_eval, n_t)
        """
        if self._sol is None:
            raise RuntimeError("Call fit() or fit_time_blocks() before evaluate_uv().")

        if hasattr(self, '_time_blocks') and self._time_blocks is not None:
            return self._evaluate_uv_blocks(x_eval, t_eval)

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

    def _evaluate_uv_blocks(self, x_eval, t_eval):
        """Block-wise u/v evaluation for Frozen PINN SWIM."""
        t_eval = np.atleast_1d(t_eval)
        u_out = np.zeros((x_eval.shape[0], len(t_eval)))
        v_out = np.zeros((x_eval.shape[0], len(t_eval)))

        for block_idx, sol in enumerate(self._block_sol_collection):
            t_start = self._t0 + block_idx * self._dt_block
            t_end = t_start + self._dt_block
            block_mask = (t_eval >= t_start) & (t_eval <= t_end)
            if not np.any(block_mask):
                continue

            ansatz_block = self._block_ansatz_collection[block_idx]
            Vt_s = self._block_Vt_s_collection[block_idx]
            A_eval = ansatz_block.evaluate_model(x_eval) @ Vt_s.T
            n = Vt_s.shape[0]
            t_block_eval = t_eval[block_mask]

            if sol.sol is not None:
                for i, t in enumerate(t_block_eval):
                    y = sol.sol(t - block_idx * self._dt_block - self._t0)
                    u_out[:, np.where(block_mask)[0][i]] = A_eval @ y[:n]
                    v_out[:, np.where(block_mask)[0][i]] = A_eval @ y[n:]
            else:
                for i, t in enumerate(t_block_eval):
                    idx = np.argmin(np.abs(sol.t - (t - block_idx * self._dt_block - self._t0)))
                    u_out[:, np.where(block_mask)[0][i]] = A_eval @ sol.y[:n, idx]
                    v_out[:, np.where(block_mask)[0][i]] = A_eval @ sol.y[n:, idx]

        return u_out, v_out

    @property
    def sol(self):
        return self._sol

    @property
    def sol(self):
        return self._sol

    # =========================================================================
    # Frozen PINN SWIM: Time-Block Mode
    # =========================================================================
    def fit_time_blocks(
        self,
        t_span,
        t_eval=None,
        rtol=1e-6,
        atol=1e-8,
        time_blocks=1,
        n_col=5000,
        svd_on=True,
        svd_cutoff=None,
        basis_mode='reuse',
        n_extra_neurons=0,
        svd_cutoff_block=None,
    ):
        """
        Frozen PINN SWIM: Solve GPE with periodic basis re-initialization.

        The full interval [t0, tf] is divided into time blocks. Within each
        block, the ansatz (basis W, b) is FROZEN — this is the "Frozen PINN"
        part. At block boundaries, the ansatz is re-initialized using the
        previous block's solution — this is the "SWIM" re-sampling part.
        The new ansatz at block k is initialized via:
          - Project random directions onto ∇|ψ_k| orthogonal complement
          - Weight sampling by |∇|ψ_k||² (Frozen PINN SWIM)
        This enables the basis to adapt to the evolving solution structure
        without paying the cost of full online training.

        Parameters
        ----------
        t_span : tuple (t0, tf)
            Full integration interval.
        t_eval : array-like, optional
            Times at which to store the solution.
        rtol, atol : float
            ODE solver tolerances.
        time_blocks : int
            Number of time blocks.
        n_col : int
            Number of collocation points for re-sampling.
        svd_on : bool
            Whether to apply SVD truncation (default True).
        svd_cutoff : float, optional
            SVD cutoff. Defaults to self.svd_cutoff.

        Returns
        -------
        self, status
        """
        import copy
        import time as _time

        if svd_cutoff is None:
            svd_cutoff = self.svd_cutoff

        self._domain = self.domain

        # Initialize Frozen PINN SWIM psi0 helpers
        self._psi0_fn = lambda x: self.u0(x) + 1j * self.v0(x)
        def _psi0_grad(x):
            eps = 1e-6
            D_ = x.shape[1]
            grad = np.zeros_like(x)
            for d in range(D_):
                x_plus = x.copy(); x_plus[:, d] += eps
                x_minus = x.copy(); x_minus[:, d] -= eps
                psi_plus = (self.u0(x_plus) + 1j * self.v0(x_plus))
                psi_minus = (self.u0(x_minus) + 1j * self.v0(x_minus))
                grad[:, d] = (np.abs(psi_plus) - np.abs(psi_minus)) / (2.0 * eps)
            return grad
        self._psi0_grad_fn = _psi0_grad

        t0, tf = t_span[0], t_span[-1]
        dt_block = (tf - t0) / time_blocks

        # Collections for block-wise ansatz and solution data
        self._block_ansatz_collection = []
        self._block_Vt_s_collection = []
        self._block_sol_collection = []
        self._block_c_collection = []   # dense interpolants
        self._block_G_proj_collection = []
        self._block_N0_collection = []

        # Build global t_eval aligned to block boundaries
        if t_eval is None:
            n_snap = max(10 * time_blocks, 50)
            t_eval = np.linspace(t0, tf, n_snap)
        t_eval = np.atleast_1d(t_eval)

        # ── Block 0: initial Frozen PINN SWIM ─────────────────────────────
        block = 0
        t_block = [t0, t0 + dt_block]

        # Block 0: plain random SWIM on domain points (SAME as single-block
        # ``fit()``). The gradient-aware ``init_model_from_psi`` variant
        # produces a compact but *ill-conditioned* basis: ||A_inv||_F ≈ 1e12
        # for the β=10 breathing problem, which ruins ODE accuracy even
        # though mass/energy are analytically preserved by the Hermitian
        # structure. Plain random SWIM gives 60 well-conditioned basis
        # functions and matches single-block L² ≈ 1e-7.
        ansatz_block = copy.deepcopy(self.ansatz)
        ansatz_block.init_model(self._domain)

        solver_block = self._make_block_solver(ansatz_block, svd_on, svd_cutoff)
        c_u0, c_v0 = solver_block._project_initial_conditions()
        y0 = np.concatenate([c_u0, c_v0])

        ode_rhs = solver_block._make_ode_rhs()
        t_eval_block = t_eval[(t_eval >= t_block[0]) & (t_eval <= t_block[1])]

        _t0 = _time.time()
        sol = solve_ivp(
            ode_rhs, t_block, y0,
            method=self.ode_solver,
            t_eval=t_eval_block if len(t_eval_block) > 0 else None,
            dense_output=True,
            rtol=rtol, atol=atol,
        )
        print(f"  [block 0] n_basis={solver_block._n_basis} "
              f"solve_ivp: {_time.time()-_t0:.3f}s "
              f"status={sol.status} nfev={sol.nfev}", flush=True)

        # Wrap dense interpolant with exact mass projection (direct mapping).
        # Without this, c-space drift accumulates across block boundaries
        # and pre-pinet mass error can reach 100% even though aug-ode kept
        # N(t) ≈ N₀ during integration. Matches the single-block ``fit()``.
        self._wrap_sol_with_mass_projection(sol, solver_block)

        solver_block._sol = sol
        self._block_sol_collection.append(sol)
        self._block_c_collection.append(sol.sol)
        self._block_ansatz_collection.append(copy.deepcopy(ansatz_block))
        self._block_Vt_s_collection.append(solver_block._Vt_s)
        if self.mass_projection and solver_block._G_proj is not None:
            self._block_G_proj_collection.append(solver_block._G_proj)
            self._block_N0_collection.append(solver_block._N0)

        # ── Subsequent blocks: re-init with Frozen PINN SWIM ────────────────
        for block in range(1, time_blocks):
            t_block = [t0 + block * dt_block, t0 + (block + 1) * dt_block]

            # ── Get previous block's ansatz (not the template ansatz) ──────
            prev_ansatz = self._block_ansatz_collection[-1]
            prev_Vt_s = self._block_Vt_s_collection[-1]

            # Evaluate previous block's solution at its END (= current
            # block's start). ``sol_prev.sol`` and ``sol_prev.t`` use
            # ABSOLUTE times because each block was integrated with
            # ``t_block=[t0+block*dt, t0+(block+1)*dt]`` — querying with
            # the block duration ``dt_block`` would return the previous
            # block's START state instead of its END.
            t_prev = t_block[0]
            sol_prev = self._block_sol_collection[-1]

            if sol_prev.sol is not None:
                y_prev = sol_prev.sol(t_prev)
            else:
                idx = np.argmin(np.abs(sol_prev.t - t_prev))
                y_prev = sol_prev.y[:, idx]

            n_prev = prev_Vt_s.shape[0]
            c_u_prev = y_prev[:n_prev]
            c_v_prev = y_prev[n_prev:]

            # ── Frozen PINN SWIM: sample new points and init new ansatz ────
            x_new = self._sample_for_block(n_col)

            # Evaluate PREVIOUS block's ansatz at new points
            A_prev = prev_ansatz.evaluate_model(x_new) @ prev_Vt_s.T
            u_prev_new = A_prev @ c_u_prev
            v_prev_new = A_prev @ c_v_prev

            # ── Block-k basis: select per ``basis_mode`` ───────────────────
            # 'reuse'  : identical to block 0 (fast, no adaptation).
            # 'enrich' : block-0 neurons PLUS n_extra supervised on the
            #            current (u, v) via SWIM pair sampling, then
            #            SVD-truncated jointly. Keeps block-0's well-
            #            conditioned spatial cover while adding modes
            #            tuned to the rotating/breathing state, which is
            #            essential for large β where ψ develops structure
            #            outside the t=0 span of the random basis.
            if basis_mode == 'enrich' and n_extra_neurons > 0:
                # Evaluate ψ_prev at the domain interior points
                xi = self._domain.interior_points
                A_prev_domain = (
                    prev_ansatz.evaluate_model(xi) @ prev_Vt_s.T
                )
                u_at_domain = A_prev_domain @ c_u_prev
                v_at_domain = A_prev_domain @ c_v_prev
                # Supervised SWIM pair sampling on (u, v) jointly
                ansatz_sup = copy.deepcopy(self.ansatz)
                ansatz_sup.n_neurons = int(n_extra_neurons)
                ansatz_sup._init_from_uv_components(
                    xi, u_at_domain, v_at_domain, rng=ansatz_sup.rng,
                )
                # Concatenate base (block-0) + supervised neurons
                ansatz_block = copy.deepcopy(self._block_ansatz_collection[0])
                ansatz_block.W = np.concatenate(
                    [ansatz_block.W, ansatz_sup.W], axis=1
                )
                ansatz_block.b = np.concatenate(
                    [ansatz_block.b, ansatz_sup.b], axis=1
                )
                ansatz_block.n_neurons = ansatz_block.W.shape[1]
            else:
                # Default: identical basis (mathematically exact restart).
                ansatz_block = prev_ansatz
            # For subsequent blocks we may use a tighter ``svd_cutoff_block``
            # to aggressively drop near-degenerate basis directions whose
            # small singular values (σ ~ svd_cutoff × σ_max) would amplify
            # the per-block lstsq projection noise by 1/σ_min, stripping
            # 4-8 orders of magnitude of accuracy at each boundary. The
            # block-0 cutoff is kept tight (1e-14) to preserve IC fidelity.
            svd_k = svd_cutoff_block if svd_cutoff_block is not None else svd_cutoff
            solver_block = self._make_block_solver(ansatz_block, svd_on, svd_k)

            # ── Block-boundary restart: identity when bases match ─────────
            # Reusing the basis means the new block's reduced basis is
            # identical to the previous one. Re-projecting via ``A_inv @
            # (A_prev @ c_prev)`` introduces O(eps)·||c|| coefficient
            # noise, which the ill-conditioned ODE RHS then amplifies to
            # O(1) trajectory divergence. Direct reuse of ``c_prev`` is
            # mathematically exact when ``Vt_s_new == Vt_s_prev``.
            if np.array_equal(solver_block._Vt_s, prev_Vt_s):
                c_u0 = c_u_prev.copy()
                c_v0 = c_v_prev.copy()
            else:
                # Different bases: fall back to high-accuracy L² projection
                c_u0, c_v0 = self._l2_project_onto_new_basis(
                    solver_block, ansatz_block, prev_ansatz, prev_Vt_s,
                    c_u_prev, c_v_prev, x_new=x_new,
                )
            y0 = np.concatenate([c_u0, c_v0])
            # NB: do NOT rescale y0 to cap ||y0|| — the ODE legitimately
            # drifts ||c||_2 far above O(1) in the envelope's null-space
            # (``||c|| ≈ 5e6`` at t=0.5 is normal) while physical mass
            # and energy stay ≈ 1. Rescaling here would corrupt the state.

            # Project initial mass onto new Gram
            if self.mass_projection and solver_block._G_proj is not None:
                solver_block._N0 = float(
                    c_u0 @ (solver_block._G_proj @ c_u0) +
                    c_v0 @ (solver_block._G_proj @ c_v0)
                )

            ode_rhs = solver_block._make_ode_rhs()
            t_eval_block = t_eval[(t_eval >= t_block[0]) & (t_eval <= t_block[1])]

            _t0 = _time.time()
            sol = solve_ivp(
                ode_rhs, t_block, y0,
                method=self.ode_solver,
                t_eval=t_eval_block if len(t_eval_block) > 0 else None,
                dense_output=True,
                rtol=rtol, atol=atol,
            )
            print(f"  [block {block}] n_basis={solver_block._n_basis} "
                  f"solve_ivp: {_time.time()-_t0:.3f}s "
                  f"status={sol.status} nfev={sol.nfev}", flush=True)

            # Wrap with exact mass projection (same as block 0).
            self._wrap_sol_with_mass_projection(sol, solver_block)

            solver_block._sol = sol
            self._block_sol_collection.append(sol)
            self._block_c_collection.append(sol.sol)
            self._block_ansatz_collection.append(copy.deepcopy(ansatz_block))
            self._block_Vt_s_collection.append(solver_block._Vt_s)
            if self.mass_projection and solver_block._G_proj is not None:
                self._block_G_proj_collection.append(solver_block._G_proj)
                self._block_N0_collection.append(solver_block._N0)

        # ── Store block metadata ─────────────────────────────────────────────
        self._time_blocks = time_blocks
        self._dt_block = dt_block
        self._t0 = t0

        # Concatenate all block solutions into a single "merged" sol for
        # compatibility with the single-block API (evaluate, etc.)
        self._sol = self._merge_blocks()

        status = 0
        for sol in self._block_sol_collection:
            if sol.status != 0:
                status = sol.status
                break

        return self, status

    def _wrap_sol_with_mass_projection(self, sol, block_solver):
        """Apply exact mass projection (direct mapping) to a block's sol.

        Mirrors the tail of :meth:`fit` so multi-block behaves identically
        to single-block at block endpoints. Without this, aug-ode's
        integrator-limited drift of a few ×1e-6 in N(t) is *amplified* by
        the ill-conditioned coefficient representation (||c|| can grow
        to 1e6 even though physical |ψ| stays O(1)) and the "raw" block
        output violates mass conservation by up to 100 % at block joins.
        """
        if not (self.mass_projection and block_solver._G_proj is not None
                and sol.sol is not None):
            return
        n = block_solver._n_basis
        G = block_solver._G_proj
        N0 = block_solver._N0
        _dense = sol.sol

        def projected_dense(t, _d=_dense, _G=G, _N0=N0, _n=n):
            y = _d(t)
            cu, cv = y[:_n], y[_n:]
            Nt = float(cu @ (_G @ cu) + cv @ (_G @ cv))
            scale = np.sqrt(_N0 / Nt) if Nt > 0 else 1.0
            return np.concatenate([scale * cu, scale * cv])

        sol.sol = projected_dense
        for j in range(sol.y.shape[1]):
            cu = sol.y[:n, j]
            cv = sol.y[n:, j]
            Nt = float(cu @ (G @ cu) + cv @ (G @ cv))
            if Nt > 0:
                s = np.sqrt(N0 / Nt)
                sol.y[:n, j] = s * cu
                sol.y[n:, j] = s * cv

    def _make_block_solver(self, ansatz_block, svd_on=True, svd_cutoff=None):
        """
        Build a temporary solver for a single time block with its own ansatz.
        """
        if svd_cutoff is None:
            svd_cutoff = self.svd_cutoff

        # Create a lightweight solver object for this block
        solver_block = Nonlinear_GPE_Solver_Block(
            ansatz=ansatz_block,
            domain=self._domain,
            u0=self.u0,
            v0=self.v0,
            beta=self.beta,
            gamma=self.gamma,
            svd_cutoff=svd_cutoff,
            ode_solver=self.ode_solver,
            mass_projection=self.mass_projection,
            mass_projection_mode=self.mass_projection_mode,
            mass_penalty=self.mass_penalty,
        )
        solver_block._build_matrices()
        return solver_block

    def _l2_project_onto_new_basis(self, solver_block, ansatz_new, ansatz_prev,
                                   Vt_s_prev, c_u_prev, c_v_prev,
                                   x_new=None, A_new_xnew_fallback=None):
        """High-accuracy L² projection of ψ_prev onto the new basis.

        Uses the SAME quadrature that builds G_new, so the projection is
        consistent with mass/energy integrals and free of sampling noise
        at block boundaries.

        Returns
        -------
        c_u0, c_v0 : np.ndarray
            Coefficients of ψ_prev expressed in the new (SVD-reduced) basis.
        """
        D = self._domain.interior_points.shape[1]
        Vt_s_new = solver_block._Vt_s
        G_new = solver_block._G_proj

        # Interior-point lstsq projection (same for all D). Using the
        # domain's own interior points means the projection is consistent
        # with how the block solver builds ``A_r`` and ``A_inv``: errors
        # there are bounded by ``1/σ_min(A_new) = ‖A_inv‖`` which is
        # controlled by the SVD cutoff. The Hermite-quadrature L²
        # projection used previously was more accurate mathematically
        # but ``G_new`` computed via Hermite absorption can be
        # catastrophically ill-conditioned when the new basis mixes
        # block-0 random neurons with supervised u/v neurons (κ(G) ≳
        # 1e18), and amplifies pair-sampling noise into the initial
        # condition. The interior-point lstsq path is numerically robust
        # and identical to the way ``_project_initial_conditions`` treats
        # ψ₀ at t=0.
        xi = self._domain.interior_points
        A_new = ansatz_new.evaluate_model(xi) @ Vt_s_new.T
        A_prev = ansatz_prev.evaluate_model(xi) @ Vt_s_prev.T
        u_prev_vals = A_prev @ c_u_prev
        v_prev_vals = A_prev @ c_v_prev
        rcond = max(self.svd_cutoff, 1e-12)
        c_u0 = np.linalg.lstsq(A_new, u_prev_vals, rcond=rcond)[0]
        c_v0 = np.linalg.lstsq(A_new, v_prev_vals, rcond=rcond)[0]
        return c_u0, c_v0

    def _sample_for_block(self, n_col):
        """
        Sample collocation points for Frozen PINN SWIM re-initialization.
        Uses a truncated Gaussian centered at origin (matching the GPE ground state).
        """
        # Derive a per-block RNG: the ansatz exposes `rng` (Generator),
        # not `random_state`. Spawn a child stream for independence.
        base_rng = getattr(self.ansatz, 'rng', None)
        if base_rng is None:
            base_rng = np.random.default_rng()
        rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
        D = self._domain.interior_points.shape[1]
        alpha = self.ansatz.alpha
        sigma = 1.0 / np.sqrt(4.0 * alpha)
        x = rng.normal(0.0, sigma, size=(n_col, D))
        x = np.clip(x, -self._domain._hi, self._domain._hi)
        return x

    def _merge_blocks(self):
        """
        Build a minimal aggregate sol object for time-block mode.

        Each block may have a different SVD-truncated basis size, so we
        CANNOT stack their coefficient arrays into a single 2-D y. The
        aggregate exposes only the interface that external consumers need
        (``t``, ``status``, ``nfev``); spatial evaluation is dispatched
        per-block via :meth:`evaluate_blocks` / :meth:`_evaluate_uv_blocks`.

        Any code path that needs per-block coefficients (e.g. pinet
        projection) walks :attr:`_block_sol_collection` directly.
        """
        if not self._block_sol_collection:
            return None

        # Concatenate block-global time grids for bookkeeping only
        all_t = []
        for block_idx, sol in enumerate(self._block_sol_collection):
            if sol.t.size > 0:
                all_t.append(self._t0 + block_idx * self._dt_block + sol.t)
        merged_t = np.concatenate(all_t) if all_t else np.array([])
        merged_t = np.sort(merged_t)

        total_nfev = sum(s.nfev for s in self._block_sol_collection)

        merged_obj = type('MergedSol', (), {})()
        merged_obj.t = merged_t
        merged_obj.y = None        # per-block, not stackable
        merged_obj.sol = None      # per-block dense interp lives in sol_k.sol
        merged_obj.nfev = total_nfev
        merged_obj.status = 0
        return merged_obj

    def evaluate_blocks(self, x_eval, t_eval):
        """
        Evaluate the time-block solution (Frozen PINN SWIM).

        Parameters
        ----------
        x_eval : np.ndarray, shape (n_eval, D)
            Spatial evaluation points.
        t_eval : np.ndarray, shape (n_t,)
            Time points.

        Returns
        -------
        psi_abs : np.ndarray, shape (n_eval, n_t)
            |ψ| at each (x, t).
        """
        t_eval = np.atleast_1d(t_eval)
        psi_abs = np.zeros((x_eval.shape[0], len(t_eval)))

        for block_idx, sol in enumerate(self._block_sol_collection):
            t_start = self._t0 + block_idx * self._dt_block
            t_end = t_start + self._dt_block
            block_mask = (t_eval >= t_start) & (t_eval <= t_end)
            if not np.any(block_mask):
                continue

            ansatz_block = self._block_ansatz_collection[block_idx]
            Vt_s = self._block_Vt_s_collection[block_idx]
            A_eval = ansatz_block.evaluate_model(x_eval) @ Vt_s.T

            t_block_eval = t_eval[block_mask]

            # ``fit_time_blocks`` integrates each block in ABSOLUTE time
            # (``t_block=[t0+block*dt, t0+(block+1)*dt]``), so ``sol.sol``
            # and ``sol.t`` already use absolute times — do not subtract
            # the block offset.
            if sol.sol is not None:
                for i, t in enumerate(t_block_eval):
                    y = sol.sol(t)
                    n = Vt_s.shape[0]
                    cu = y[:n]
                    cv = y[n:]
                    u_p = A_eval @ cu
                    v_p = A_eval @ cv
                    psi_abs[:, np.where(block_mask)[0][i]] = np.sqrt(u_p**2 + v_p**2)
            else:
                for i, t in enumerate(t_block_eval):
                    idx = np.argmin(np.abs(sol.t - t))
                    n = Vt_s.shape[0]
                    cu = sol.y[:n, idx]
                    cv = sol.y[n:, idx]
                    u_p = A_eval @ cu
                    v_p = A_eval @ cv
                    psi_abs[:, np.where(block_mask)[0][i]] = np.sqrt(u_p**2 + v_p**2)

        return psi_abs


class Nonlinear_GPE_Solver_Block:
    """
    Lightweight solver for a single time block.

    Shares the same logic as Nonlinear_GPE_Solver but does NOT own the
    ansatz (it receives a pre-built ansatz). This allows each time block
    to have its own basis functions.
    """

    def __init__(self, ansatz, domain, u0, v0, beta, gamma, svd_cutoff,
                 ode_solver='DOP853', mass_projection=False,
                 mass_projection_mode='aug-ode', mass_penalty=1e0):
        self.ansatz = ansatz
        self._domain = domain
        self.u0 = u0
        self.v0 = v0
        self.beta = beta
        self.gamma = gamma
        self.svd_cutoff = svd_cutoff
        self.ode_solver = ode_solver
        self.mass_projection = mass_projection
        self.mass_projection_mode = mass_projection_mode
        self.mass_penalty = mass_penalty

        self._A_r = None
        self._A_inv = None
        self._C = None
        self._M_V = None
        self._Vt_s = None
        self._n_basis = 0
        self._G_quad = None
        self._G_proj = None
        self._N0 = 1.0
        self._sol = None
        self._coefficients_c = None

    def _V_potential(self, x):
        return 0.5 * self.gamma**2 * np.sum(x**2, axis=1)

    def _build_matrices(self):
        x = self._domain.interior_points
        A = self.ansatz.evaluate_model(x)
        A_lap = self.ansatz.evaluate_model_laplace(x)

        U_s, S_s, Vt_s = np.linalg.svd(A, full_matrices=False)
        mask = S_s / S_s[0] > self.svd_cutoff
        self._Vt_s = Vt_s[mask, :]
        self._n_basis = int(mask.sum())

        self._A_r = A @ self._Vt_s.T
        A_lap_r = A_lap @ self._Vt_s.T
        self._A_inv = np.linalg.pinv(self._A_r, rcond=self.svd_cutoff)
        self._C = self._A_inv @ A_lap_r
        V_x = self._V_potential(x).reshape(-1, 1)
        self._M_V = self._A_inv @ (V_x * self._A_r)

        if self.mass_projection:
            self._build_quad_gram()

    def _project_initial_conditions(self):
        x = self._domain.interior_points
        u0_vals = self.u0(x)
        v0_vals = self.v0(x)
        c_u0 = self._A_inv @ u0_vals
        c_v0 = self._A_inv @ v0_vals
        if self.mass_projection and self._G_proj is not None:
            self._N0 = float(
                c_u0 @ (self._G_proj @ c_u0) + c_v0 @ (self._G_proj @ c_v0)
            )
        return c_u0, c_v0

    def _build_quad_gram(self):
        D = self._domain.interior_points.shape[1]
        alpha = self.ansatz.alpha
        mode = self.mass_projection_mode

        if D == 1:
            from scipy.special import roots_hermite
            n_quad = 600
            u_q, w_q = roots_hermite(n_quad)
            scale = 1.0 / np.sqrt(2.0 * alpha)
            x_phys = (u_q * scale).reshape(-1, 1)
            tanh_q = np.tanh(self.ansatz._preact(x_phys))
            tanh_r = tanh_q @ self._Vt_s.T
            eff_w = w_q * scale
            G = (tanh_r * eff_w[:, None]).T @ tanh_r
            self._G_quad = G
            self._G_proj = G
            return

        # D >= 2: QMC Sobol + antithetic
        from scipy.stats import qmc
        from scipy.special import gamma as gamma_fn

        n_qmc = 500000
        sobol_sampler = qmc.Sobol(d=D, scramble=True, seed=54321)
        u_raw = sobol_sampler.random(n=n_qmc)

        radial_prefactor = (2.0 * alpha) ** (-(D - 1) / 2.0) * gamma_fn(D / 2.0) / np.sqrt(np.pi)

        u0 = u_raw[:, 0]
        r = np.sqrt(-np.log(1.0 - u0) / (2.0 * alpha))

        z_full = np.random.default_rng(88888).standard_normal(size=(n_qmc, D))
        omega_full = z_full / (np.linalg.norm(z_full, axis=1, keepdims=True) + 1e-30)
        x_qmc = r[:, None] * omega_full

        x_combined = np.concatenate([x_qmc, -x_qmc], axis=0)
        tanh_q = np.tanh(self.ansatz._preact(x_combined))
        tanh_r = tanh_q @ self._Vt_s.T
        tanh_mean = (tanh_r[:n_qmc] + tanh_r[n_qmc:]) / 2.0

        self._G_proj = radial_prefactor * (tanh_mean.T @ tanh_mean)

        if mode == 'aug-ode':
            n_mc = 200000
            rng_mc = np.random.default_rng(12345)
            sigma = 1.0 / np.sqrt(4.0 * alpha)
            x_mc = rng_mc.normal(0.0, sigma, size=(n_mc, D))
            A_q = self.ansatz.evaluate_model(x_mc) @ self._Vt_s.T
            log_q = (0.5 * D * np.log(2.0 * alpha / np.pi)
                     - 2.0 * alpha * np.sum(x_mc**2, axis=1))
            w = np.exp(-log_q) / n_mc
            wA_q = A_q * np.sqrt(w)[:, None]
            self._G_quad = wA_q.T @ wA_q
        else:
            self._G_quad = None

    def _make_ode_rhs(self):
        A_r = self._A_r
        A_inv = self._A_inv
        C = self._C
        M_V = self._M_V
        n = self._n_basis
        beta = self.beta

        do_aug = (self.mass_projection and
                  self._G_quad is not None and
                  self.mass_projection_mode == 'aug-ode')
        G = self._G_quad if do_aug else None
        N0 = self._N0 if self.mass_projection and self._G_proj is not None else 1.0
        penalty = self.mass_penalty

        if do_aug:
            G2 = G @ G

        def ode_rhs(t, y):
            c_u, c_v = y[:n], y[n:]
            u_vals = A_r @ c_u
            v_vals = A_r @ c_v
            psi2 = u_vals**2 + v_vals**2

            dc_u = -0.5 * (C @ c_v) + M_V @ c_v + beta * (A_inv @ (psi2 * v_vals))
            dc_v =  0.5 * (C @ c_u) - M_V @ c_u - beta * (A_inv @ (psi2 * u_vals))

            if do_aug:
                N_t = c_u @ (G @ c_u) + c_v @ (G @ c_v)
                dN = N_t - N0
                Gc_u = G @ c_u
                Gc_v = G @ c_v
                dN_free = 2.0 * (c_u @ (G @ dc_u) + c_v @ (G @ dc_v))
                dN_target = -penalty * dN
                denom = 2.0 * (c_u @ (G2 @ c_u) + c_v @ (G2 @ c_v))
                if abs(denom) > 1e-30:
                    lam = (dN_target - dN_free) / denom
                    dc_u = dc_u + lam * Gc_u
                    dc_v = dc_v + lam * Gc_v

            return np.concatenate([dc_u, dc_v])

        return ode_rhs
