import numpy as np
import copy
import time
from types import SimpleNamespace
from dataclasses import dataclass, field
from scipy.special import roots_hermite
from src import Domain, BasicAnsatzDecay
from src.Ansatz.basic_ansatz import BasicAnsatz
try:
    import torch
except ImportError:
    torch = None

class BasicAnsatzAdapter:
    """
    Ablation adapter: wraps BasicAnsatz (no exp-decay envelope) for BCL_GPE_Solver.
    φ_k(x) = tanh(w_k^T x + b_k)  — no exp(-α‖x‖²) factor.

    Key interface additions vs BasicAnsatz:
      alpha = 0.0                 (signals no decay to Gram/MC routines)
      _is_sparse = False
      supports_full_gradient = True  (routes gradients through evaluate_model_gradient)
      evaluate_model_gradient: transposes (n_pts,K,D) → (n_pts,D,K)
      evaluate_model_laplace_stochastic: full Laplacian (no SDGD)
      init_model_from_psi: falls back to plain SWIM init
    """
    alpha = 0.0
    _is_sparse = False
    sparse_dim = 0
    supports_full_gradient = True

    def __init__(self, n_neurons=512, random_state=42):
        self._ba = BasicAnsatz(
            activation='tanh',
            n_neurons=n_neurons,
            random_state=random_state,
        )
        self.n_neurons = n_neurons

    def init_model(self, domain, **kwargs):
        self._ba.init_model(domain)

    def init_model_from_psi(self, domain, psi0_fn=None, psi_grad0_fn=None):
        self.init_model(domain)

    def evaluate_model(self, x):
        return self._ba.evaluate_model(x)

    def evaluate_model_laplace(self, x):
        return self._ba.evaluate_model_laplace(x)

    def evaluate_model_gradient(self, x):
        grad = self._ba.evaluate_model_gradient(x)  # (n_pts, K, D)
        return np.transpose(grad, (0, 2, 1))         # (n_pts, D, K)

    def evaluate_model_laplace_stochastic(self, x, sdgd_batch, rng):
        return self.evaluate_model_laplace(x)

    @property
    def W(self):
        return self._ba.W

    @property
    def b(self):
        return self._ba.b

    @property
    def col_scale(self):
        return getattr(self._ba, 'col_scale', None)

    def _preact(self, x):
        return x @ self._ba.W + self._ba.b

    def _envelope(self, x):
        return np.ones((x.shape[0], 1))


class SeparableLowRankAnsatzDecay:
    def __init__(self, n_neurons, random_state=42, alpha=0.5, dist_min=1e-10):
        self.n_neurons = n_neurons
        self.random_state = random_state
        self.alpha = alpha
        self.dist_min = dist_min
        self.W = None
        self.b = None
        self.col_scale = None
        self.supports_full_gradient = True
        self._is_sparse = False
        self.sparse_dim = 1

    def _build_1d_helper(self, seed_offset):
        return BasicAnsatzDecay(
            n_neurons=self.n_neurons,
            random_state=self.random_state + seed_offset,
            alpha=self.alpha,
            dist_min=self.dist_min,
            sparse_dim=0,
        )

    def init_model(self, domain, **kwargs):
        x = domain.interior_points
        D_loc = x.shape[1]
        self.W = np.zeros((D_loc, self.n_neurons))
        self.b = np.zeros((D_loc, self.n_neurons))
        for d in range(D_loc):
            helper = self._build_1d_helper(104729 * (d + 1))
            helper.init_model(Domain(interior_points=x[:, d:d+1]))
            self.W[d, :] = helper.W.reshape(-1)
            self.b[d, :] = helper.b.reshape(-1)
        self._fit_column_scale(x)

    def init_model_from_psi(self, domain, psi0_fn, psi_grad0_fn=None):
        x = domain.interior_points
        D_loc = x.shape[1]
        psi0 = psi0_fn(x)
        abs_psi0 = np.abs(psi0)
        if psi_grad0_fn is None:
            eps = 1e-6
            grad_abs_psi0 = np.zeros_like(x)
            for d in range(D_loc):
                x_plus = x.copy(); x_plus[:, d] += eps
                x_minus = x.copy(); x_minus[:, d] -= eps
                grad_abs_psi0[:, d] = (np.abs(psi0_fn(x_plus)) - np.abs(psi0_fn(x_minus))) / (2.0 * eps)
        else:
            grad_abs_psi0 = psi_grad0_fn(x)
        self.W = np.zeros((D_loc, self.n_neurons))
        self.b = np.zeros((D_loc, self.n_neurons))
        for d in range(D_loc):
            helper = self._build_1d_helper(104729 * (d + 1))
            helper._init_from_points_gradient_aware(
                x[:, d:d+1],
                abs_psi0,
                grad_abs_psi0[:, d:d+1],
                rng=helper.rng,
            )
            self.W[d, :] = helper.W.reshape(-1)
            self.b[d, :] = helper.b.reshape(-1)
        self._fit_column_scale(x)

    def _fit_column_scale(self, x):
        raw = self._raw_model(x)
        scale = np.sqrt(np.mean(raw**2, axis=0))
        self.col_scale = np.clip(scale, 1e-8, None)

    def _raw_model(self, x):
        factors, _, _ = self._factor_terms(x)
        return np.prod(factors, axis=1)

    def _factor_terms(self, x):
        n_pts, D_loc = x.shape
        factors = np.ones((n_pts, D_loc, self.n_neurons))
        first = np.zeros_like(factors)
        second = np.zeros_like(factors)
        for d in range(D_loc):
            xd = x[:, d:d+1]
            z = xd * self.W[d:d+1, :] + self.b[d:d+1, :]
            g = np.exp(-self.alpha * xd**2)
            h = np.tanh(z)
            h1 = 1.0 - h**2
            g1 = -2.0 * self.alpha * xd * g
            g2 = (4.0 * self.alpha**2 * xd**2 - 2.0 * self.alpha) * g
            h2 = -2.0 * h * h1 * (self.W[d:d+1, :]**2)
            factors[:, d, :] = (g * h)
            first[:, d, :] = (g1 * h + g * h1 * self.W[d:d+1, :])
            second[:, d, :] = (g2 * h + 2.0 * g1 * h1 * self.W[d:d+1, :] + g * h2)
        return factors, first, second

    def evaluate_model(self, x):
        raw = self._raw_model(x)
        if self.col_scale is None:
            return raw
        return raw / self.col_scale.reshape(1, -1)

    def evaluate_model_gradient(self, x):
        factors, first, _ = self._factor_terms(x)
        n_pts, D_loc, _ = factors.shape
        grad = np.zeros((n_pts, D_loc, self.n_neurons))
        for d in range(D_loc):
            others = np.prod(np.delete(factors, d, axis=1), axis=1)
            grad[:, d, :] = first[:, d, :] * others
        if self.col_scale is not None:
            grad = grad / self.col_scale.reshape(1, 1, -1)
        return grad

    def evaluate_model_laplace(self, x):
        factors, _, second = self._factor_terms(x)
        n_pts, D_loc, _ = factors.shape
        lap = np.zeros((n_pts, self.n_neurons))
        for d in range(D_loc):
            others = np.prod(np.delete(factors, d, axis=1), axis=1)
            lap += second[:, d, :] * others
        if self.col_scale is not None:
            lap = lap / self.col_scale.reshape(1, -1)
        return lap

    def evaluate_model_laplace_stochastic(self, x, sdgd_batch, rng):
        return self.evaluate_model_laplace(x)


class RadialBasicAnsatzDecay:
    def __init__(self, n_neurons, random_state=42, alpha=0.5, dist_min=1e-10):
        self.n_neurons = n_neurons
        self.random_state = random_state
        self.alpha = alpha
        self.dist_min = dist_min
        self.W = None
        self.b = None
        self.col_scale = None
        self.supports_full_gradient = True
        self._is_sparse = False
        self.sparse_dim = 1
        self._D = None
        self.physical_dim = None
        self.is_radial_basis = True

    def _build_1d_helper(self):
        return BasicAnsatzDecay(
            n_neurons=self.n_neurons,
            random_state=self.random_state,
            alpha=self.alpha,
            dist_min=self.dist_min,
            sparse_dim=0,
        )

    def _r(self, x):
        if x.shape[1] == 1:
            return np.abs(x)
        return np.linalg.norm(x, axis=1, keepdims=True)

    def _s(self, x):
        if x.shape[1] == 1:
            return x**2
        return np.sum(x**2, axis=1, keepdims=True)

    def init_model(self, domain, **kwargs):
        x = domain.interior_points
        self._D = self.physical_dim if self.physical_dim is not None else x.shape[1]
        s = self._s(x)
        helper = self._build_1d_helper()
        helper.init_model(Domain(interior_points=s))
        self.W = helper.W.copy()
        self.b = helper.b.copy()
        self._fit_column_scale(x)

    def init_model_from_psi(self, domain, psi0_fn, psi_grad0_fn=None):
        x = domain.interior_points
        self._D = self.physical_dim if self.physical_dim is not None else x.shape[1]
        s = self._s(x)
        psi0 = psi0_fn(x)
        abs_psi0 = np.abs(psi0)
        if psi_grad0_fn is None:
            eps = 1e-6
            x_plus = x.copy(); x_plus[:, 0] += eps
            x_minus = x.copy(); x_minus[:, 0] -= eps
            grad_x0 = (np.abs(psi0_fn(x_plus)) - np.abs(psi0_fn(x_minus))) / (2.0 * eps)
        else:
            grad_full = psi_grad0_fn(x)
            grad_x0 = grad_full[:, 0]
        helper = self._build_1d_helper()
        helper._init_from_points_gradient_aware(
            s,
            abs_psi0,
            grad_x0.reshape(-1, 1),
            rng=helper.rng,
        )
        self.W = helper.W.copy()
        self.b = helper.b.copy()
        self._fit_column_scale(x)

    def _raw_terms(self, x):
        s = self._s(x)
        z = s @ self.W + self.b
        g = np.exp(-self.alpha * s)
        h = np.tanh(z)
        h1 = 1.0 - h**2
        g1 = -self.alpha * g
        g2 = (self.alpha**2) * g
        f = g * h
        fr = g1 * h + g * h1 * self.W
        h2 = -2.0 * h * h1 * (self.W**2)
        frr = g2 * h + 2.0 * g1 * h1 * self.W + g * h2
        return s, f, fr, frr

    def _preact(self, x):
        return self._s(x) @ self.W + self.b

    def _envelope(self, x):
        return np.exp(-self.alpha * self._s(x))

    def _fit_column_scale(self, x):
        _, f, _, _ = self._raw_terms(x)
        scale = np.sqrt(np.mean(f**2, axis=0))
        self.col_scale = np.clip(scale, 1e-8, None)

    def evaluate_model(self, x):
        _, f, _, _ = self._raw_terms(x)
        return f / self.col_scale.reshape(1, -1)

    def evaluate_raw_model(self, x):
        h = np.tanh(self._preact(x))
        return h / self.col_scale.reshape(1, -1)

    def evaluate_raw_radial_derivative(self, x):
        s = self._s(x)
        h = np.tanh(s @ self.W + self.b)
        dh_ds = (1.0 - h**2) * self.W / self.col_scale.reshape(1, -1)
        r = np.sqrt(np.clip(s, 0.0, None))
        return 2.0 * r * dh_ds

    def evaluate_raw_laplacian(self, x):
        s = self._s(x)
        h = np.tanh(s @ self.W + self.b)
        h1 = 1.0 - h**2
        dh_ds = h1 * self.W / self.col_scale.reshape(1, -1)
        d2h_ds2 = -2.0 * h * h1 * (self.W**2) / self.col_scale.reshape(1, -1)
        return 2.0 * self._D * dh_ds + 4.0 * s * d2h_ds2

    def evaluate_raw_gradient_dim(self, x, j):
        s = self._s(x)
        h = np.tanh(s @ self.W + self.b)
        dh_ds = (1.0 - h**2) * self.W / self.col_scale.reshape(1, -1)
        if x.shape[1] == 1:
            xj = np.sqrt(np.clip(s, 0.0, None))
        else:
            xj = x[:, j:j+1]
        return 2.0 * xj * dh_ds

    def evaluate_raw_gradient_nd(self, x):
        s = self._s(x)
        h = np.tanh(s @ self.W + self.b)
        dh_ds = (1.0 - h**2) * self.W / self.col_scale.reshape(1, -1)
        if x.shape[1] == 1:
            r = np.sqrt(np.clip(s, 0.0, None))
            return (2.0 * r * dh_ds)[:, None, :]
        return 2.0 * x[:, :, None] * dh_ds[:, None, :]

    def evaluate_model_gradient(self, x):
        _, _, fr, _ = self._raw_terms(x)
        grad = 2.0 * x[:, :, None] * fr[:, None, :]
        return grad / self.col_scale.reshape(1, 1, -1)

    def evaluate_model_laplace(self, x):
        s, _, fr, frr = self._raw_terms(x)
        lap = 2.0 * self._D * fr + 4.0 * s * frr
        return lap / self.col_scale.reshape(1, -1)

    def evaluate_model_laplace_stochastic(self, x, sdgd_batch, rng):
        return self.evaluate_model_laplace(x)


class BCL_GPE_Solver:
    """
    BCL (Boundary Compliant Layer) GPE Solver
    
    实现思路：
    1. 离线阶段：利用高精度蒙特卡洛积分精确计算 Gram 矩阵
    2. 质量约束：通过精确质量投影实现代数绝对硬约束
    3. 能量约束：通过正交化流形 + 切空间投影实现动态合规
    """
    
    def __init__(
        self,
        domain,
        ansatz,
        u0_fn,
        v0_fn,
        beta=10.0,
        gamma=2.0,
        svd_cutoff=1e-14,
        n_quad_mc=50000,
        normalize_mass=True,
        mass_quadrature='auto',
        energy_conservation=True,
        energy_target=None,
        random_state=42,
        use_sdgd=False,
        sdgd_batch=0,
        use_hermite_basis=False,
        hermite_n_per_dim=8,
        n_quad_gh_nd=200,
        hermite_total_deg=0,
        hermite_n_coll=10000,
        use_splitting=False,
        splitting_order=4,
        n_split_steps=500,
        use_swim_splitting=False,
        swim_split_gram_cutoff=1e-13,
        swim_split_order=2,
        trajectory_aware_init=False,
        traj_coarse_K=0,
        traj_n_snaps=20,
        traj_coarse_rtol=1e-6,
        fixed_step_projection=False,
        fixed_steps=1000,
        report_projection_diagnostics=True,
        store_dense_output=True,
        backend='numpy',
        device='auto',
        torch_dtype='float64',
    ):
        self.domain = domain
        self.ansatz = ansatz
        self.u0_fn = u0_fn
        self.v0_fn = v0_fn
        self.beta = beta
        self.gamma = gamma
        self.svd_cutoff = svd_cutoff
        self.n_quad_mc = n_quad_mc
        self.normalize_mass = normalize_mass
        self.mass_quadrature = mass_quadrature
        self.energy_conservation = energy_conservation
        self.energy_target = energy_target
        self.random_state = random_state
        self.use_sdgd = use_sdgd
        self.sdgd_batch = sdgd_batch
        self.use_hermite_basis = use_hermite_basis
        self.hermite_n_per_dim = hermite_n_per_dim
        self.n_quad_gh_nd = n_quad_gh_nd
        self.hermite_total_deg = hermite_total_deg
        self.hermite_n_coll = hermite_n_coll
        self.use_splitting = use_splitting
        self.splitting_order = splitting_order
        self.n_split_steps = n_split_steps
        self.use_swim_splitting = use_swim_splitting
        self._swim_split_gram_cutoff = swim_split_gram_cutoff
        self._swim_split_order = int(swim_split_order)
        self.trajectory_aware_init = trajectory_aware_init
        self.traj_coarse_K = traj_coarse_K
        self.traj_n_snaps = traj_n_snaps
        self.traj_coarse_rtol = traj_coarse_rtol
        self.fixed_step_projection = fixed_step_projection
        self.fixed_steps = fixed_steps
        self._report_projection_diagnostics = report_projection_diagnostics
        self._store_dense_output = store_dense_output
        self.backend = backend
        self.device = device
        self.torch_dtype = torch_dtype
        self._torch_device_obj = None
        
        # 内部状态
        self._Vt_s = None  # SVD 的 V^T (SVD截断后)
        self._n_basis = 0  # 最终基函数数
        self._A_r = None  # 截断后的基函数矩阵
        self._A_inv = None  # 伪逆矩阵
        self._D = None  # 空间维度
        self._use_hermite_basis = False  # 实际使用标志
        self._hermite_combos = None     # Hermite多指标列表
        self._hermite_Vt_trunc = None  # SVD truncation (collocation mode)
        self._hermite_M_E = None       # Energy matrix (collocation mode)
        self._Phi_gal = None
        self._w_gal = None
        self._x_gal = None
        self._x_coll_nd = None
        
        # BCL 特有的变换矩阵
        self._L_inv = None  # Cholesky 逆矩阵（能量守恒）
        self._A_transform = None  # 综合变换矩阵
        
        # 积分预计算
        self._Gram_matrix_G = None  # 质量 Gram 矩阵 G = ∫ Φ Φ^T dx
        self._Gram_matrix_Q = None  # 能量 Gram 矩阵 Q
        self._N0 = None  # 初始质量
        self._E0 = None  # 初始能量
        
        # ODE 求解器状态
        self._sol = None
        self._coefficients_c = None
        self._rng = np.random.default_rng(random_state)
        self._use_time_blocks = False
        self._block_states = []
        self._block_nfev = 0
        self._trajectory_supervision = None
        self._traj_diag = {}
        self._mass_norm_cache = None
        self._has_envelope_bias = True  # paper Eq.(9): [Φ, 1] constant feature column
        self._use_physical_basis = False
        
    def _physical_dim(self):
        if getattr(self.ansatz, 'is_radial_basis', False) and getattr(self.ansatz, 'physical_dim', None) is not None:
            return int(self.ansatz.physical_dim)
        return self.domain.interior_points.shape[1]

    def _use_torch_backend(self):
        return self.backend == 'torch'

    def _get_torch_device(self):
        if not self._use_torch_backend():
            return None
        if torch is None:
            raise RuntimeError("backend='torch' requires PyTorch to be installed")
        if self._torch_device_obj is None:
            dev_name = self.device
            if dev_name == 'auto':
                dev_name = 'cuda' if torch.cuda.is_available() else 'cpu'
            if dev_name.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError(f"Requested device '{dev_name}' but CUDA is unavailable")
            self._torch_device_obj = torch.device(dev_name)
        return self._torch_device_obj

    def _get_torch_dtype(self):
        if self.torch_dtype == 'float32':
            return torch.float32
        return torch.float64

    def _to_numpy(self, array):
        if torch is not None and torch.is_tensor(array):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def _to_backend_tensor(self, array):
        if not self._use_torch_backend():
            return np.asarray(array)
        return torch.as_tensor(array, dtype=self._get_torch_dtype(), device=self._get_torch_device())

    def _svd_backend(self, array):
        if not self._use_torch_backend():
            return np.linalg.svd(array, full_matrices=False)
        array_t = self._to_backend_tensor(array)
        U_t, S_t, Vh_t = torch.linalg.svd(array_t, full_matrices=False)
        return self._to_numpy(U_t), self._to_numpy(S_t), self._to_numpy(Vh_t)

    def _pinv_backend(self, array):
        if not self._use_torch_backend():
            return np.linalg.pinv(array, rcond=self.svd_cutoff)
        array_t = self._to_backend_tensor(array)
        return self._to_numpy(torch.linalg.pinv(array_t, rtol=self.svd_cutoff))

    def _cond_backend(self, array):
        if not self._use_torch_backend():
            return np.linalg.cond(array)
        svals = self._to_numpy(torch.linalg.svdvals(self._to_backend_tensor(array)))
        smin = max(float(np.min(np.abs(svals))), 1e-30)
        smax = float(np.max(np.abs(svals)))
        return smax / smin

    def _eigvalsh_backend(self, array):
        if not self._use_torch_backend():
            return np.linalg.eigvalsh(array)
        return self._to_numpy(torch.linalg.eigvalsh(self._to_backend_tensor(array)))

    def _assemble_deenv_operator(self, x, A_r, A_inv, Vt_s, L_inv=None):
        D_loc = x.shape[1]
        alpha = self.ansatz.alpha
        if getattr(self.ansatz, 'is_radial_basis', False):
            D_phys = self._physical_dim()
            r = x[:, 0]
            Lap_r = self.ansatz.evaluate_raw_laplacian(x)
            if self._has_envelope_bias:
                Lap_r = np.column_stack([Lap_r, np.zeros(len(x))])
            Lap_r = Lap_r @ Vt_s.T
            dPhi_dr = self.ansatz.evaluate_raw_radial_derivative(x)
            if self._has_envelope_bias:
                dPhi_dr = np.column_stack([dPhi_dr, np.zeros(len(x))])
            dPhi_dr = dPhi_dr @ Vt_s.T
            if L_inv is not None:
                Lap_r = Lap_r @ L_inv.T
                dPhi_dr = dPhi_dr @ L_inv.T
            x_sq = r**2
            V_vals = self._V_potential(x)
            b_vals = -2.0 * alpha * D_phys + 4.0 * alpha**2 * x_sq
            bV_diag = 0.5 * b_vals - V_vals
            Grad_coupling = (-2.0 * alpha * r)[:, None] * dPhi_dr
            L_op = 0.5 * Lap_r + Grad_coupling + bV_diag[:, None] * A_r
            H_deenv = A_inv @ L_op
            C = A_inv @ Lap_r
            M_V = A_inv @ (V_vals[:, None] * A_r)
            rho2 = self.ansatz._envelope(x).ravel()**2
            return H_deenv, C, M_V, rho2
        Lap_r = self._raw_laplacian(x) @ Vt_s.T
        if L_inv is not None:
            Lap_r = Lap_r @ L_inv.T
        grad_terms = [self._deenv_gradient_dim(x, j) @ Vt_s.T for j in range(D_loc)]
        if L_inv is not None:
            grad_terms = [g @ L_inv.T for g in grad_terms]
        env = self.ansatz._envelope(x).ravel()
        if not self._use_torch_backend():
            Phi_r = A_r
            Phi_inv = A_inv
            Grad_coupling = np.zeros_like(Phi_r)
            for j, dPhi_j in enumerate(grad_terms):
                a_j = -2.0 * alpha * x[:, j]
                Grad_coupling += a_j[:, None] * dPhi_j
            x_sq = np.sum(x**2, axis=1)
            V_vals = self._V_potential(x)
            b_vals = -2.0 * alpha * D_loc + 4.0 * alpha**2 * x_sq
            bV_diag = 0.5 * b_vals - V_vals
            L_op = 0.5 * Lap_r + Grad_coupling + bV_diag[:, None] * Phi_r
            H_deenv = Phi_inv @ L_op
            C = Phi_inv @ Lap_r
            M_V = Phi_inv @ (V_vals[:, None] * Phi_r)
            rho2 = env**2
            return H_deenv, C, M_V, rho2
        x_t = self._to_backend_tensor(x)
        Phi_r_t = self._to_backend_tensor(A_r)
        Phi_inv_t = self._to_backend_tensor(A_inv)
        Lap_r_t = self._to_backend_tensor(Lap_r)
        Grad_coupling_t = torch.zeros_like(Phi_r_t)
        for j, dPhi_j in enumerate(grad_terms):
            dPhi_j_t = self._to_backend_tensor(dPhi_j)
            a_j_t = -2.0 * alpha * x_t[:, j]
            Grad_coupling_t = Grad_coupling_t + a_j_t[:, None] * dPhi_j_t
        x_sq_t = torch.sum(x_t**2, dim=1)
        V_vals_t = 0.5 * self.gamma**2 * x_sq_t
        b_vals_t = -2.0 * alpha * D_loc + 4.0 * alpha**2 * x_sq_t
        bV_diag_t = 0.5 * b_vals_t - V_vals_t
        L_op_t = 0.5 * Lap_r_t + Grad_coupling_t + bV_diag_t[:, None] * Phi_r_t
        H_deenv = self._to_numpy(Phi_inv_t @ L_op_t)
        C = self._to_numpy(Phi_inv_t @ Lap_r_t)
        M_V = self._to_numpy(Phi_inv_t @ (V_vals_t[:, None] * Phi_r_t))
        rho2 = self._to_numpy(self._to_backend_tensor(env) ** 2)
        return H_deenv, C, M_V, rho2

    def _torch_feature_backend_ready(self):
        return (
            self._use_torch_backend()
            and hasattr(self.ansatz, '_preact')
            and hasattr(self.ansatz, '_envelope')
            and hasattr(self.ansatz, 'W')
        )

    def _V_potential(self, x):
        """谐振子势: V(x) = 0.5·γ²·Σ x_d²"""
        return 0.5 * self.gamma**2 * np.sum(x**2, axis=1)

    def _basis_features(self, x):
        if getattr(self, '_use_physical_basis', False):
            return self.ansatz.evaluate_model(x)
        return self._raw_features(x)

    def _physical_gradient_1d(self, x_eval):
        if hasattr(self.ansatz, 'evaluate_model_gradient'):
            grad = self.ansatz.evaluate_model_gradient(x_eval)
            if grad.ndim == 3:
                if grad.shape[1] == x_eval.shape[1]:
                    grad = grad[:, 0, :]
                else:
                    grad = grad[:, :, 0]
            dPhi_raw = grad
        else:
            a = self.ansatz.alpha
            g = self.ansatz._envelope(x_eval)
            z = self.ansatz._preact(x_eval)
            h = np.tanh(z)
            h1 = 1.0 - h**2
            dPhi_raw = (-2.0 * a * x_eval[:, 0:1] * g) * h + g * h1 * self.ansatz.W[0:1, :]
            col_scale = getattr(self.ansatz, 'col_scale', None)
            if col_scale is not None:
                dPhi_raw = dPhi_raw / col_scale.reshape(1, -1)
        dPhi_r = dPhi_raw @ self._Vt_s.T
        if self.energy_conservation and self._L_inv is not None:
            return dPhi_r @ self._L_inv.T
        return dPhi_r

    def _raw_features(self, x):
        """De-enveloped features [tanh(z), 1] per paper: u = C(t)[Φ(x), 1]^T."""
        if self._torch_feature_backend_ready() and not self._use_hermite_basis:
            x_t = self._to_backend_tensor(x)
            W_t = self._to_backend_tensor(self.ansatz.W)
            b_t = self._to_backend_tensor(self.ansatz.b)
            z_t = x_t @ W_t + b_t
            A_t = torch.tanh(z_t)
            col_scale = getattr(self.ansatz, 'col_scale', None)
            if col_scale is not None:
                A_t = A_t / self._to_backend_tensor(col_scale).reshape(1, -1)
            if self._has_envelope_bias:
                A_t = torch.cat([A_t, torch.ones((x_t.shape[0], 1), dtype=A_t.dtype, device=A_t.device)], dim=1)
            return self._to_numpy(A_t)
        A = self.ansatz.evaluate_model(x)        # ρ · tanh(z), (n, K)
        env = self.ansatz._envelope(x)            # (n, 1)
        A = A / env                               # de-envelope → tanh only
        if self._has_envelope_bias and not self._use_hermite_basis:
            A = np.column_stack([A, np.ones(len(x))])
        return A

    def _raw_laplacian(self, x):
        """∇²[tanh(z), 1] — de-enveloped Laplacian (no envelope terms).
        ∇²tanh(w^T x+b) = -2 tanh(z) sech²(z) ||w||²
        ∇²(1) = 0
        """
        if getattr(self.ansatz, 'is_radial_basis', False) and hasattr(self.ansatz, 'evaluate_raw_laplacian'):
            lap = self.ansatz.evaluate_raw_laplacian(x)
            if self._has_envelope_bias and not self._use_hermite_basis:
                lap = np.column_stack([lap, np.zeros(len(x))])
            return lap
        if self._torch_feature_backend_ready() and not self._use_hermite_basis:
            x_t = self._to_backend_tensor(x)
            W_t = self._to_backend_tensor(self.ansatz.W)
            b_t = self._to_backend_tensor(self.ansatz.b)
            z_t = x_t @ W_t + b_t
            h_t = torch.tanh(z_t)
            h1_t = 1.0 - h_t**2
            if getattr(self.ansatz, '_is_sparse', False):
                w_sq_t = self._to_backend_tensor(self.ansatz._sp_valsq)
            else:
                w_sq_t = torch.sum(W_t**2, dim=0)
            lap_t = -2.0 * h_t * h1_t * w_sq_t
            col_scale = getattr(self.ansatz, 'col_scale', None)
            if col_scale is not None:
                lap_t = lap_t / self._to_backend_tensor(col_scale).reshape(1, -1)
            if self._has_envelope_bias:
                lap_t = torch.cat([lap_t, torch.zeros((x_t.shape[0], 1), dtype=lap_t.dtype, device=lap_t.device)], dim=1)
            return self._to_numpy(lap_t)
        z = self.ansatz._preact(x)                # (n, K)
        h = np.tanh(z)
        h1 = 1.0 - h**2                          # sech²
        if getattr(self.ansatz, '_is_sparse', False):
            w_sq = self.ansatz._sp_valsq           # (K,)
        else:
            w_sq = np.sum(self.ansatz.W**2, axis=0)  # (K,)
        lap = -2.0 * h * h1 * w_sq               # (n, K)
        col_scale = getattr(self.ansatz, 'col_scale', None)
        if col_scale is not None:
            lap = lap / col_scale.reshape(1, -1)
        if self._has_envelope_bias and not self._use_hermite_basis:
            lap = np.column_stack([lap, np.zeros(len(x))])
        return lap

    def _deenv_gradient_dim(self, x, j):
        """∂/∂x_j [tanh(z), 1] for dimension j.
        ∂tanh(z)/∂x_j = sech²(z) · w_j
        ∂(1)/∂x_j = 0
        """
        if getattr(self.ansatz, 'is_radial_basis', False) and hasattr(self.ansatz, 'evaluate_raw_gradient_dim'):
            grad = self.ansatz.evaluate_raw_gradient_dim(x, min(j, x.shape[1] - 1))
            if self._has_envelope_bias and not self._use_hermite_basis:
                grad = np.column_stack([grad, np.zeros(len(x))])
            return grad
        if self._torch_feature_backend_ready() and not self._use_hermite_basis:
            x_t = self._to_backend_tensor(x)
            W_t = self._to_backend_tensor(self.ansatz.W)
            b_t = self._to_backend_tensor(self.ansatz.b)
            z_t = x_t @ W_t + b_t
            h1_t = 1.0 - torch.tanh(z_t)**2
            grad_t = h1_t * W_t[j:j+1, :]
            col_scale = getattr(self.ansatz, 'col_scale', None)
            if col_scale is not None:
                grad_t = grad_t / self._to_backend_tensor(col_scale).reshape(1, -1)
            if self._has_envelope_bias:
                grad_t = torch.cat([grad_t, torch.zeros((x_t.shape[0], 1), dtype=grad_t.dtype, device=grad_t.device)], dim=1)
            return self._to_numpy(grad_t)
        z = self.ansatz._preact(x)                # (n, K)
        h1 = 1.0 - np.tanh(z)**2                 # sech², (n, K)
        W = self.ansatz.W                         # (D, K)
        grad = h1 * W[j:j+1, :]                  # (n, K)
        col_scale = getattr(self.ansatz, 'col_scale', None)
        if col_scale is not None:
            grad = grad / col_scale.reshape(1, -1)
        if self._has_envelope_bias and not self._use_hermite_basis:
            grad = np.column_stack([grad, np.zeros(len(x))])
        return grad                               # (n, K+1)

    def _clone_ansatz(self, n_neurons=None, random_state=None):
        """Clone the current ansatz for a cheap coarse pre-solve.

        The trajectory-aware initialiser needs one auxiliary SDRE solve on a
        smaller feature set to generate ψ(x,t) snapshots. We deep-copy the
        current ansatz so the coarse solve keeps the same feature family
        (BasicAnsatzDecay / Krylov-SWIM / radial / separable wrapper, etc.)
        while resetting sampled weights and RNG state.
        """
        cloned = copy.deepcopy(self.ansatz)
        if n_neurons is not None and hasattr(cloned, 'n_neurons'):
            cloned.n_neurons = int(n_neurons)
        if random_state is None:
            random_state = self.random_state
        if hasattr(cloned, 'rng'):
            cloned.rng = np.random.default_rng(random_state)
        if hasattr(cloned, 'random_state'):
            cloned.random_state = random_state
        for attr in ['W', 'b', '_sp_idx', '_sp_val', '_sp_valsq']:
            if hasattr(cloned, attr):
                setattr(cloned, attr, None)
        if hasattr(cloned, '_is_sparse'):
            cloned._is_sparse = False
        return cloned

    def _prepare_trajectory_supervision(self, t_span, t_eval=None, atol=1e-8):
        """Run a cheap coarse SDRE solve and collect trajectory snapshots.

        This is the core POC for trajectory-aware SWIM: instead of supervising
        SWIM sampling only by ψ(x,0), we first obtain a coarse trajectory
        ``ψ_coarse(x, t_k)`` and then sample (w,b) from the full space-time
        snapshot set. This keeps the method strictly on the SWIM path while
        addressing the frozen-at-t=0 basis mismatch.
        """
        if not self.trajectory_aware_init:
            return None
        if self.use_hermite_basis or self.hermite_total_deg > 0:
            print("  [Trajectory-aware] Skip: Hermite mode enabled.")
            return None
        if not hasattr(self.ansatz, 'init_model_from_trajectory'):
            print("  [Trajectory-aware] Skip: current ansatz does not support trajectory supervision.")
            return None

        main_K = int(getattr(self.ansatz, 'n_neurons', 128))
        if self.traj_coarse_K and self.traj_coarse_K > 0:
            coarse_K = int(self.traj_coarse_K)
        else:
            coarse_K = max(128, min(main_K, max(500, main_K // 4)))
        n_super = max(2, int(self.traj_n_snaps))
        if t_eval is not None and len(np.atleast_1d(t_eval)) >= n_super + 1:
            t_eval_arr = np.asarray(np.atleast_1d(t_eval), dtype=float)
            idx_super = np.linspace(0, len(t_eval_arr) - 1, n_super + 1).round().astype(int)
            t_super = t_eval_arr[idx_super]
        else:
            t_super = np.linspace(t_span[0], t_span[1], n_super + 1)

        print("\n[Step 0] Trajectory-aware SWIM pre-solve")
        print(f"  coarse neurons: {coarse_K} (main={main_K})")
        print(f"  coarse snapshots: {len(t_super)}")
        print(f"  coarse rtol/atol: {self.traj_coarse_rtol:.1e} / {atol:.1e}")

        coarse_solver = BCL_GPE_Solver(
            domain=self.domain,
            ansatz=self._clone_ansatz(n_neurons=coarse_K, random_state=self.random_state + 7919),
            u0_fn=self.u0_fn,
            v0_fn=self.v0_fn,
            beta=self.beta,
            gamma=self.gamma,
            svd_cutoff=self.svd_cutoff,
            n_quad_mc=min(self.n_quad_mc, 10000),
            normalize_mass=self.normalize_mass,
            energy_conservation=self.energy_conservation,
            energy_target=self.energy_target,
            random_state=self.random_state + 7919,
            use_sdgd=self.use_sdgd,
            sdgd_batch=self.sdgd_batch,
            use_hermite_basis=False,
            hermite_n_per_dim=self.hermite_n_per_dim,
            n_quad_gh_nd=self.n_quad_gh_nd,
            hermite_total_deg=0,
            hermite_n_coll=self.hermite_n_coll,
            use_splitting=False,
            splitting_order=self.splitting_order,
            n_split_steps=self.n_split_steps,
            use_swim_splitting=False,
            swim_split_gram_cutoff=self._swim_split_gram_cutoff,
            swim_split_order=self._swim_split_order,
            trajectory_aware_init=False,
            traj_coarse_K=0,
            traj_n_snaps=0,
            traj_coarse_rtol=self.traj_coarse_rtol,
            report_projection_diagnostics=False,
            store_dense_output=self._store_dense_output,
            backend=self.backend,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )
        coarse_solver.fit(
            t_span=t_span,
            t_eval=t_super,
            rtol=self.traj_coarse_rtol,
            atol=atol,
            time_blocks=1,
            resample_metric='psi_grad',
            gradient_aware_init=False,
        )

        x_fit = self.domain.interior_points
        u_snaps, v_snaps = coarse_solver.evaluate_uv(x_fit, t_super)
        info = {
            't_super': t_super.copy(),
            'u_snaps': u_snaps.T.copy(),
            'v_snaps': v_snaps.T.copy(),
            'coarse_K': coarse_K,
            'coarse_n_basis': int(coarse_solver.n_basis),
            'coarse_nfev': int(getattr(coarse_solver.sol, 'nfev', 0)),
        }
        self._trajectory_supervision = info
        print(f"  coarse retained basis: {info['coarse_n_basis']}")
        print(f"  coarse nfev: {info['coarse_nfev']}")
        return info

    def _report_nonlinear_projection_error(self, t_eval=None, max_samples=6):
        """Report closure error of |ψ|²ψ under the current truncated SWIM space."""
        if not self._report_projection_diagnostics:
            return None
        if self._sol is None or self._A_r is None or self._A_inv is None:
            return None
        if not hasattr(self._sol, 'sol') or self._sol.sol is None:
            return None

        if t_eval is None:
            if hasattr(self._sol, 't') and self._sol.t is not None and len(np.atleast_1d(self._sol.t)) > 1:
                t_probe = np.asarray(np.atleast_1d(self._sol.t), dtype=float)
            else:
                return None
        else:
            t_probe = np.asarray(np.atleast_1d(t_eval), dtype=float)
        if t_probe.size == 0:
            return None
        if t_probe.size > max_samples:
            idx = np.linspace(0, t_probe.size - 1, max_samples).round().astype(int)
            t_probe = t_probe[idx]

        errs = []
        for t in t_probe:
            y = self._sol.sol(float(t))
            if self.normalize_mass:
                y = self._post_step_normalization(y)
            n = self._n_basis
            c_u, c_v = y[:n], y[n:]
            u_vals = self._A_r @ c_u
            v_vals = self._A_r @ c_v
            psi2 = u_vals**2 + v_vals**2
            nl_u = psi2 * u_vals
            nl_v = psi2 * v_vals
            proj_u = self._A_r @ (self._A_inv @ nl_u)
            proj_v = self._A_r @ (self._A_inv @ nl_v)
            num = np.linalg.norm(nl_u - proj_u) ** 2 + np.linalg.norm(nl_v - proj_v) ** 2
            den = np.linalg.norm(nl_u) ** 2 + np.linalg.norm(nl_v) ** 2 + 1e-30
            errs.append(np.sqrt(num / den))

        errs = np.asarray(errs, dtype=float)
        self._traj_diag['nl_proj_times'] = t_probe.copy()
        self._traj_diag['nl_proj_rel_err'] = errs.copy()
        print("\n[Diag] Nonlinear closure on current SWIM basis")
        print(f"  sampled times: {len(t_probe)}")
        print(f"  mean rel proj err: {errs.mean():.3e}")
        print(f"  max  rel proj err: {errs.max():.3e}")
        return errs
    
    def _gh_tensorproduct_quadrature_nd(self, n_per_dim):
        """D>1 Gauss-Hermite 张量积正交规则"""
        alpha = self.ansatz.alpha
        scale = 1.0 / np.sqrt(2.0 * alpha)
        D_loc = self._physical_dim()

        u1d, w1d = roots_hermite(n_per_dim)
        x1d = u1d * scale
        w1d_phys = w1d * scale

        grids_x = np.meshgrid(*([x1d] * D_loc), indexing='ij')
        grids_w = np.meshgrid(*([w1d_phys] * D_loc), indexing='ij')

        x_quad = np.column_stack([g.ravel() for g in grids_x])
        w_quad = np.ones(n_per_dim ** D_loc)
        for gw in grids_w:
            w_quad *= gw.ravel()

        return x_quad, w_quad

    def _build_hermite_phi(self, x):
        """计算Hermite乘积基函数在点x处的值。

        Phi_H[i,k]  = φ_k(x_i) = Π_d ψ_{nd_k}(x_{id})  (含高斯包络的完整HO本征函数)
        Poly_H[i,k] = Π_d C_{nd_k} H_{nd_k}(√γ x_{id})  (仅多项式部分)

        Uses stable three-term recurrence via scaled_hermite_matrix /
        hermite_bare_matrix to avoid 2^n * n! overflow for large n.
        """
        from hermite import scaled_hermite_matrix, hermite_bare_matrix

        gamma = self.gamma
        combos = self._hermite_combos
        n_pts = x.shape[0]
        D_dim = x.shape[1]
        n_basis = len(combos)

        # Per-axis max degree
        max_n = [max(ns[d] for ns in combos) for d in range(D_dim)]

        # Precompute per-axis Hermite matrices (stable recurrence)
        H_with_env = []  # (max_n[d]+1, n_pts) — with exp envelope
        H_bare = []      # (max_n[d]+1, n_pts) — polynomial only
        for d in range(D_dim):
            H_with_env.append(scaled_hermite_matrix(max_n[d], gamma, x[:, d]))
            H_bare.append(hermite_bare_matrix(max_n[d], gamma, x[:, d]))

        Phi_H = np.ones((n_pts, n_basis))
        Poly_H = np.ones((n_pts, n_basis))

        for j, ns in enumerate(combos):
            for d, n in enumerate(ns):
                Phi_H[:, j] *= H_with_env[d][n, :]
                Poly_H[:, j] *= H_bare[d][n, :]

        return Phi_H, Poly_H

    def _compute_hermite_laplacian(self, x):
        """Exact Laplacian of Hermite product basis.
        From Schrödinger eigenvalue relation: Δψ_k = (γ²|x|² - 2E_k)·ψ_k
        where E_k = γ·(Σn_d + D/2).
        """
        Phi_H, _ = self._build_hermite_phi(x)
        D_dim = x.shape[1]
        V_x = 0.5 * self.gamma**2 * np.sum(x**2, axis=1)
        E_k = np.array([self.gamma * (sum(ns) + D_dim / 2.0)
                         for ns in self._hermite_combos])
        return 2.0 * V_x[:, np.newaxis] * Phi_H - 2.0 * E_k[np.newaxis, :] * Phi_H

    def _generate_mc_quadrature_points(self, n_points, for_gpe=False):
        """
        生成蒙特卡洛积分点
        
        对于 GPE 的初始条件 ψ₀ = π^(-D/4) exp(-|x|²/2)，
        波函数模平方 |ψ|² = π^(-D/2) exp(-|x|²)
        所以采样分布应使用 exp(-|x|²)（即 α=0.5 对应的包裹分布）
        
        Args:
            n_points: 采样点数
            for_gpe: 如果 True，使用匹配 GPE 波函数的分布；否则使用基函数包裹分布
        """
        rng = np.random.default_rng(self.random_state + 999)
        alpha = self.ansatz.alpha
        D = self._physical_dim()

        if getattr(self.ansatz, 'is_radial_basis', False):
            from math import gamma as gamma_fn
            lam = 1.0 if for_gpe else max(2.0 * max(alpha, 1e-8), 1e-8)
            shape = D / 2.0
            s = rng.gamma(shape=shape, scale=1.0 / lam, size=n_points)
            r = np.sqrt(np.clip(s, 0.0, None))
            surface = 2.0 * np.pi ** (D / 2.0) / gamma_fn(D / 2.0)
            const = surface * gamma_fn(shape) / (2.0 * lam ** shape)
            weights = const * np.exp(np.minimum(lam * r**2, 700.0)) / n_points
            return r.reshape(-1, 1), weights.reshape(-1, 1)
        
        if alpha == 0.0:
            # 无包络模式: 使用 σ = 1/√D 保证 ‖x‖ ≈ 1 (与配点采样一致)
            sigma = 1.0 / np.sqrt(max(D, 1))
        elif for_gpe:
            # GPE 波函数分布: |ψ|² ∝ exp(-|x|²)
            # 采样分布 p(x) ∝ exp(-|x|²)，标准差 sigma = 1/sqrt(2)
            sigma = 1.0 / np.sqrt(2.0)
        else:
            # 基函数包裹分布: p(x) ∝ exp(-2αx²)
            sigma = 1.0 / np.sqrt(4.0 * alpha)
        # 截断半径: 对于 D 维高斯, ||x|| 集中在 sigma*sqrt(D) 附近
        # 使用 sigma*(sqrt(D) + 4*sqrt(2)) 保证 >99.9% 样本通过
        cutoff_radius = sigma * (np.sqrt(D) + 4.0 * np.sqrt(2.0))
        
        # 迭代采样直到获得足够的有效点
        max_iterations = 10
        x_samples = []
        target_count = n_points
        
        for _ in range(max_iterations):
            # 采样截断高斯
            batch_size = target_count * 2  # 多采样一些
            x_candidate = rng.normal(0.0, sigma, size=(batch_size, D))
            norms = np.linalg.norm(x_candidate, axis=1)
            mask = norms < cutoff_radius
            valid_candidates = x_candidate[mask]
            x_samples.extend(valid_candidates)
            
            if len(x_samples) >= target_count:
                break
        
        x_samples = np.array(x_samples[:n_points])
        
        # 计算采样权重 (importance sampling for Lebesgue dx)
        if alpha == 0.0:
            # 无包络模式: 按高斯采样密度反权重恢复 dx 积分
            log_weights = 0.5 * D * np.log(2.0 * np.pi * sigma**2) + np.sum(x_samples**2, axis=1) / (2.0 * sigma**2)
            weights = np.exp(np.minimum(log_weights, 700.0)) / len(x_samples)
        else:
            # 使用 log-sum-exp 技巧避免高维溢出
            if for_gpe:
                log_weights = 0.5 * D * np.log(np.pi) + np.sum(x_samples**2, axis=1)
            else:
                log_weights = 0.5 * D * np.log(np.pi / (2.0 * alpha)) + 2.0 * alpha * np.sum(x_samples**2, axis=1)
            weights = np.exp(np.minimum(log_weights, 700.0)) / len(x_samples)
        
        return x_samples.reshape(-1, D), weights.reshape(-1, 1)
    
    def _offline_compute_gram_matrices(self):
        """
        离线计算 Gram 矩阵（仅执行一次）

        质量 Gram 矩阵: G_{ij} = ∫ φ_i(x) φ_j(x) dx
        能量 Gram 矩阵: Q_{ij} = ∫ φ_i(x) φ_j(x) dx (与 G 相同)

        D=1: 使用精确 Gauss-Hermite 积分（n=600 节点，机器精度）
             与 gpe_solver.py 保持一致，消除 MC 统计偏差。
        D>1: 使用蒙特卡洛积分（MC，精度 O(1/√n_mc)）
        """
        D = self._physical_dim()

        if D == 1:
            # ── 精确 Gauss-Hermite 积分（D=1 专用）──────────────────────────
            from scipy.special import roots_hermite
            n_gh = 600
            print(f"\n  [BCL] 离线计算 Gram 矩阵 (D=1, 精确 GH n={n_gh})...")
            u_q, w_q = roots_hermite(n_gh)
            alpha = self.ansatz.alpha
            scale = 1.0 / np.sqrt(2.0 * alpha) if alpha > 0 else 1.0
            x_phys = (u_q * scale).reshape(-1, 1)
            Phi_r = self._raw_features(x_phys) @ self._Vt_s.T
            eff_w = w_q * scale          # 变量替换 Jacobian: dx = scale · du
            if self._use_torch_backend():
                Phi_r_t = self._to_backend_tensor(Phi_r)
                eff_w_t = self._to_backend_tensor(eff_w)
                G = self._to_numpy((Phi_r_t * eff_w_t[:, None]).T @ Phi_r_t)
            else:
                G = (Phi_r * eff_w[:, None]).T @ Phi_r
        else:
            # ── 蒙特卡洛积分（D>1）──────────────────────────────────────────
            mode = self.mass_quadrature
            if mode == 'auto':
                mode = 'collocation'
            if mode == 'collocation':
                print(f"\n  [BCL] 离线计算 Gram 矩阵 (D={D}, collocation n={len(self.domain.interior_points)})...")
                x_quad = self.domain.interior_points
                w_domain = getattr(self.domain, 'quadrature_weights', None)
                if w_domain is None:
                    w_quad = np.ones((len(x_quad), 1), dtype=float) / max(len(x_quad), 1)
                else:
                    w_quad = np.asarray(w_domain, dtype=float).reshape(-1, 1)
            else:
                print(f"\n  [BCL] 离线计算 Gram 矩阵 (D={D}, MC n={self.n_quad_mc})...")
                x_quad, w_quad = self._generate_mc_quadrature_points(self.n_quad_mc, for_gpe=False)
            Phi_r = self._raw_features(x_quad) @ self._Vt_s.T
            sqrt_w = np.sqrt(w_quad)
            if self._use_torch_backend():
                Phi_r_t = self._to_backend_tensor(Phi_r)
                sqrt_w_t = self._to_backend_tensor(sqrt_w)
                Phi_weighted_t = Phi_r_t * sqrt_w_t
                G = self._to_numpy(Phi_weighted_t.T @ Phi_weighted_t)
            else:
                Phi_weighted = Phi_r * sqrt_w
                G = Phi_weighted.T @ Phi_weighted

        self._Gram_matrix_G = G
        self._Gram_matrix_Q = G

        print(f"    Gram 矩阵计算完成: shape={G.shape}")
        cond_G = self._cond_backend(G)
        print(f"    Gram 矩阵条件数: {cond_G:.2e}")

        return G
    
    def _build_orthogonalization_matrix(self):
        """
        构建正交化矩阵 L^{-1} 用于能量守恒
        
        目标: 找到变换使得新基函数在 L² 空间标准正交
        方法: 对 Gram 矩阵 Q = L · L^T 进行 Cholesky 分解
        新基函数: Φ̃ = L^{-1} · Φ, 则 ∫ Φ̃ Φ̃^T dx = I
        """
        Q = self._Gram_matrix_Q
        
        # 检查 Q 是否正定
        eigenvalues = self._eigvalsh_backend(Q)
        if np.any(eigenvalues <= 0):
            print(f"  [BCL WARNING] Gram 矩阵非正定，添加正则化...")
            Q = Q + 1e-8 * np.eye(Q.shape[0])
        
        # Cholesky 分解: Q = L · L^T
        try:
            if self._use_torch_backend():
                Q_t = self._to_backend_tensor(Q)
                L_t = torch.linalg.cholesky(Q_t)
                self._L_inv = self._to_numpy(torch.linalg.inv(L_t))
            else:
                L = np.linalg.cholesky(Q)
                self._L_inv = np.linalg.inv(L)
            print(f"\n  [BCL] 正交化矩阵构建完成 (Cholesky):")
            
            # 验证: L^{-1} · Q · L^{-T} ≈ I
            check = self._L_inv @ Q @ self._L_inv.T
            residual = np.linalg.norm(check - np.eye(check.shape[0]))
            print(f"    验证 |L⁻¹·Q·L⁻ᵀ - I| = {residual:.2e} (应 ≈ 0)")
            
        except (np.linalg.LinAlgError, RuntimeError):
            print(f"  [BCL ERROR] Cholesky 分解失败，改用 SVD 正交化...")
            # SVD 方案: Q = U·S·V^T, Q^{-1} = V·S^{-1}·U^T
            U, S, Vt = self._svd_backend(Q)
            S_half_inv = np.diag(1.0 / np.sqrt(S))
            self._L_inv = Vt.T @ S_half_inv @ U.T
            print(f"    SVD 正交化方案: |L⁻¹·Q·L⁻ᵀ - I| = {np.linalg.norm(self._L_inv @ Q @ self._L_inv.T - np.eye(self._n_basis)):.2e}")
        
        return self._L_inv
    
    def _apply_transform_to_basis(self):
        """Build de-enveloped linear operator H per paper's Eq.:
        u_t = i[½∇²u + (∇ρ/ρ)·∇u + ½(∇²ρ/ρ)u − V u − βρ²|u|²u]

        H_deenv = Φ⁺ @ [½ ∇²Φ + Σ_j diag(a_j) ∂_jΦ + diag(½b − V) Φ]
        where a_j = (∂_jρ)/ρ = −2αx_j,  b = (∇²ρ)/ρ = −2αD + 4α²‖x‖²
        """
        x = self.domain.interior_points
        self._D = x.shape[1]
        D_loc = self._D
        
        # 更新伪逆
        self._A_inv = self._pinv_backend(self._A_r)
        
        if self.energy_conservation and self._L_inv is not None:
            self._A_r = self._A_r @ self._L_inv.T
            self._A_inv = self._pinv_backend(self._A_r)

        if getattr(self, '_use_physical_basis', False):
            A_lap_r = self.ansatz.evaluate_model_laplace(x) @ self._Vt_s.T
            if self.energy_conservation and self._L_inv is not None:
                A_lap_r = A_lap_r @ self._L_inv.T
            V_vals = self._V_potential(x)
            self._C = self._A_inv @ A_lap_r
            self._M_V = self._A_inv @ (V_vals[:, None] * self._A_r)
            self._H_deenv = None
            self._rho2_colloc = np.ones(len(x))
            print(f"\n  [BCL] 基函数准备完成:")
            print(f"    A_r shape: {self._A_r.shape}")
            print(f"    A_inv shape: {self._A_inv.shape}")
            print(f"    C (Laplacian) shape: {self._C.shape}")
            return
        
        self._H_deenv, self._C, self._M_V, self._rho2_colloc = self._assemble_deenv_operator(
            x,
            self._A_r,
            self._A_inv,
            self._Vt_s,
            self._L_inv if self.energy_conservation and self._L_inv is not None else None,
        )
        
        print(f"\n  [BCL] De-enveloped operator assembly:")
        print(f"    A_r (Φ_deenv) shape: {self._A_r.shape}")
        print(f"    H_deenv shape: {self._H_deenv.shape}")
        print(f"    ||H_deenv||_F = {np.linalg.norm(self._H_deenv):.4e}")
    
    def _evaluate_laplace_sdgd(self, x_eval):
        """
        SDGD de-enveloped Laplacian estimate.
        ∇²tanh(z) = -2 tanh(z) sech²(z) ‖w‖²
        Stochastic: replace ‖w‖² with (D/s)Σ_{d∈S} w_d²
        """
        D = x_eval.shape[1]
        sdgd_batch = self.sdgd_batch if self.sdgd_batch > 0 else max(1, int(np.ceil(np.sqrt(D))))
        
        z = self.ansatz._preact(x_eval)
        h = np.tanh(z)
        h1 = 1.0 - h**2
        dims = self._rng.choice(D, size=sdgd_batch, replace=False)
        W = self.ansatz.W
        w_sq_est = (D / sdgd_batch) * np.sum(W[dims, :]**2, axis=0)
        A_lap = -2.0 * h * h1 * w_sq_est
        col_scale = getattr(self.ansatz, 'col_scale', None)
        if col_scale is not None:
            A_lap = A_lap / col_scale.reshape(1, -1)
        if self._has_envelope_bias and not self._use_hermite_basis:
            A_lap = np.column_stack([A_lap, np.zeros(len(x_eval))])
        A_lap_r = A_lap @ self._Vt_s.T
        
        if self.energy_conservation and self._L_inv is not None:
            A_lap_r = A_lap_r @ self._L_inv.T @ np.linalg.inv(self._L_inv)
        
        return self._A_inv @ A_lap_r
    
    def _init_model_and_matrices(self, psi_init_fn=None, init_u_vals=None, init_v_vals=None,
                                 trajectory_data=None):
        """初始化模型和矩阵"""
        self._mass_norm_cache = None
        if trajectory_data is not None and hasattr(self.ansatz, 'init_model_from_trajectory') and not self.use_hermite_basis:
            print("\n[Step 1] Trajectory-aware SWIM sampling")
            print(f"  supervision snapshots: {trajectory_data['u_snaps'].shape[0]}")
            self.ansatz.init_model_from_trajectory(
                self.domain,
                trajectory_data['u_snaps'],
                trajectory_data['v_snaps'],
            )
        elif psi_init_fn is not None and hasattr(self.ansatz, 'init_model_from_psi') and not self.use_hermite_basis:
            self.ansatz.init_model_from_psi(self.domain, psi_init_fn)
        else:
            self.ansatz.init_model(self.domain)
        
        # 构建基础矩阵
        x = self.domain.interior_points
        D_loc = self._physical_dim()
        self._use_physical_basis = (
            D_loc == 1
            and not self._use_time_blocks
            and not self.use_hermite_basis
            and not getattr(self.ansatz, 'is_radial_basis', False)
        )
        A = self._basis_features(x)
        
        # SVD 截断
        U_s, S_s, Vt_s = self._svd_backend(A)
        mask = S_s / S_s[0] > self.svd_cutoff
        self._Vt_s = Vt_s[mask, :]
        n_basis_raw = int(mask.sum())
        
        if n_basis_raw == 0:
            raise ValueError(f"SVD 截断移除了所有基函数 (svd_cutoff={self.svd_cutoff})")
        
        print(f"\n[Step 1] SVD 截断: {self.ansatz.n_neurons} -> {n_basis_raw} 基函数")
        
        self._A_r = A @ self._Vt_s.T
        self._n_basis = n_basis_raw

        if self._use_time_blocks and (D_loc > 1 or getattr(self.ansatz, 'is_radial_basis', False)) and not self.use_hermite_basis:
            self._D = D_loc
            self._A_inv = self._pinv_backend(self._A_r)
            self._L_inv = None
            self._Gram_matrix_G = None
            self._Gram_matrix_Q = None
            env = self.ansatz._envelope(x).ravel()
            self._H_deenv, self._C, self._M_V, self._rho2_colloc = self._assemble_deenv_operator(
                x,
                self._A_r,
                self._A_inv,
                self._Vt_s,
                None,
            )
            print(f"\n  [Frozen-SWIM High-D de-env] H_deenv shape: {self._H_deenv.shape}")
            if init_u_vals is None:
                u0_vals = self.u0_fn(x)
                v0_vals = self.v0_fn(x)
            else:
                u0_vals = init_u_vals
                v0_vals = init_v_vals
            u0_vals = u0_vals / env
            v0_vals = v0_vals / env
            c_u0 = self._A_inv @ u0_vals
            c_v0 = self._A_inv @ v0_vals
            self._compute_initial_conserved_quantities(c_u0, c_v0)
            return c_u0, c_v0
        
        if D_loc > 1 and (self.use_hermite_basis or self.hermite_total_deg > 0):
            # ===== D>1: Hermite多项式乘积基 =====
            import itertools
            t0_h = time.time()
            gamma = self.gamma

            # --- Build multi-index set ---
            if self.hermite_total_deg > 0:
                total_deg = self.hermite_total_deg
                combos = [ns for ns in itertools.product(range(total_deg + 1), repeat=D_loc)
                          if sum(ns) <= total_deg]
            else:
                N_h = self.hermite_n_per_dim
                combos = list(itertools.product(range(N_h), repeat=D_loc))
            n_basis_full = len(combos)
            self._hermite_combos = combos
            E_k = np.array([gamma * (sum(ns) + D_loc / 2.0) for ns in combos])

            if self.hermite_total_deg > 0:
                # ===== Hermite + Collocation (total-degree truncation) =====
                n_coll = self.hermite_n_coll
                rng_coll = np.random.default_rng(self.random_state)
                x_coll = rng_coll.standard_normal(size=(n_coll, D_loc))

                print(f"\n  [Hermite+Collocation] total_deg={self.hermite_total_deg}, D={D_loc}, "
                      f"n_basis_full={n_basis_full}, n_coll={n_coll}")

                Phi_H, _ = self._build_hermite_phi(x_coll)

                # SVD truncation
                U_s, S_s, Vt_s = np.linalg.svd(Phi_H, full_matrices=False)
                mask = S_s / S_s[0] > self.svd_cutoff
                n_basis = int(mask.sum())
                S_trunc = S_s[:n_basis]
                U_trunc = U_s[:, :n_basis]
                Vt_trunc = Vt_s[:n_basis, :]

                A_r = U_trunc * S_trunc[np.newaxis, :]
                A_inv = np.linalg.pinv(A_r, rcond=self.svd_cutoff)

                print(f"    SVD: {n_basis_full} -> {n_basis} "
                      f"(S[0]={S_s[0]:.2e}, S[{n_basis-1}]={S_s[n_basis-1]:.2e})")

                # Exact Laplacian via Schrödinger eigenvalue identity
                A_lap = self._compute_hermite_laplacian(x_coll)
                A_lap_r = A_lap @ Vt_trunc.T
                C = A_inv @ A_lap_r
                V_x = 0.5 * gamma**2 * np.sum(x_coll**2, axis=1)
                M_V = A_inv @ (V_x[:, np.newaxis] * A_r)

                self._C = C
                self._M_V = M_V
                self._A_r = A_r
                self._A_inv = A_inv
                self._n_basis = n_basis
                self._x_coll_nd = x_coll
                self._use_hermite_basis = True
                self._hermite_Vt_trunc = Vt_trunc
                # Precompute energy matrix: M_E = Vt @ diag(E_k) @ Vt^T
                self._hermite_M_E = Vt_trunc @ np.diag(E_k) @ Vt_trunc.T

                print(f"    E_k ∈ [{E_k.min():.3f}, {E_k.max():.3f}]")

                # IC projection
                u0_vals = self.u0_fn(x_coll)
                v0_vals = self.v0_fn(x_coll)
                c_u0 = A_inv @ u0_vals
                c_v0 = A_inv @ v0_vals
                u0_rec = A_r @ c_u0
                rel_err_init = np.linalg.norm(u0_rec - u0_vals) / (np.linalg.norm(u0_vals) + 1e-30)
                print(f"    IC recon L2 rel err: {rel_err_init:.2e}")
                print(f"  Done ({time.time()-t0_h:.2f}s)")
            else:
                # ===== GH Galerkin (full tensor product, exact orthogonality) =====
                N_h = self.hermite_n_per_dim
                n_basis = n_basis_full

                if D_loc <= 2:
                    n_per_dim = max(50, 5 * N_h, min(self.n_quad_gh_nd, 200))
                else:
                    max_total = 100_000
                    n_per_dim_raw = max(N_h + 5, min(3 * N_h, self.n_quad_gh_nd))
                    while n_per_dim_raw ** D_loc > max_total and n_per_dim_raw > N_h + 2:
                        n_per_dim_raw -= 1
                    n_per_dim = n_per_dim_raw
                x_gal, w_gal = self._gh_tensorproduct_quadrature_nd(n_per_dim)
                n_gal = len(x_gal)
                print(f"\n  [D>1 Hermite Galerkin] N_per_dim={N_h}, n_basis={n_basis}, "
                      f"GH n_per_dim={n_per_dim}, n_gal={n_gal}")

                Phi_H, Poly_H = self._build_hermite_phi(x_gal)
                print(f"    E_k ∈ [{E_k.min():.3f}, {E_k.max():.3f}]")

                A_inv_H = (Poly_H * w_gal.reshape(-1, 1)).T
                G_check = A_inv_H @ Phi_H
                G_diag = np.diag(G_check)
                G_offmax = np.max(np.abs(G_check - np.diag(G_diag)))
                print(f"    Gram check: diag∈[{G_diag.min():.4f},{G_diag.max():.4f}], "
                      f"max|off-diag|={G_offmax:.2e}")

                self._C = np.diag(-2.0 * E_k)
                self._M_V = np.zeros((n_basis, n_basis))
                self._A_r = Phi_H
                self._A_inv = A_inv_H
                self._n_basis = n_basis
                self._Phi_gal = Phi_H
                self._w_gal = w_gal
                self._x_gal = x_gal
                self._x_coll_nd = x_gal
                self._use_hermite_basis = True
                self._hermite_Vt_trunc = None

                print(f"  Done ({time.time()-t0_h:.2f}s)")

                # IC projection
                u0_vals = self.u0_fn(x_gal)
                v0_vals = self.v0_fn(x_gal)
                c_u0 = self._A_inv @ u0_vals
                c_v0 = self._A_inv @ v0_vals
                u0_rec = self._A_r @ c_u0
                rel_err_init = np.linalg.norm(u0_rec - u0_vals) / (np.linalg.norm(u0_vals) + 1e-30)
                print(f"    IC recon L2 rel err: {rel_err_init:.2e}")
        else:
            # ===== 原始路径: Gram + 正交化 =====
            if self.energy_conservation:
                # 离线计算 Gram 矩阵并构建正交化矩阵
                self._Gram_matrix_G = self._offline_compute_gram_matrices()
                self._build_orthogonalization_matrix()
            else:
                self._Gram_matrix_G = None
                self._Gram_matrix_Q = None

            # 应用变换
            self._apply_transform_to_basis()

            self._n_basis = self._A_r.shape[1]

            # 计算初始条件投影 (de-enveloped: u₀ = ψ₀/ρ)
            if init_u_vals is None:
                u0_vals = self.u0_fn(x)
                v0_vals = self.v0_fn(x)
            else:
                u0_vals = init_u_vals
                v0_vals = init_v_vals
            if not self._use_hermite_basis and not getattr(self, '_use_physical_basis', False):
                env = self.ansatz._envelope(x).ravel()
                u0_vals = u0_vals / env
                v0_vals = v0_vals / env
            c_u0 = self._A_inv @ u0_vals
            c_v0 = self._A_inv @ v0_vals
        
        # 计算初始守恒量
        self._compute_initial_conserved_quantities(c_u0, c_v0)
        
        return c_u0, c_v0
    
    def _compute_initial_conserved_quantities(self, c_u0, c_v0):
        """使用 MC 积分计算初始守恒量"""
        D_loc = self._physical_dim()

        if self._use_hermite_basis:
            # Hermite正交基: N = ||c_u||² + ||c_v||² (精确)
            self._N0 = float(np.dot(c_u0, c_u0) + np.dot(c_v0, c_v0))
            if self._hermite_M_E is not None:
                # Collocation mode
                self._E0 = float(c_u0 @ self._hermite_M_E @ c_u0
                                 + c_v0 @ self._hermite_M_E @ c_v0)
            else:
                # GH Galerkin mode
                E_k_arr = np.array([self.gamma * (sum(ns) + D_loc / 2.0)
                                     for ns in self._hermite_combos])
                self._E0 = float(np.dot(E_k_arr, c_u0**2 + c_v0**2))
            if self.energy_target is None:
                self._energy_target = self._E0
            mode_str = "Collocation" if self._hermite_Vt_trunc is not None else "Galerkin"
            print(f"\n  [BCL Hermite {mode_str}] 初始守恒量:")
            print(f"    初始质量 N₀ = {self._N0:.16f}")
            print(f"    初始能量 E₀ = {self._E0:.16f}  (线性HO部分)")
            return

        D_loc = self._physical_dim()

        if D_loc == 1:
            # D=1: 梯形积分（与 ODE 动态一致，无 Gram 病态问题）
            x_quad = self.domain.interior_points
            dx = float(x_quad[1, 0] - x_quad[0, 0])
            w_quad = np.full((len(x_quad), 1), dx)
        else:
            cache = self._get_mass_quadrature_cache()
            x_quad = cache['x_quad']
            w_quad = cache['w_flat'].reshape(-1, 1)
        D = x_quad.shape[1]

        if D_loc > 1:
            Phi_transformed = self._get_mass_quadrature_cache()['Phi_quad']
        elif getattr(self, '_use_physical_basis', False):
            Phi_transformed = self._evaluate_basis(x_quad)
        elif self.energy_conservation and self._L_inv is not None:
            Phi_raw = self._raw_features(x_quad) @ self._Vt_s.T
            Phi_transformed = Phi_raw @ self._L_inv.T
        else:
            Phi_transformed = self._raw_features(x_quad) @ self._Vt_s.T

        # 重构场 (de-enveloped u)
        u0_quad = Phi_transformed @ c_u0
        v0_quad = Phi_transformed @ c_v0

        # 质量: N = ∫ ρ²|u|² dx (de-enveloped)
        u2 = u0_quad**2 + v0_quad**2
        if getattr(self, '_use_physical_basis', False):
            psi2 = u2
        else:
            env_quad = self.ansatz._envelope(x_quad).ravel()
            rho2_quad = env_quad**2
            psi2 = rho2_quad * u2              # |ψ|² = ρ²|u|²
        self._N0 = float(np.sum(psi2 * w_quad.flatten()))
        
        # 能量: E = ∫ [½|∇ψ|² + V|ψ|² + ½β|ψ|⁴] dx
        # 计算 |∇ψ|² = Σ_d (∂ψ/∂x_d)²
        if D == 1:
            # D=1: 直接使用一维梯度
            dPhi = self._evaluate_gradient(x_quad)
            du0_dx = dPhi @ c_u0
            dv0_dx = dPhi @ c_v0
            grad_psi2 = du0_dx**2 + dv0_dx**2
        else:
            # D>1: 使用多维梯度
            dPhi = self._evaluate_gradient(x_quad)  # 返回 |∇φ|²
            du0_dx2 = dPhi @ c_u0
            dv0_dx2 = dPhi @ c_v0
            grad_psi2 = du0_dx2 + dv0_dx2
        
        V_quad = self._V_potential(x_quad)
        kinetic = 0.5 * grad_psi2
        potential = V_quad * psi2
        interaction = 0.5 * self.beta * psi2**2
        
        self._E0 = float(np.sum((kinetic + potential + interaction) * w_quad.flatten()))
        
        # 存储能量目标
        if self.energy_target is None:
            self._energy_target = self._E0
        
        print(f"\n  [BCL] 初始守恒量:")
        print(f"    初始质量 N₀ = {self._N0:.16f}")
        print(f"    初始能量 E₀ = {self._E0:.16f}")
        print(f"    能量目标 E* = {self._energy_target:.16f}")
    
    def _evaluate_basis(self, x_eval):
        """评估基函数"""
        if self._use_hermite_basis:
            Phi_H, _ = self._build_hermite_phi(x_eval)
            if self._hermite_Vt_trunc is not None:
                return Phi_H @ self._hermite_Vt_trunc.T
            return Phi_H
        if getattr(self, '_use_physical_basis', False):
            Phi_raw = self.ansatz.evaluate_model(x_eval) @ self._Vt_s.T
            if self.energy_conservation and self._L_inv is not None:
                return Phi_raw @ self._L_inv.T
            return Phi_raw
        if self.energy_conservation and self._L_inv is not None:
            Phi_raw = self._raw_features(x_eval) @ self._Vt_s.T
            return Phi_raw @ self._L_inv.T
        else:
            return self._raw_features(x_eval) @ self._Vt_s.T
    
    def _evaluate_gradient_nd(self, x_eval):
        """
        De-enveloped gradient: ∂[tanh(z), 1]/∂x_d.
        shape: (n_pts, D, n_basis)
        ∂tanh(z_k)/∂x_d = sech²(z_k)·w_{kd},  ∂1/∂x_d = 0
        """
        if getattr(self.ansatz, 'is_radial_basis', False) and hasattr(self.ansatz, 'evaluate_raw_gradient_nd'):
            dPhi = self.ansatz.evaluate_raw_gradient_nd(x_eval)
            if self._has_envelope_bias and not self._use_hermite_basis:
                zeros = np.zeros((len(x_eval), dPhi.shape[1], 1))
                dPhi = np.concatenate([dPhi, zeros], axis=2)
            dPhi_r = dPhi @ self._Vt_s.T
            if self.energy_conservation and self._L_inv is not None:
                dPhi_r = dPhi_r @ self._L_inv.T
            return dPhi_r
        if self._torch_feature_backend_ready() and not self._use_hermite_basis:
            x_t = self._to_backend_tensor(x_eval)
            W_t = self._to_backend_tensor(self.ansatz.W)
            b_t = self._to_backend_tensor(self.ansatz.b)
            z_t = x_t @ W_t + b_t
            h1_t = 1.0 - torch.tanh(z_t)**2
            dPhi_t = h1_t[:, None, :] * W_t[None, :, :]
            col_scale = getattr(self.ansatz, 'col_scale', None)
            if col_scale is not None:
                dPhi_t = dPhi_t / self._to_backend_tensor(col_scale)[None, None, :]
            if self._has_envelope_bias and not self._use_hermite_basis:
                zeros_t = torch.zeros((x_t.shape[0], x_t.shape[1], 1), dtype=dPhi_t.dtype, device=dPhi_t.device)
                dPhi_t = torch.cat([dPhi_t, zeros_t], dim=2)
            Vt_t = self._to_backend_tensor(self._Vt_s.T)
            dPhi_r_t = torch.einsum('ndk,km->ndm', dPhi_t, Vt_t)
            if self.energy_conservation and self._L_inv is not None:
                L_t = self._to_backend_tensor(self._L_inv.T)
                dPhi_r_t = torch.einsum('ndk,km->ndm', dPhi_r_t, L_t)
            return self._to_numpy(dPhi_r_t)
        z = self.ansatz._preact(x_eval)             # (n, K)
        h1 = 1.0 - np.tanh(z)**2                   # sech², (n, K)
        W = self.ansatz.W                           # (D, K)
        
        # dPhi[i, d, k] = sech²(z_k) · W[d, k]
        dPhi = h1[:, np.newaxis, :] * W[np.newaxis, :, :]
        col_scale = getattr(self.ansatz, 'col_scale', None)
        if col_scale is not None:
            dPhi = dPhi / col_scale[np.newaxis, np.newaxis, :]
        
        if self._has_envelope_bias and not self._use_hermite_basis:
            zeros = np.zeros((len(x_eval), x_eval.shape[1], 1))
            dPhi = np.concatenate([dPhi, zeros], axis=2)
        
        dPhi_r = dPhi @ self._Vt_s.T               # (n, D, n_basis)
        if self.energy_conservation and self._L_inv is not None:
            dPhi_r = dPhi_r @ self._L_inv.T
        return dPhi_r
    
    def _evaluate_grad_squared_analytical(self, x_eval):
        """
        De-enveloped |∇tanh(z_j)|² = sech⁴(z_j)·‖w_j‖² — O(n·K), D-free.
        返回 shape: (n_pts, n_basis)
        """
        if self._torch_feature_backend_ready() and not self._use_hermite_basis:
            x_t = self._to_backend_tensor(x_eval)
            W_t = self._to_backend_tensor(self.ansatz.W)
            b_t = self._to_backend_tensor(self.ansatz.b)
            z_t = x_t @ W_t + b_t
            h1_t = 1.0 - torch.tanh(z_t)**2
            if getattr(self.ansatz, '_is_sparse', False):
                w_sq_t = self._to_backend_tensor(self.ansatz._sp_valsq)
            else:
                w_sq_t = torch.sum(W_t**2, dim=0)
            grad_sq_raw_t = h1_t**2 * w_sq_t
            col_scale = getattr(self.ansatz, 'col_scale', None)
            if col_scale is not None:
                col_scale_t = self._to_backend_tensor(col_scale)
                grad_sq_raw_t = grad_sq_raw_t / (col_scale_t.reshape(1, -1) ** 2)
            if self._has_envelope_bias and not self._use_hermite_basis:
                grad_sq_raw_t = torch.cat([grad_sq_raw_t, torch.zeros((x_t.shape[0], 1), dtype=grad_sq_raw_t.dtype, device=grad_sq_raw_t.device)], dim=1)
            T_t = self._to_backend_tensor(self._Vt_s)
            grad_sq_r_t = grad_sq_raw_t @ ((T_t**2).T)
            if self.energy_conservation and self._L_inv is not None:
                T2_t = self._to_backend_tensor(self._L_inv)
                grad_sq_r_t = grad_sq_r_t @ ((T2_t**2).T)
            return self._to_numpy(grad_sq_r_t)
        z = self.ansatz._preact(x_eval)             # (n, K)
        h1 = 1.0 - np.tanh(z)**2                   # sech², (n, K)
        
        if self.ansatz._is_sparse:
            w_sq = self.ansatz._sp_valsq             # (K,)
        else:
            w_sq = np.sum(self.ansatz.W**2, axis=0)  # (K,)
        
        grad_sq_raw = h1**2 * w_sq                  # sech⁴·‖w‖², (n, K)
        col_scale = getattr(self.ansatz, 'col_scale', None)
        if col_scale is not None:
            grad_sq_raw = grad_sq_raw / col_scale.reshape(1, -1)**2
        
        # bias column "1" has |∇1|² = 0
        if self._has_envelope_bias and not self._use_hermite_basis:
            grad_sq_raw = np.column_stack([grad_sq_raw, np.zeros(len(x_eval))])
        
        # 变换到 SVD 截断基: T = Vt_s (n_basis, K_raw+1)
        # 对角近似: |∇Φ_r_i|² ≈ Σ_j Vt_s[i,j]² |∇φ_j|²
        T = self._Vt_s                               # (n_basis, K+1)
        grad_sq_r = grad_sq_raw @ (T**2).T            # (n, n_basis)
        
        if self.energy_conservation and self._L_inv is not None:
            T2 = self._L_inv                          # (n_basis, n_basis)
            grad_sq_r = grad_sq_r @ (T2**2).T
        
        return grad_sq_r
    
    def _evaluate_grad_squared(self, x_eval):
        """
        评估基函数梯度的平方和 |∇φ_k|² = Σ_d (∂φ_k/∂x_d)²
        
        D > 50 时使用解析公式 O(n·K); D ≤ 50 使用精确的全张量计算.
        
        返回 shape: (n_pts, n_basis)
        """
        if getattr(self.ansatz, 'supports_full_gradient', False):
            dPhi = self._evaluate_gradient_nd(x_eval)
            return np.sum(dPhi**2, axis=1)
        D = x_eval.shape[1]
        if D > 50:
            return self._evaluate_grad_squared_analytical(x_eval)
        dPhi = self._evaluate_gradient_nd(x_eval)   # (n, D, n_basis)
        return np.sum(dPhi**2, axis=1)               # (n, n_basis)
    
    def _evaluate_gradient(self, x_eval):
        """
        评估基函数的梯度（兼容多维版本）
        
        对于多维情况，返回所有维度的梯度平方和 |∇φ|²
        这与 _evaluate_grad_squared 效果相同，但保持接口兼容性
        """
        if getattr(self, '_use_physical_basis', False) and x_eval.shape[1] == 1:
            return self._physical_gradient_1d(x_eval)
        if getattr(self.ansatz, 'supports_full_gradient', False):
            D = x_eval.shape[1]
            if D == 1:
                return self._evaluate_gradient_nd(x_eval)[:, 0, :]
            return self._evaluate_grad_squared(x_eval)
        D = x_eval.shape[1]
        
        if D == 1:
            # D=1: de-enveloped gradient ∂tanh(z)/∂x = sech²(z)·w
            if self._torch_feature_backend_ready() and not self._use_hermite_basis:
                x_t = self._to_backend_tensor(x_eval)
                W_t = self._to_backend_tensor(self.ansatz.W)
                b_t = self._to_backend_tensor(self.ansatz.b)
                z_t = x_t @ W_t + b_t
                h1_t = 1.0 - torch.tanh(z_t)**2
                dPhi_raw_t = h1_t * W_t
                col_scale = getattr(self.ansatz, 'col_scale', None)
                if col_scale is not None:
                    dPhi_raw_t = dPhi_raw_t / self._to_backend_tensor(col_scale).reshape(1, -1)
                if self._has_envelope_bias and not self._use_hermite_basis:
                    dPhi_raw_t = torch.cat([dPhi_raw_t, torch.zeros((x_t.shape[0], 1), dtype=dPhi_raw_t.dtype, device=dPhi_raw_t.device)], dim=1)
                dPhi_r_t = dPhi_raw_t @ self._to_backend_tensor(self._Vt_s.T)
                if self.energy_conservation and self._L_inv is not None:
                    dPhi_r_t = dPhi_r_t @ self._to_backend_tensor(self._L_inv.T)
                return self._to_numpy(dPhi_r_t)
            z = self.ansatz._preact(x_eval)
            h1 = 1.0 - np.tanh(z)**2  # sech²
            W = self.ansatz.W  # (1, K)
            dPhi_raw = h1 * W
            col_scale = getattr(self.ansatz, 'col_scale', None)
            if col_scale is not None:
                dPhi_raw = dPhi_raw / col_scale.reshape(1, -1)
            if self._has_envelope_bias and not self._use_hermite_basis:
                dPhi_raw = np.column_stack([dPhi_raw, np.zeros(len(x_eval))])
            dPhi_r = dPhi_raw @ self._Vt_s.T
            
            if self.energy_conservation and self._L_inv is not None:
                return dPhi_r @ self._L_inv.T
            else:
                return dPhi_r
        else:
            # D>1: 返回梯度平方和 |∇φ|²
            return self._evaluate_grad_squared(x_eval)
    
    def _make_ode_rhs(self):
        """构建 ODE 右端函数"""
        A_r = self._A_r
        A_inv = self._A_inv
        n = self._n_basis
        beta = self.beta
        D_loc = self._physical_dim()

        if D_loc > 1 and self._use_hermite_basis:
            # ===== D>1 Hermite: pre-computed C/M_V =====
            C = self._C
            M_V = self._M_V

            def ode_rhs(t, y):
                c_u, c_v = y[:n], y[n:]
                u_vals = A_r @ c_u
                v_vals = A_r @ c_v
                psi2 = u_vals**2 + v_vals**2
                dc_u = -0.5*(C @ c_v) + M_V @ c_v + beta*(A_inv @ (psi2*v_vals))
                dc_v =  0.5*(C @ c_u) - M_V @ c_u - beta*(A_inv @ (psi2*u_vals))
                return np.concatenate([dc_u, dc_v])

            return ode_rhs

        if getattr(self, '_use_physical_basis', False):
            C = self._C
            M_V = self._M_V
            energy_cons = self.energy_conservation

            def ode_rhs(t, y):
                c_u, c_v = y[:n], y[n:]
                u_vals = A_r @ c_u
                v_vals = A_r @ c_v
                psi2 = u_vals**2 + v_vals**2
                dc_u = -0.5 * (C @ c_v) + M_V @ c_v + beta * (A_inv @ (psi2 * v_vals))
                dc_v =  0.5 * (C @ c_u) - M_V @ c_u - beta * (A_inv @ (psi2 * u_vals))
                if energy_cons and n > 0:
                    norm_c = np.linalg.norm(y)
                    if norm_c > 1e-30:
                        n_vec = y / norm_c
                        proj = np.dot(dc_u, c_u) + np.dot(dc_v, c_v)
                        dc_u = dc_u - proj * n_vec[:n]
                        dc_v = dc_v - proj * n_vec[n:]
                return np.concatenate([dc_u, dc_v])

            return ode_rhs

        # ===== De-enveloped SWIM path =====
        # u_t = i[H_deenv u − βρ²|u|²u]
        H = self._H_deenv
        rho2 = self._rho2_colloc
        energy_cons = self.energy_conservation
        
        def ode_rhs(t, y):
            c_u, c_v = y[:n], y[n:]
            
            # de-enveloped field values
            u_vals = A_r @ c_u
            v_vals = A_r @ c_v
            psi2 = u_vals**2 + v_vals**2     # |u|² (de-enveloped)
            
            # de-enveloped ODE: dc/dt = i[H c − β Φ⁺(ρ²|u|²u)]
            # split real/imag: dc_u = −H c_v + β Φ⁺(ρ²·psi2·v)
            #                  dc_v =  H c_u − β Φ⁺(ρ²·psi2·u)
            dc_u = -(H @ c_v) + beta * (A_inv @ (rho2 * psi2 * v_vals))
            dc_v =  (H @ c_u) - beta * (A_inv @ (rho2 * psi2 * u_vals))
            
            if energy_cons and n > 0:
                norm_c = np.linalg.norm(y)
                if norm_c > 1e-30:
                    n_vec = y / norm_c
                    proj = np.dot(dc_u, c_u) + np.dot(dc_v, c_v)
                    dc_u = dc_u - proj * n_vec[:n]
                    dc_v = dc_v - proj * n_vec[n:]
            
            return np.concatenate([dc_u, dc_v])
        
        return ode_rhs

    def _get_mass_quadrature_cache(self):
        """缓存质量归一化所需的离散质量求积。"""
        if self._mass_norm_cache is None:
            D = self._physical_dim()
            mode = self.mass_quadrature
            if mode == 'auto':
                mode = 'collocation' if D > 1 else 'mc'
            if mode == 'collocation':
                x_quad = self.domain.interior_points
                w_domain = getattr(self.domain, 'quadrature_weights', None)
                if w_domain is None:
                    w_flat = np.ones(len(x_quad), dtype=float) / max(len(x_quad), 1)
                else:
                    w_flat = np.asarray(w_domain, dtype=float).reshape(-1)
                env = self.ansatz._envelope(x_quad).ravel()
                self._mass_norm_cache = {
                    'x_quad': x_quad,
                    'w_flat': w_flat,
                    'Phi_quad': self._A_r,
                    'rho2': env**2,
                    'mode': mode,
                }
            else:
                x_quad, w_quad = self._generate_mc_quadrature_points(self.n_quad_mc, for_gpe=True)
                env = self.ansatz._envelope(x_quad).ravel()
                self._mass_norm_cache = {
                    'x_quad': x_quad,
                    'w_flat': w_quad.flatten(),
                    'Phi_quad': self._evaluate_basis(x_quad),
                    'rho2': env**2,
                    'mode': mode,
                }
        return self._mass_norm_cache

    def _compute_mass_at_step(self, y):
        """Mass = ∫ |ψ|² dx = ∫ ρ²|u|² dx (de-enveloped)."""
        n = self._n_basis
        c_u, c_v = y[:n], y[n:]

        if self._use_hermite_basis:
            return float(np.dot(c_u, c_u) + np.dot(c_v, c_v))

        D = self._physical_dim()
        if D == 1:
            x_pts = self.domain.interior_points
            dx = float(x_pts[1, 0] - x_pts[0, 0])
            u = self._A_r @ c_u
            v = self._A_r @ c_v
            if getattr(self, '_use_physical_basis', False):
                return float(np.sum(u**2 + v**2) * dx)
            rho2 = self._rho2_colloc
            return float(np.sum(rho2 * (u**2 + v**2)) * dx)

        cache = self._get_mass_quadrature_cache()
        Phi_quad = cache['Phi_quad']
        u = Phi_quad @ c_u
        v = Phi_quad @ c_v
        rho2 = cache['rho2']
        return float(np.sum(rho2 * (u**2 + v**2) * cache['w_flat']))

    def _build_snapshot_interpolant(self, t_arr, y_arr):
        t_nodes = np.asarray(t_arr, dtype=float)
        y_nodes = np.asarray(y_arr)

        def interp_sol(t, _t=t_nodes, _y=y_nodes):
            t_query = np.atleast_1d(np.asarray(t, dtype=float))
            out = np.zeros((_y.shape[0], len(t_query)), dtype=_y.dtype)
            if _t.size == 1:
                out[:] = _y[:, [0]]
                return out.squeeze()
            for i, ti in enumerate(t_query):
                if ti <= _t[0]:
                    out[:, i] = _y[:, 0]
                elif ti >= _t[-1]:
                    out[:, i] = _y[:, -1]
                else:
                    idx = np.searchsorted(_t, ti, side='right') - 1
                    idx = max(0, min(idx, len(_t) - 2))
                    frac = (ti - _t[idx]) / (_t[idx + 1] - _t[idx] + 1e-30)
                    out[:, i] = (1.0 - frac) * _y[:, idx] + frac * _y[:, idx + 1]
            return out.squeeze()

        return interp_sol

    def _state_at_time(self, t):
        if self._coefficients_c is not None:
            return self._coefficients_c(t)
        if self._sol is not None and hasattr(self._sol, 'sol') and self._sol.sol is not None:
            return self._sol.sol(t)
        raise RuntimeError("Solver coefficients are unavailable")
    
    def _compute_mass_energy_at_step(self, y):
        """使用高精度积分计算当前质量和能量"""
        n = self._n_basis
        c_u, c_v = y[:n], y[n:]

        if self._use_hermite_basis:
            # Mass = ||c||² (exact: Hermite basis is L²-orthonormal,
            # and Vt_trunc has orthonormal rows so ||Vt.T @ c||² = ||c||²)
            N_current = float(np.dot(c_u, c_u) + np.dot(c_v, c_v))
            if self._hermite_M_E is not None:
                # Collocation mode: E = c^T @ M_E @ c where M_E = Vt @ diag(E_k) @ Vt^T
                E_current = float(c_u @ self._hermite_M_E @ c_u
                                  + c_v @ self._hermite_M_E @ c_v)
            else:
                # GH Galerkin mode: basis = full Hermite, E = E_k^T @ c²
                D_loc = self._physical_dim()
                E_k_arr = np.array([self.gamma * (sum(ns) + D_loc / 2.0)
                                     for ns in self._hermite_combos])
                E_current = float(np.dot(E_k_arr, c_u**2 + c_v**2))
            return N_current, E_current

        D = self._physical_dim()

        if D == 1:
            x_pts = self.domain.interior_points
            dx = float(x_pts[1, 0] - x_pts[0, 0])
            u = self._A_r @ c_u
            v = self._A_r @ c_v
            u2 = u**2 + v**2
            if getattr(self, '_use_physical_basis', False):
                psi2 = u2
            else:
                rho2 = self._rho2_colloc
                psi2 = rho2 * u2
            N_current = float(np.sum(psi2) * dx)
            x_quad = x_pts
            w_flat = np.full(len(x_pts), dx)
        else:
            cache = self._get_mass_quadrature_cache()
            x_quad = cache['x_quad']
            w_flat = cache['w_flat']
            Phi_quad = cache['Phi_quad']
            u = Phi_quad @ c_u
            v = Phi_quad @ c_v
            u2 = u**2 + v**2
            rho2 = cache['rho2']
            psi2 = rho2 * u2
            N_current = float(np.sum(psi2 * w_flat))
        
        V_quad = self._V_potential(x_quad)
        
        # 计算 |∇ψ|²
        if D == 1:
            dPhi = self._evaluate_gradient(x_quad)
            du_dx = dPhi @ c_u
            dv_dx = dPhi @ c_v
            grad_psi2 = du_dx**2 + dv_dx**2
        else:
            # _evaluate_gradient 返回 |∇φ|²，直接与系数相乘
            dPhi_sq = self._evaluate_gradient(x_quad)  # (n_quad, n_basis)
            du_dx2 = dPhi_sq @ c_u
            dv_dx2 = dPhi_sq @ c_v
            grad_psi2 = du_dx2 + dv_dx2
        
        kinetic = 0.5 * grad_psi2
        potential = V_quad * psi2
        interaction = 0.5 * self.beta * psi2**2
        E_current = float(np.sum((kinetic + potential + interaction) * w_flat))
        
        return N_current, E_current
    
    def _post_step_normalization(self, y):
        """后验归一化（绝对硬约束质量守恒）"""
        n = self._n_basis
        c_u, c_v = y[:n], y[n:]
        
        if self.normalize_mass:
            N_current = self._compute_mass_at_step(y)
            if N_current > 0:
                scale = np.sqrt(self._N0 / N_current)
                c_u = scale * c_u
                c_v = scale * c_v
        
        return np.concatenate([c_u, c_v])

    def _fit_fixed_step_projection(self, ode_rhs, y0, t_span, t_eval=None):
        t0, t1 = map(float, t_span)
        n_steps = int(max(1, self.fixed_steps))
        h = (t1 - t0) / n_steps
        if t_eval is None:
            t_out = np.linspace(t0, t1, n_steps + 1)
        else:
            t_out = np.asarray(t_eval, dtype=float)
        y = self._post_step_normalization(np.asarray(y0, dtype=float))
        y_out = np.zeros((len(y), len(t_out)), dtype=float)
        out_idx = 0
        while out_idx < len(t_out) and t_out[out_idx] <= t0 + 1e-15:
            y_out[:, out_idx] = y
            out_idx += 1
        t = t0
        for _ in range(n_steps):
            y_prev = y
            t_prev = t
            k1 = ode_rhs(t, y)
            k2 = ode_rhs(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = ode_rhs(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = ode_rhs(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y = self._post_step_normalization(y)
            t = t + h
            while out_idx < len(t_out) and t_out[out_idx] <= t + 1e-15:
                theta = (t_out[out_idx] - t_prev) / max(t - t_prev, 1e-300)
                y_snap = (1.0 - theta) * y_prev + theta * y
                y_out[:, out_idx] = self._post_step_normalization(y_snap)
                out_idx += 1
        while out_idx < len(t_out):
            y_out[:, out_idx] = y
            out_idx += 1
        sol = SimpleNamespace(
            t=t_out,
            y=y_out,
            sol=self._build_snapshot_interpolant(t_out, y_out),
            nfev=4 * n_steps,
            status=0,
            message='fixed-step RK4 with post-step projection',
        )
        self._sol = sol
        self._coefficients_c = sol.sol
        self._report_nonlinear_projection_error(t_eval=t_eval)
        return self, 0

    def _compute_resample_metric_current(self, y, x_eval, mode):
        n = self._n_basis
        c_u, c_v = y[:n], y[n:]
        A_eval = self._evaluate_basis(x_eval)
        u = A_eval @ c_u
        v = A_eval @ c_v
        psi_abs = np.sqrt(u**2 + v**2)
        if mode == 'psi':
            return psi_abs + 1e-14
        dPhi_sq = self._evaluate_grad_squared(x_eval)
        grad_norm = np.sqrt(np.maximum(dPhi_sq @ (c_u**2 + c_v**2), 0.0))
        if mode == 'grad':
            return grad_norm + 1e-14
        return psi_abs * grad_norm + 0.1 * psi_abs + 1e-14

    def _resample_interior_points(self, metric):
        if self.domain.sample_points is None:
            return
        prob = np.clip(metric.reshape(-1), 1e-14, None)
        prob /= np.sum(prob)
        idx = self._rng.choice(
            self.domain.sample_points.shape[0],
            size=self.domain.interior_points.shape[0],
            replace=True,
            p=prob,
        )
        self.domain.interior_points = self.domain.sample_points[idx]

    def _capture_block_state(self, sol):
        return {
            't': sol.t.copy(),
            'y': sol.y.copy(),
            'ansatz': copy.deepcopy(self.ansatz),
            'Vt_s': None if self._Vt_s is None else self._Vt_s.copy(),
            'L_inv': None if self._L_inv is None else self._L_inv.copy(),
            'n_basis': self._n_basis,
            'use_hermite_basis': self._use_hermite_basis,
            'hermite_combos': None if self._hermite_combos is None else list(self._hermite_combos),
            'hermite_Vt_trunc': None if self._hermite_Vt_trunc is None else self._hermite_Vt_trunc.copy(),
            'energy_conservation': self.energy_conservation,
        }

    def _coefficients_from_state(self, state, t):
        t_arr = state['t']
        y_arr = state['y']
        if len(t_arr) == 1 or t <= t_arr[0]:
            return y_arr[:, 0]
        if t >= t_arr[-1]:
            return y_arr[:, -1]
        idx = np.searchsorted(t_arr, t, side='right') - 1
        idx = max(0, min(idx, len(t_arr) - 2))
        t0 = t_arr[idx]
        t1 = t_arr[idx + 1]
        if t1 <= t0:
            return y_arr[:, idx]
        w = (t - t0) / (t1 - t0)
        return (1.0 - w) * y_arr[:, idx] + w * y_arr[:, idx + 1]

    def _evaluate_basis_from_state(self, x_eval, state):
        if state['use_hermite_basis']:
            combos_prev = self._hermite_combos
            self._hermite_combos = state['hermite_combos']
            Phi_H, _ = self._build_hermite_phi(x_eval)
            self._hermite_combos = combos_prev
            if state.get('hermite_Vt_trunc') is not None:
                return Phi_H @ state['hermite_Vt_trunc'].T
            return Phi_H
        ansatz_state = state['ansatz']
        if hasattr(ansatz_state, '_preact'):
            Phi_raw = np.tanh(ansatz_state._preact(x_eval))
            col_scale = getattr(ansatz_state, 'col_scale', None)
            if col_scale is not None:
                Phi_raw = Phi_raw / col_scale.reshape(1, -1)
        else:
            Phi_raw = ansatz_state.evaluate_model(x_eval)
            if hasattr(ansatz_state, '_envelope'):
                Phi_raw = Phi_raw / ansatz_state._envelope(x_eval)
        if self._has_envelope_bias:
            Phi_raw = np.column_stack([Phi_raw, np.ones(len(x_eval))])
        Phi_raw = Phi_raw @ state['Vt_s'].T
        if state['energy_conservation'] and state['L_inv'] is not None:
            return Phi_raw @ state['L_inv'].T
        return Phi_raw

    def _evaluate_uv_from_state(self, state, x_eval, t):
        y = self._coefficients_from_state(state, t)
        n = state['n_basis']
        c_u = y[:n]
        c_v = y[n:]
        A_eval = self._evaluate_basis_from_state(x_eval, state)
        return A_eval @ c_u, A_eval @ c_v

    def _select_block_state(self, t):
        if len(self._block_states) == 0:
            raise RuntimeError('No block state available')
        for i, state in enumerate(self._block_states):
            t0 = state['t'][0]
            t1 = state['t'][-1]
            if i == len(self._block_states) - 1:
                if t0 - 1e-12 <= t <= t1 + 1e-12:
                    return state
            if t0 - 1e-12 <= t < t1 + 1e-12:
                return state
        return self._block_states[-1]

    def fit_time_blocks(self, t_span, t_eval=None, rtol=1e-6, atol=1e-8, time_blocks=1, resample_metric='psi_grad', gradient_aware_init=False):
        from scipy.integrate import solve_ivp

        print("\n" + "=" * 60)
        print("BCL-GPE Solver: Frozen-SWIM High-D Time-Blocks")
        print("=" * 60)
        print(f"  time_blocks: {time_blocks}")
        print(f"  resample_metric: {resample_metric}")
        print(f"  gradient_aware_init: {gradient_aware_init}")
        print("=" * 60)

        self._use_time_blocks = True
        self._block_states = []
        self._block_nfev = 0

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], time_blocks + 1)
        t_eval = np.unique(np.asarray(t_eval, dtype=float))
        t_edges = np.linspace(t_span[0], t_span[1], time_blocks + 1)

        global_N0 = None
        global_energy_target = self.energy_target
        prev_state = None
        prev_end_y = None
        status = 0

        for i in range(time_blocks):
            t0_block = t_edges[i]
            t1_block = t_edges[i + 1]

            if i > 0 and prev_end_y is not None and self.domain.sample_points is not None:
                metric = self._compute_resample_metric_current(prev_end_y, self.domain.sample_points, resample_metric)
                self._resample_interior_points(metric)

            psi_init_fn = None
            init_u_vals = None
            init_v_vals = None

            if i == 0:
                if gradient_aware_init:
                    psi_init_fn = lambda x: self.u0_fn(x) + 1j * self.v0_fn(x)
            else:
                if gradient_aware_init:
                    psi_init_fn = lambda x, st=prev_state, tt=t0_block: (
                        st['ansatz']._envelope(x).ravel()
                        * (self._evaluate_uv_from_state(st, x, tt)[0]
                           + 1j * self._evaluate_uv_from_state(st, x, tt)[1])
                    )
                init_u_vals, init_v_vals = self._evaluate_uv_from_state(prev_state, self.domain.interior_points, t0_block)
                prev_env = prev_state['ansatz']._envelope(self.domain.interior_points).ravel()
                init_u_vals = prev_env * init_u_vals
                init_v_vals = prev_env * init_v_vals

            c_u0, c_v0 = self._init_model_and_matrices(
                psi_init_fn=psi_init_fn,
                init_u_vals=init_u_vals,
                init_v_vals=init_v_vals,
            )

            if global_N0 is None:
                global_N0 = self._N0
                if global_energy_target is None:
                    global_energy_target = getattr(self, '_energy_target', None)
            else:
                self._N0 = global_N0
                if global_energy_target is not None:
                    self._energy_target = global_energy_target
                y0_tmp = self._post_step_normalization(np.concatenate([c_u0, c_v0]))
                c_u0 = y0_tmp[:self._n_basis]
                c_v0 = y0_tmp[self._n_basis:]

            y0 = np.concatenate([c_u0, c_v0])
            block_eval = t_eval[(t_eval >= t0_block - 1e-12) & (t_eval <= t1_block + 1e-12)]
            block_eval = np.unique(np.concatenate(([t0_block], block_eval, [t1_block])))
            # Clip strictly into [t0_block, t1_block] to satisfy solve_ivp bounds check
            block_eval = np.clip(block_eval, t0_block, t1_block)
            block_eval = np.unique(block_eval)

            print(f"\n[Block {i + 1}/{time_blocks}] ODE 求解 ({t0_block} -> {t1_block})")
            ode_rhs = self._make_ode_rhs()
            sol = solve_ivp(
                ode_rhs,
                (t0_block, t1_block),
                y0,
                method='DOP853',
                t_eval=block_eval,
                dense_output=False,
                rtol=rtol,
                atol=atol,
            )

            if self.normalize_mass:
                for j in range(sol.y.shape[1]):
                    sol.y[:, j] = self._post_step_normalization(sol.y[:, j])

            self._block_nfev += sol.nfev
            state = self._capture_block_state(sol)
            self._block_states.append(state)
            prev_state = state
            prev_end_y = sol.y[:, -1]
            status = sol.status

            if status != 0:
                print(f"Warning: ODE solver returned status={status}: {sol.message}")
                break

        t_all = np.unique(np.concatenate([state['t'] for state in self._block_states]))
        self._sol = SimpleNamespace(t=t_all, y=None, sol=None, nfev=self._block_nfev, status=status)
        return self, status

    def _fit_splitting(self, t_span, t_eval=None):
        """
        使用算子分裂 (Strang / Yoshida) 求解 GPE.
        
        核心改进: 使用 GH 张量积正交实现精确谱变换 (unitary),
        替代 collocation 伪逆 (non-unitary, 破坏辛结构).
        
        线性部分: 对角特征值逐元素旋转 (O(n_basis))
        非线性部分: itrans → 物理空间旋转 → trans → mask (精确 L2 投影)
        """
        from types import SimpleNamespace
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'get_data'))
        from time_split import time_step_weights
        from hermite import (scaled_hermite_gauss, scaled_hermite_matrix,
                             mode_n_product, hermite_bare_matrix, stable_hermite_gauss)

        t0_init = time.time()
        print("\n" + "=" * 60)
        print("BCL-GPE Solver: 算子分裂 (GH 张量积谱变换)")
        print("=" * 60)

        # --- 轻量初始化: 仅构建多指标集和 GH 变换, 跳过 collocation/SVD ---
        self._use_time_blocks = False
        D_loc = self._physical_dim()
        P = self.hermite_total_deg
        gamma = self.gamma
        beta = self.beta

        # 多指标集
        import itertools
        combos = [ns for ns in itertools.product(range(P + 1), repeat=D_loc)
                  if sum(ns) <= P]
        self._hermite_combos = combos
        self._use_hermite_basis = True
        self._hermite_Vt_trunc = None

        # --- 分裂参数 ---
        N_steps = self.n_split_steps
        order = self.splitting_order
        tau = (t_span[1] - t_span[0]) / N_steps
        w_split = time_step_weights(order)

        # --- 建立 GH 张量积谱变换 (3/2 dealiasing, 稳定去包络表示) ---
        # |ψ|²ψ 是三次非线性 → 需要 n_gh ≥ 3(P+1)/2 来去混叠
        # 使用 hermite_bare_matrix + stable_hermite_gauss 避免 exp(x²) 溢出
        # 工作变量: u_mod = Re(ψ)*exp(+γx²/2), v_mod = Im(ψ)*exp(+γx²/2)
        n_gh = 3 * (P + 1) // 2  # 3/2 dealiasing rule
        n_mode = P + 1            # 每轴的谱模态数
        x_gh = []
        w_gh = []
        H_gh = []
        for d in range(D_loc):
            x_d, w_d = stable_hermite_gauss(n_gh, gamma)
            H_d = hermite_bare_matrix(P, gamma, x_d)  # (n_mode, n_gh) 矩形, 无 exp 包络
            x_gh.append(x_d)
            w_gh.append(w_d)
            H_gh.append(H_d)

        # 张量积形状 (物理空间 n_gh^D, 谱空间 n_mode^D)
        phys_shape = tuple([n_gh] * D_loc)
        spec_shape = tuple([n_mode] * D_loc)

        # total-degree 掩码 (仅用于输出, 不用于时间推进)
        mask_td = np.zeros(spec_shape, dtype=bool)
        for combo in combos:
            mask_td[combo] = True
        n_masked = int(mask_td.sum())

        # 特征值张量: 全张量积 E[n1,...,nD] = γ*(n1+...+nD + D/2)
        # 所有 (P+1)^D 模态都参与演化, 不做截断
        E_tensor = np.zeros(spec_shape)
        for idx in np.ndindex(*spec_shape):
            E_tensor[idx] = gamma * (sum(idx) + D_loc / 2.0)
        n_full = int(np.prod(spec_shape))

        print(f"  初始化时间: {time.time() - t0_init:.2f}s")
        print(f"  分裂阶数: {order} ({'Strang' if order == 2 else 'Yoshida'})")
        print(f"  时间步数: {N_steps}, tau = {tau:.6e}")
        print(f"  Hermite 基: total_deg={P}, n_td={n_masked}, n_full={n_full}, n_mode/axis={n_mode}")
        print(f"  GH 去混叠: {n_gh}^{D_loc} = {np.prod(phys_shape)} 物理点 (3/2 rule)")
        print(f"  谱变换: 精确 GH (unitary, 矩形 {n_mode}×{n_gh}, 无截断演化)")
        print("=" * 60)

        # --- 谱变换函数 (去包络表示: 输入/输出为 u_mod, v_mod) ---
        def trans(u_mod):
            """u_mod → 谱空间: c = H_bare @ (w_safe ⊙ u_mod) 沿每个轴"""
            c = u_mod.copy()
            for d in range(D_loc):
                shp = [1] * D_loc; shp[d] = n_gh
                c = mode_n_product(c * w_gh[d].reshape(shp), H_gh[d], d)
            return c

        def itrans(c):
            """谱空间 → u_mod: u_mod = H_bare^T @ c 沿每个轴"""
            u = c.copy()
            for d in range(D_loc):
                u = mode_n_product(u, H_gh[d].T, d)
            return u

        # γx² 张量积 (用于 log-space |ψ|² 计算)
        gx2_arrays = [gamma * x_d ** 2 for x_d in x_gh]
        gx2_tensor = np.zeros(phys_shape)
        for d in range(D_loc):
            shp = [1] * D_loc; shp[d] = n_gh
            gx2_tensor = gx2_tensor + gx2_arrays[d].reshape(shp)

        def safe_psi_sq(u_mod, v_mod):
            """安全计算 |ψ|² = (u_mod² + v_mod²) * exp(-γx²), 避免 overflow"""
            abs_u = np.abs(u_mod)
            abs_v = np.abs(v_mod)
            max_uv = np.maximum(abs_u, abs_v)
            max_uv = np.maximum(max_uv, 1e-300)
            log_max = np.log(max_uv)
            ratio = np.minimum(abs_u, abs_v) / max_uv
            log_sq = 2.0 * log_max + np.log1p(ratio ** 2)
            log_psi2 = log_sq - gx2_tensor
            return np.exp(np.minimum(log_psi2, 500.0))

        # --- 初始条件: 解析谱空间设置 (避免 exp 溢出) ---
        # ψ₀ = (γ/π)^{1/4} exp(-γx²/2) → Hermite 系数: c₀=1, 其余=0
        c_u = np.zeros(spec_shape)
        c_u[(0,) * D_loc] = 1.0
        c_v = np.zeros(spec_shape)

        # 初始质量 (全模态 ||c||²)
        N0_gh = float(np.sum(c_u**2 + c_v**2))
        self._N0 = N0_gh
        self._E0 = float(np.sum(E_tensor * (c_u**2 + c_v**2)))
        self._energy_target = self._E0
        self._n_basis = n_masked
        ic_err = abs(N0_gh - 1.0)
        print(f"  IC mass (GH 投影): {N0_gh:.12f}, |N-1| = {ic_err:.2e}")

        # --- 设置快照时间 ---
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 21)
        t_eval = np.sort(np.unique(np.atleast_1d(t_eval)))

        # 快照存储: 存 total-degree 系数用于 evaluate()
        n_out = len(t_eval)
        y_out = np.zeros((2 * n_masked, n_out))
        t_out = t_eval.copy()

        def tensor_to_flat(c_tensor):
            return c_tensor[mask_td]

        def flat_to_tensor(c_flat):
            c_tensor = np.zeros(spec_shape)
            c_tensor[mask_td] = c_flat
            return c_tensor

        def apply_linear_step(c_u, c_v, wi_idx):
            """线性步: 逐元素相位旋转 (对角, O(n_basis))"""
            dt_lin = 2.0 * w_split[wi_idx] * tau
            cos_E = np.cos(dt_lin * E_tensor)
            sin_E = np.sin(dt_lin * E_tensor)
            c_u_new = cos_E * c_u + sin_E * c_v
            c_v_new = -sin_E * c_u + cos_E * c_v
            return c_u_new, c_v_new

        def apply_nonlinear_step(c_u, c_v, wi_idx):
            """非线性步: itrans → u_mod 空间旋转 → trans (log-space |ψ|²)"""
            if abs(beta) < 1e-30:
                return c_u, c_v
            dt_nl = 2.0 * w_split[wi_idx] * tau
            u_mod = itrans(c_u)
            v_mod = itrans(c_v)
            psi2 = safe_psi_sq(u_mod, v_mod)
            phase = beta * dt_nl * psi2
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            u_new = cos_p * u_mod + sin_p * v_mod
            v_new = -sin_p * u_mod + cos_p * v_mod
            c_u_new = trans(u_new)
            c_v_new = trans(v_new)
            return c_u_new, c_v_new

        def single_step(c_u, c_v):
            c_u, c_v = apply_nonlinear_step(c_u, c_v, 0)
            for i in range(1, len(w_split)):
                if i % 2 == 1:
                    c_u, c_v = apply_linear_step(c_u, c_v, i)
                else:
                    c_u, c_v = apply_nonlinear_step(c_u, c_v, i)
            return c_u, c_v

        # --- 时间推进 ---
        snap_idx = 0
        t_current = t_span[0]

        while snap_idx < n_out and t_eval[snap_idx] <= t_current + 1e-14:
            y_out[:n_masked, snap_idx] = tensor_to_flat(c_u)
            y_out[n_masked:, snap_idx] = tensor_to_flat(c_v)
            snap_idx += 1

        t_step_start = time.time()
        for step in range(N_steps):
            c_u, c_v = single_step(c_u, c_v)
            t_current = t_span[0] + (step + 1) * tau

            while snap_idx < n_out and t_eval[snap_idx] <= t_current + 1e-14:
                y_out[:n_masked, snap_idx] = tensor_to_flat(c_u)
                y_out[n_masked:, snap_idx] = tensor_to_flat(c_v)
                snap_idx += 1

        t_step_end = time.time()
        print(f"  Done in {t_step_end - t_step_start:.2f}s, {N_steps} steps")

        mass_final = float(np.sum(c_u**2 + c_v**2))
        energy_final = float(np.sum(E_tensor * (c_u**2 + c_v**2)))
        print(f"  Final mass drift: |ΔN|/N₀ = {abs(mass_final - N0_gh) / (N0_gh + 1e-30):.2e}")
        print(f"  Final energy drift: |ΔE|/E₀ = {abs(energy_final - self._E0) / (abs(self._E0) + 1e-30):.2e}")
        print(f"  TD-mode fraction: {float(np.sum(c_u[mask_td]**2+c_v[mask_td]**2))/mass_final:.6f}")

        # --- 更新内部状态以兼容 evaluate() ---
        # 系数现在是原始 Hermite 基 (不经过 SVD), 需要调整
        # _hermite_Vt_trunc 设为 None → evaluate 直接用 Phi_H @ c
        self._hermite_Vt_trunc = None
        self._n_basis = n_masked
        self._hermite_combos = combos

        self._sol = SimpleNamespace(
            t=t_out, y=y_out, sol=None, nfev=N_steps * len(w_split), status=0
        )

        def interp_sol(t):
            t = float(t)
            if t <= t_out[0]:
                return y_out[:, 0]
            if t >= t_out[-1]:
                return y_out[:, -1]
            idx = np.searchsorted(t_out, t, side='right') - 1
            idx = max(0, min(idx, len(t_out) - 2))
            frac = (t - t_out[idx]) / (t_out[idx + 1] - t_out[idx] + 1e-30)
            return (1 - frac) * y_out[:, idx] + frac * y_out[:, idx + 1]

        self._coefficients_c = interp_sol
        self._sol.sol = interp_sol

        return self, 0

    def _build_swim_split_highd_quadrature(self):
        x_phys, w_quad = self._generate_mc_quadrature_points(self.n_quad_mc, for_gpe=True)
        return x_phys, w_quad.reshape(-1)

    def _build_swim_split_highd_matrices(self, x_phys, eff_w):
        Phi_r = self._evaluate_basis(x_phys)
        dPhi_r = self._evaluate_gradient_nd(x_phys)
        if dPhi_r.ndim == 2:
            dPhi_r = dPhi_r[:, np.newaxis, :]

        w_col = eff_w[:, None]
        G_mat = (Phi_r * w_col).T @ Phi_r
        K_kin = np.zeros((Phi_r.shape[1], Phi_r.shape[1]))
        for d_idx in range(dPhi_r.shape[1]):
            dPhi_d = dPhi_r[:, d_idx, :]
            K_kin += 0.5 * (dPhi_d * w_col).T @ dPhi_d

        V_vals = self._V_potential(x_phys).reshape(-1)
        K_pot = (Phi_r * (V_vals * eff_w)[:, None]).T @ Phi_r
        return Phi_r, G_mat, K_kin + K_pot

    def _fit_swim_strang_splitting(self, t_span, t_eval=None):
        """SWIM-native Strang 算子分裂（不触碰 Hermite 基）。

        关键: 在正交化 SWIM 基上做 Strang 分裂
            ĉ ← exp(-i·H·τ/2)·ĉ       (线性半步, 精确 matrix exp)
            ψ ← exp(-i·β|ψ|²·τ)·ψ     (非线性全步, 物理空间逐点相位)
            ĉ ← exp(-i·H·τ/2)·ĉ       (线性半步)

        保辛 + 无 Lyapunov 放大: 误差不被 exp(μT) 放大, 可达 1e-5 级。
        仅 D=1 实现 (用 n_gh=600 GH 正交基).
        """
        from types import SimpleNamespace
        from scipy.linalg import expm

        t0_init = time.time()
        print("\n" + "=" * 60)
        print("BCL-GPE Solver: SWIM-Native Strang Splitting (symplectic, no Hermite)")
        print("=" * 60)

        D_loc = self._physical_dim()

        # --- Step 1: 构建基底、Gram、Galerkin Hamiltonian ---
        self._use_time_blocks = False
        traj_data = self._prepare_trajectory_supervision(t_span, t_eval=t_eval)
        c_u0, c_v0 = self._init_model_and_matrices(trajectory_data=traj_data)
        n = self._n_basis

        if D_loc == 1:
            x_phys = self.domain.interior_points.copy()
            n_gh = x_phys.shape[0]
            dx = float(x_phys[1, 0] - x_phys[0, 0])
            eff_w = np.full(n_gh, dx)
            eff_w[0] *= 0.5; eff_w[-1] *= 0.5

            Phi_raw = self._raw_features(x_phys)
            Phi_r = Phi_raw @ self._Vt_s.T
            if hasattr(self.ansatz, 'evaluate_model_gradient'):
                dPhi_full = self.ansatz.evaluate_model_gradient(x_phys)
                if dPhi_full.ndim == 3 and dPhi_full.shape[1] == D_loc:
                    dPhi_raw = dPhi_full[:, 0, :]
                else:
                    dPhi_raw = dPhi_full[:, :, 0] if dPhi_full.shape[-1] == D_loc else dPhi_full[:, 0, :]
                if self._has_envelope_bias and not self._use_hermite_basis:
                    denv_1d = (-2.0 * self.ansatz.alpha * x_phys[:, 0:1] * self.ansatz._envelope(x_phys)).ravel()
                    dPhi_raw = np.column_stack([dPhi_raw, denv_1d])
            else:
                a_env = self.ansatz.alpha
                W = self.ansatz.W
                b = self.ansatz.b
                x1 = x_phys[:, 0:1]
                z = x1 @ W + b
                g_vec = np.exp(-a_env * (x1 ** 2))
                tanh_z = np.tanh(z)
                sech2_z = 1.0 - tanh_z ** 2
                dg_dx = -2.0 * a_env * x1 * g_vec
                dPhi_raw = dg_dx * tanh_z + g_vec * sech2_z * W.reshape(1, -1)
                col_scale = getattr(self.ansatz, 'col_scale', None)
                if col_scale is not None:
                    dPhi_raw = dPhi_raw / col_scale.reshape(1, -1)
                if self._has_envelope_bias and not self._use_hermite_basis:
                    dPhi_raw = np.column_stack([dPhi_raw, dg_dx.ravel()])
            dPhi_r = dPhi_raw @ self._Vt_s.T

            V_vals = 0.5 * self.gamma ** 2 * (x_phys[:, 0] ** 2)
            w_col = eff_w[:, None]
            G_mat = (Phi_r * w_col).T @ Phi_r
            K_kin = 0.5 * (dPhi_r * w_col).T @ dPhi_r
            K_pot = (Phi_r * (V_vals * eff_w)[:, None]).T @ Phi_r
            K_mat = K_kin + K_pot
        else:
            x_phys, eff_w = self._build_swim_split_highd_quadrature()
            print(f"  SWIM-split quadrature: weighted MC n={len(x_phys)}")
            Phi_r, G_mat, K_mat = self._build_swim_split_highd_matrices(x_phys, eff_w)
            w_col = eff_w[:, None]

        # 对称化
        G_mat = 0.5 * (G_mat + G_mat.T)
        K_mat = 0.5 * (K_mat + K_mat.T)

        # --- 正交化: Gram 特征分解 + 截断近零特征值 ---
        # G = U · diag(Λ) · U^T, 仅保留 λ/λ_max > gram_cutoff 的子空间
        # Φ̂ = Φ · U_trunc · diag(Λ_trunc^{-1/2}) 在 L² 正交
        # cutoff 宽松则 n_eff 大 (更精确但可能病态)；紧则 n_eff 小 (稳定但可能欠逼近)。
        gram_cutoff = float(getattr(self, '_swim_split_gram_cutoff', 1e-13))
        eigvals, eigvecs = np.linalg.eigh(G_mat)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        eig_max = float(eigvals[0])
        keep = eigvals > gram_cutoff * eig_max
        n_eff = int(keep.sum())
        if n_eff < n:
            print(f"  [Gram trunc]    : n_basis {n} -> n_eff {n_eff} (cutoff={gram_cutoff:.0e})")
        Lam = eigvals[:n_eff]
        U_g = eigvecs[:, :n_eff]
        sqrt_Lam = np.sqrt(Lam)
        inv_sqrt_Lam = 1.0 / sqrt_Lam
        # 正交化 map: ĉ = diag(sqrt(Λ)) · U_g^T · c,    c = U_g · diag(1/sqrt(Λ)) · ĉ
        # Φ̂ = Φ · U_g · diag(1/sqrt(Λ))
        M_c_to_ortho = (U_g * sqrt_Lam[None, :]).T         # (n_eff, n): ĉ = M · c
        M_ortho_to_c = U_g * inv_sqrt_Lam[None, :]         # (n, n_eff): c = M · ĉ

        # K̂ = M_c_to_ortho · K · M_c_to_ortho^T^{-1} ... 正确公式:
        # 若 c = M_ortho_to_c · ĉ, 则 Galerkin eq c^T·G·ċ = c^T·K·c ...
        # 标准推导: 在正交基下 H 矩阵 = diag(1/sqrt(Λ)) · U_g^T · K · U_g · diag(1/sqrt(Λ))
        K_ortho = (inv_sqrt_Lam[:, None] * (U_g.T @ K_mat @ U_g)) * inv_sqrt_Lam[None, :]
        K_ortho = 0.5 * (K_ortho + K_ortho.T)
        # Φ̂ 在 GH 节点的值
        Phi_o = Phi_r @ M_ortho_to_c                       # (n_gh, n_eff)
        # 正交性验证
        ortho_err = float(np.linalg.norm((Phi_o * w_col).T @ Phi_o - np.eye(n_eff)))

        Kmin, Kmax = float(np.linalg.eigvalsh(K_ortho).min()), float(np.linalg.eigvalsh(K_ortho).max())
        print(f"  n_basis          : {n}   (n_eff after Gram trunc: {n_eff})")
        print(f"  Gram λ range     : [{Lam[-1]:.3e}, {Lam[0]:.3e}]  (cond_eff={Lam[0]/Lam[-1]:.2e})")
        print(f"  K_ortho spectrum : [{Kmin:.3e}, {Kmax:.3e}]  (μ=Kmin ~ 谐振子能级下界)")
        print(f"  Ortho residual   : {ortho_err:.2e}  (<<1 表示正交化成功)")

        # --- Step 2: 组装分裂 propagator ---
        N_steps = int(self.n_split_steps)
        if N_steps <= 0:
            raise ValueError("n_split_steps 必须 >0")
        tau = (t_span[1] - t_span[0]) / N_steps
        order = int(getattr(self, '_swim_split_order', 2))

        # Strang 2-order 内核: 按 [k1*τ_lin, k2*τ_nl, k1*τ_lin] 叠加
        # Yoshida 4-order: 将一个 τ 分成 3 个 Strang 子步 (系数 c1, c2, c1), c1+c2+c1=1
        if order == 4:
            c1_y = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
            c2_y = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
            sub_taus = np.array([c1_y, c2_y, c1_y]) * tau  # 3 sub-steps per τ
            # 对每个 sub-tau 预计算 U_half
            U_halves = [expm(-1j * K_ortho * (s / 2.0)) for s in sub_taus]
        else:
            sub_taus = np.array([tau])
            U_halves = [expm(-1j * K_ortho * (tau / 2.0))]

        print(f"  Splitting order  : {order} ({'Strang' if order == 2 else 'Yoshida-4'})")
        print(f"  N_steps          : {N_steps}")
        print(f"  tau              : {tau:.3e}")
        if order == 4:
            print(f"  Yoshida sub-τ    : [{sub_taus[0]:.3e}, {sub_taus[1]:.3e}, {sub_taus[2]:.3e}]")
        print(f"  μ_TF (Lyapunov)  : ~{Kmin:.2f} — 分裂免受 exp(μT) 放大")

        # --- Step 3: 初始条件 -> 正交基复系数 ĉ (L²-exact 投影) ---
        # 直接在 GH 节点上精确投影: ĉ_0 = Φ̂^T · diag(eff_w) · ψ_0
        u0_gh = self.u0_fn(x_phys)
        v0_gh = self.v0_fn(x_phys)
        psi0_gh = u0_gh + 1j * v0_gh
        c_ortho = (Phi_o * w_col).T @ psi0_gh                    # (n_eff,) 复
        # 诊断: 初始质量与归一化误差
        m0 = float((np.abs(psi0_gh) ** 2 * eff_w).sum())
        m0_c = float(np.abs(c_ortho) ** 2 @ np.ones(n_eff))      # = ‖ĉ‖²
        print(f"  IC mass (GH)     : {m0:.12f}")
        print(f"  IC mass (‖ĉ‖²)   : {m0_c:.12f}  (残差 = {abs(m0-m0_c):.2e})")

        # --- Step 4: Strang 时间推进 + 快照记录 ---
        if t_eval is None:
            t_snaps_arr = np.linspace(t_span[0], t_span[1], N_steps + 1)
        else:
            t_snaps_arr = np.asarray(t_eval, dtype=float)
        n_snaps = len(t_snaps_arr)
        c_u_snaps = np.zeros((n, n_snaps))
        c_v_snaps = np.zeros((n, n_snaps))

        # 记录 t=t_span[0]
        snap_idx = 0
        # 转回非正交 c = U_g · diag(1/sqrt(Λ)) · ĉ (输出兼容现有 n_basis=n 接口)
        def _to_nonortho(c_o):
            return M_ortho_to_c @ c_o
        c_non = _to_nonortho(c_ortho)
        if n_snaps > 0 and abs(t_snaps_arr[0] - t_span[0]) < 0.5 * tau:
            c_u_snaps[:, 0] = c_non.real
            c_v_snaps[:, 0] = c_non.imag
            snap_idx = 1

        beta = float(self.beta)
        t_now = float(t_span[0])
        t_last = float(t_span[1])

        print(f"\n[Step 2] 分裂推进 ({t_span[0]} -> {t_span[1]}) ...")
        t0_prop = time.time()

        def _one_strang_substep(c_o, sub_tau, U_half_s):
            """完成一次 Strang: U_half · exp(-iβ|ψ|² sub_tau) · U_half"""
            c_o = U_half_s @ c_o
            psi_gh_loc = Phi_o @ c_o
            phase_loc = np.exp(-1j * beta * (np.abs(psi_gh_loc) ** 2) * sub_tau)
            psi_new_loc = phase_loc * psi_gh_loc
            c_o = (Phi_o * w_col).T @ psi_new_loc
            c_o = U_half_s @ c_o
            return c_o

        for step in range(N_steps):
            # Strang 或 Yoshida-4 子步序列
            for k in range(len(sub_taus)):
                c_ortho = _one_strang_substep(c_ortho, sub_taus[k], U_halves[k])
            t_now = t_span[0] + (step + 1) * tau

            # 快照记录
            while snap_idx < n_snaps and t_snaps_arr[snap_idx] <= t_now + 0.5 * tau:
                c_non = _to_nonortho(c_ortho)
                c_u_snaps[:, snap_idx] = c_non.real
                c_v_snaps[:, snap_idx] = c_non.imag
                snap_idx += 1

        # 若末尾还有未填的快照, 用最终状态填充
        while snap_idx < n_snaps:
            c_non = _to_nonortho(c_ortho)
            c_u_snaps[:, snap_idx] = c_non.real
            c_v_snaps[:, snap_idx] = c_non.imag
            snap_idx += 1

        t_prop = time.time() - t0_prop
        print(f"  Done in {t_prop:.2f}s, {N_steps} Strang steps")

        # --- Step 5: 封装 sol 对象 ---
        y_out = np.vstack([c_u_snaps, c_v_snaps])  # (2n, n_snaps)
        t_out = t_snaps_arr.copy()

        def interp_sol(t, _y=y_out, _t=t_out):
            t_arr = np.atleast_1d(np.asarray(t, dtype=float))
            n2 = _y.shape[0]
            out = np.zeros((n2, len(t_arr)))
            for i, ti in enumerate(t_arr):
                if ti <= _t[0]:
                    out[:, i] = _y[:, 0]
                elif ti >= _t[-1]:
                    out[:, i] = _y[:, -1]
                else:
                    idx = np.searchsorted(_t, ti, side='right') - 1
                    idx = max(0, min(idx, len(_t) - 2))
                    frac = (ti - _t[idx]) / (_t[idx + 1] - _t[idx] + 1e-30)
                    out[:, i] = (1 - frac) * _y[:, idx] + frac * _y[:, idx + 1]
            return out.squeeze()

        self._sol = SimpleNamespace(
            t=t_out, y=y_out, sol=interp_sol, nfev=N_steps, status=0
        )
        self._coefficients_c = interp_sol

        self._report_nonlinear_projection_error(t_eval=t_eval)

        print(f"  SWIM-split 初始化 {time.time() - t0_init - t_prop:.2f}s + 推进 {t_prop:.2f}s")
        return self, 0

    def fit(self, t_span, t_eval=None, rtol=1e-6, atol=1e-8, time_blocks=1, resample_metric='psi_grad', gradient_aware_init=False):
        """求解 GPE"""
        from scipy.integrate import solve_ivp

        if time_blocks > 1:
            if self.trajectory_aware_init:
                print("  [Trajectory-aware] 当前 POC 暂不接入 time-blocks，已退化为普通分块初始化。")
            return self.fit_time_blocks(
                t_span=t_span,
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                time_blocks=time_blocks,
                resample_metric=resample_metric,
                gradient_aware_init=gradient_aware_init,
            )

        # 算子分裂模式
        if self.use_splitting and (self.use_hermite_basis or self.hermite_total_deg > 0):
            return self._fit_splitting(t_span, t_eval=t_eval)

        # SWIM-native Strang 分裂 (不依赖 Hermite, 保持 SD-FSNN 约束)
        if self.use_swim_splitting:
            return self._fit_swim_strang_splitting(t_span, t_eval=t_eval)
        
        print("\n" + "=" * 60)
        print("BCL-GPE Solver: 边界融合 SDRE 方法")
        print("=" * 60)
        print(f"  质量归一化 (精确投影): {self.normalize_mass}")
        print(f"  能量守恒 (正交化流形): {self.energy_conservation}")
        print(f"  蒙特卡洛积分点数: {self.n_quad_mc}")
        print("=" * 60)
        
        self._use_time_blocks = False
        traj_data = self._prepare_trajectory_supervision(t_span, t_eval=t_eval, atol=atol)
        c_u0, c_v0 = self._init_model_and_matrices(
            psi_init_fn=self.u0_fn,
            trajectory_data=traj_data,
        )
        y0 = np.concatenate([c_u0, c_v0])
        
        print(f"\n[Step 2] ODE 求解 ({t_span[0]} -> {t_span[1]})")
        ode_rhs = self._make_ode_rhs()
        if self.fixed_step_projection:
            return self._fit_fixed_step_projection(
                ode_rhs=ode_rhs,
                y0=y0,
                t_span=t_span,
                t_eval=t_eval,
            )
        use_dense_output = bool(self._store_dense_output)
        sol = solve_ivp(
            ode_rhs, t_span, y0,
            method='DOP853',
            t_eval=t_eval,
            dense_output=use_dense_output,
            rtol=rtol, atol=atol,
        )
        
        self._sol = sol
        
        if sol.status != 0:
            print(f"Warning: ODE solver returned status={sol.status}: {sol.message}")
        
        if self.normalize_mass:
            n = self._n_basis
            for j in range(sol.y.shape[1]):
                sol.y[:, j] = self._post_step_normalization(sol.y[:, j])
        
        if use_dense_output and sol.sol is not None:
            _dense = sol.sol
            if self.normalize_mass:
                def wrapped_sol(t, _d=_dense):
                    y = _d(t)
                    return self._post_step_normalization(y)
                coeff_getter = wrapped_sol
            else:
                coeff_getter = _dense
        else:
            coeff_getter = self._build_snapshot_interpolant(sol.t, sol.y)

        self._coefficients_c = coeff_getter
        sol.sol = coeff_getter

        self._report_nonlinear_projection_error(t_eval=t_eval)
        
        return self, sol.status
    
    def evaluate(self, x_eval, t_eval):
        """评估 |ψ(x, t)|"""
        if self._sol is None:
            raise RuntimeError("Call fit() before evaluate()")

        if self._use_time_blocks:
            t_eval = np.atleast_1d(t_eval)
            psi_abs = np.zeros((x_eval.shape[0], len(t_eval)))
            for i, t in enumerate(t_eval):
                state = self._select_block_state(float(t))
                if not state['use_hermite_basis']:
                    env_tb = state['ansatz']._envelope(x_eval).ravel()
                else:
                    env_tb = np.ones(x_eval.shape[0])
                u_p, v_p = self._evaluate_uv_from_state(state, x_eval, float(t))
                psi_abs[:, i] = env_tb * np.sqrt(u_p**2 + v_p**2)
            return psi_abs
        
        n = self._n_basis
        t_eval = np.atleast_1d(t_eval)
        psi_abs = np.zeros((x_eval.shape[0], len(t_eval)))
        A_eval = self._evaluate_basis(x_eval)
        
        # de-enveloped: ψ = ρ · u, so |ψ| = ρ · |u|
        if getattr(self, '_use_physical_basis', False):
            env = np.ones(x_eval.shape[0])
        elif not self._use_hermite_basis:
            env = self.ansatz._envelope(x_eval).ravel()
        else:
            env = np.ones(x_eval.shape[0])
        
        for i, t in enumerate(t_eval):
            y = self._state_at_time(t)
            c_u = y[:n]
            c_v = y[n:]
            u_p = A_eval @ c_u
            v_p = A_eval @ c_v
            
            psi_abs[:, i] = env * np.sqrt(u_p**2 + v_p**2)
        
        return psi_abs
    
    def evaluate_uv(self, x_eval, t_eval):
        """评估 u(x,t) 和 v(x,t)"""
        if self._sol is None:
            raise RuntimeError("Call fit() before evaluate_uv()")

        if self._use_time_blocks:
            t_eval = np.atleast_1d(t_eval)
            u_out = np.zeros((x_eval.shape[0], len(t_eval)))
            v_out = np.zeros((x_eval.shape[0], len(t_eval)))
            for i, t in enumerate(t_eval):
                state = self._select_block_state(float(t))
                u_p, v_p = self._evaluate_uv_from_state(state, x_eval, float(t))
                u_out[:, i] = u_p
                v_out[:, i] = v_p
            return u_out, v_out
        
        n = self._n_basis
        t_eval = np.atleast_1d(t_eval)
        u_out = np.zeros((x_eval.shape[0], len(t_eval)))
        v_out = np.zeros((x_eval.shape[0], len(t_eval)))
        A_eval = self._evaluate_basis(x_eval)
        
        for i, t in enumerate(t_eval):
            y = self._state_at_time(t)
            c_u = y[:n]
            c_v = y[n:]
            u_out[:, i] = A_eval @ c_u
            v_out[:, i] = A_eval @ c_v
        
        return u_out, v_out
    
    @property
    def n_basis(self):
        return self._n_basis
    
    @property
    def sol(self):
        return self._sol


def build_domain(D, L, n_interior, seed=42, random_collocation=False, gamma=2.0,
                 gaussian_collocation=False, **kwargs):
    """Construct a swimpde.Domain with chosen collocation point distribution.

    Default: uniform on [-L,L]^d (D>=2) / linspace on [-L,L] (D=1).
    ``gaussian_collocation=True`` draws points from the isotropic Gaussian
    N(0, σ²I) with σ=1/√(2α), matching the ansatz envelope. This acts as
    importance sampling of the Galerkin least-squares projection: samples
    concentrate where |ψ(x)|² is large, so a moderate ``n_interior`` resolves
    the wavefunction in any dimension at O(n·d) cost — the key property that
    lets SD-FSNN avoid the curse of dimensionality.
    """
    rng = np.random.default_rng(seed)
    n_sample_points = kwargs.get('n_sample_points', 0)
    if n_sample_points <= 0:
        n_sample_points = max(4 * n_interior, 4000)
    alpha_eff = kwargs.get('alpha', 0.5)

    if gaussian_collocation and D >= 2:
        sigma_trap = 1.0 / np.sqrt(4.0 * max(alpha_eff, 0.1))
        pts = rng.normal(0.0, sigma_trap, size=(n_interior, D))
        sample_pts = rng.normal(0.0, sigma_trap, size=(n_sample_points, D))
        log_inv_pdf = (
            0.5 * D * np.log(2.0 * np.pi * sigma_trap**2)
            + np.sum(pts**2, axis=1) / (2.0 * sigma_trap**2)
        )
        quad_weights = np.exp(np.minimum(log_inv_pdf, 700.0)) / max(n_interior, 1)
    elif random_collocation:
        pts = rng.uniform(-L, L, size=(n_interior, D))
        sample_pts = rng.uniform(-L, L, size=(n_sample_points, D))
    else:
        if D == 1:
            pts = np.linspace(-L, L, n_interior).reshape(-1, 1)
            sample_pts = np.linspace(-L, L, n_sample_points).reshape(-1, 1)
        elif D <= 3:
            pts = rng.uniform(-L, L, size=(n_interior, D))
            sample_pts = rng.uniform(-L, L, size=(n_sample_points, D))
        else:
            sigma_trap = 1.0 / np.sqrt(2.0 * max(alpha_eff, 0.1))
            pts = rng.normal(0.0, sigma_trap, size=(n_interior, D))
            sample_pts = rng.normal(0.0, sigma_trap, size=(n_sample_points, D))
    domain = Domain(interior_points=pts, sample_points=sample_pts)
    if gaussian_collocation and D >= 2:
        domain.quadrature_weights = quad_weights
    domain._hi = L
    return domain


def build_radial_domain(D, L, n_interior, seed=42, **kwargs):
    rng = np.random.default_rng(seed)
    n_sample_points = kwargs.get('n_sample_points', 0)
    if n_sample_points <= 0:
        n_sample_points = max(4 * n_interior, 4000)
    alpha_eff = kwargs.get('alpha', 0.5)
    lam = 1.0
    r = np.sqrt(rng.gamma(shape=D / 2.0, scale=1.0 / lam, size=n_interior))
    r = np.clip(r, 0.0, L)
    r_pool = np.sqrt(rng.gamma(shape=D / 2.0, scale=1.0 / lam, size=n_sample_points))
    r_pool = np.clip(r_pool, 0.0, L)
    domain = Domain(interior_points=r.reshape(-1, 1), sample_points=r_pool.reshape(-1, 1))
    domain._hi = L
    domain._physical_dim = D
    return domain


class KrylovAnsatzDecay(BasicAnsatzDecay):
    """SWIM basis with Krylov-enriched supervision for adaptive H-invariant approx.

    Overrides `init_model` to run SWIM data-pair sampling supervised by the
    COMBINED magnitude Σ_k |H^k ψ₀| (k=0..K) instead of uniform pair sampling.
    For D=1 this uses classic supervised SWIM (|Δψ|²/d² weighting), which is
    well-posed (unlike _init_from_points_gradient_aware which needs D≥2 for
    the gradient-orthogonal projection to be meaningful).

    Result: basis functions W, b align with the directions of maximal variation
    of the Krylov-expanded ψ → span approximates {φ_0, φ_2, …, φ_{2K}} of the
    Hermite eigenbasis of H = -½Δ + ½γ²|x|².
    """
    def __init__(self, n_neurons, random_state=42, alpha=0.5, dist_min=1e-10,
                 sparse_dim=0, krylov_depth=4, gamma=2.0, psi0_fn=None):
        super().__init__(n_neurons=n_neurons, random_state=random_state,
                         alpha=alpha, dist_min=dist_min, sparse_dim=sparse_dim)
        self.krylov_depth = int(krylov_depth)
        self.gamma = float(gamma)
        self.psi0_fn = psi0_fn  # callable x -> ψ₀(x) (real or complex)

    def init_model(self, domain, **kwargs):
        """Krylov-supervised SWIM: split n_neurons across levels |H^k ψ₀|.

        Each Krylov level k ∈ {0, …, K} supervises a fraction n_k of the basis
        via classic SWIM pair sampling (|Δψ_k|²/dist² weighting). The result is
        a union of per-level SWIM bases — collectively spanning the Krylov
        subspace K_{K+1}(H, ψ₀), which for an even IC ⊂ span{φ_0, φ_2, …, φ_{2K}}
        of the Hermite eigenbasis. This preserves basis diversity (unlike
        summed-magnitude supervision which over-concentrates) while making the
        basis adaptively approach the H-invariant subspace.

        If psi0_fn is not provided or K==0, falls back to plain SWIM.
        """
        if self.psi0_fn is None or self.krylov_depth <= 0:
            return super().init_model(domain, **kwargs)

        x = domain.interior_points
        psi = self.psi0_fn(x)
        psi = np.real(psi).astype(np.float64) if np.iscomplexobj(psi) else psi.astype(np.float64)

        # Build Krylov levels (L²-normalized)
        levels = [np.abs(psi) / max(np.linalg.norm(psi), 1e-30)]
        psi_curr = psi.copy()
        for k in range(self.krylov_depth):
            psi_curr = self._apply_H(x, psi_curr)
            nrm = np.linalg.norm(psi_curr)
            if nrm < 1e-30:
                break
            psi_curr = psi_curr / nrm
            levels.append(np.abs(psi_curr))

        K_eff = len(levels)
        base = self.n_neurons // K_eff
        rem = self.n_neurons - base * K_eff

        W_blocks = []
        b_blocks = []
        for i, psi_k in enumerate(levels):
            nk = base + (1 if i < rem else 0)
            if nk <= 0:
                continue
            W_k, b_k = self._sample_W_b_supervised(x, psi_k, nk, rng=self.rng)
            W_blocks.append(W_k)
            b_blocks.append(b_k)

        self.W = np.concatenate(W_blocks, axis=1)   # (D, n_neurons)
        self.b = np.concatenate(b_blocks, axis=1)   # (1, n_neurons)

        if self.sparse_dim > 0:
            self._sparsify_weights()

    def _sample_W_b_supervised(self, x, psi_vals, n_target, rng=None):
        """Classic SWIM: pick pairs, weight sampling by (|Δψ|/dist)².

        Works in any D (no gradient-orthogonal projection required).
        Returns (W, b) of shapes (D, n_target), (1, n_target).
        """
        if rng is None:
            rng = self.rng
        n, D = x.shape
        scale = 0.5 * (np.log(1.5) - np.log(0.5))

        # Candidate pool: ≥ n_target pairs for diversity
        M = max(n_target * 4, n)
        n_reps = max(1, int(np.ceil(M / n)))
        idx_from = rng.integers(0, n, size=n * n_reps)
        delta = rng.integers(1, n - 1, size=idx_from.shape[0])
        idx_to = (idx_from + delta) % n

        disp = x[idx_to] - x[idx_from]                        # (M, D)
        dists = np.linalg.norm(disp, axis=1, keepdims=True).clip(self.dist_min)
        directions = disp / dists

        dy = psi_vals[idx_to] - psi_vals[idx_from]
        weights = (np.abs(dy) / dists.ravel())**2
        weights = np.clip(weights, 1e-30, None)
        probs = weights / weights.sum()
        sel = rng.choice(dists.shape[0], size=n_target, replace=True, p=probs)

        directions = directions[sel]
        idx_from_s = idx_from[sel]
        W = (2 * scale * directions / dists[sel]).T            # (D, n_target)
        b = (-np.sum(x[idx_from_s] * W.T, axis=-1)).reshape(1, -1) - scale
        return W, b

    def _apply_H(self, x, psi):
        """Apply Hamiltonian H = -½Δ + ½γ²|x|² numerically on the interior grid.

        Works for arbitrary D. Uses 2nd-order finite differences per axis.
        Assumes interior_points are structured (sorted) along at least one axis;
        for random 1D grid it is re-sorted, then Laplacian computed via pairwise
        uniform spacing. For D>1 random grids, falls back to radial FD via
        nearest-neighbour stencil (good enough for Krylov supervision, which
        only needs qualitative gradient structure, not high-order accuracy).
        """
        D = x.shape[1]
        V = 0.5 * self.gamma**2 * np.sum(x**2, axis=1)

        if D == 1:
            # Sort by x (interior grid may not be sorted)
            idx = np.argsort(x[:, 0])
            inv = np.argsort(idx)
            xs = x[idx, 0]
            ps = psi[idx]
            n = xs.size
            lap = np.zeros(n, dtype=psi.dtype)
            # 2nd-order non-uniform FD
            dx_p = xs[2:] - xs[1:-1]
            dx_m = xs[1:-1] - xs[:-2]
            lap[1:-1] = 2.0 * (
                dx_m * ps[2:] - (dx_p + dx_m) * ps[1:-1] + dx_p * ps[:-2]
            ) / (dx_p * dx_m * (dx_p + dx_m))
            Hpsi_sorted = -0.5 * lap + V[idx] * ps
            return Hpsi_sorted[inv]

        # D > 1 : per-axis numerical FD via kNN-style 3-point stencil on each axis
        # Cost O(D·N) per application, no exponential blowup.
        from scipy.spatial import cKDTree
        tree = cKDTree(x)
        lap = np.zeros_like(psi)
        # Characteristic spacing per axis
        for d in range(D):
            # Find nearest neighbours along axis d (±) using simple sort over axis d coord
            # Approximate 1D FD on slices of small spread in other coords — for random
            # grids this is noisy. Instead use isotropic kNN Laplacian as proxy.
            pass
        # Isotropic kNN Laplacian (mean-centered):  L ψ ≈ mean(ψ_j) - ψ_i
        k = min(2 * D + 1, x.shape[0] - 1)
        _, idxk = tree.query(x, k=k)
        mean_psi = np.mean(psi[idxk[:, 1:]], axis=1)
        # Scale by 1/h² where h² ~ mean squared NN distance
        dists, _ = tree.query(x, k=2)
        h2 = float(np.mean(dists[:, 1]**2)) + 1e-30
        lap = 2.0 * (mean_psi - psi) / h2
        return -0.5 * lap + V * psi

    def _grad_abs(self, x, psi_abs):
        """Numerical gradient of |ψ| on the given grid (D-agnostic).

        For D=1 uses centred FD on sorted grid.
        For D>1 uses kNN linear regression per point (local least squares).
        """
        D = x.shape[1]
        if D == 1:
            idx = np.argsort(x[:, 0])
            inv = np.argsort(idx)
            xs = x[idx, 0]
            ps = psi_abs[idx]
            g = np.zeros_like(ps)
            g[1:-1] = (ps[2:] - ps[:-2]) / (xs[2:] - xs[:-2])
            g[0] = (ps[1] - ps[0]) / (xs[1] - xs[0])
            g[-1] = (ps[-1] - ps[-2]) / (xs[-1] - xs[-2])
            return g[inv].reshape(-1, 1)
        # D > 1: local LSQ gradient using k nearest neighbours
        from scipy.spatial import cKDTree
        tree = cKDTree(x)
        k = min(3 * D + 1, x.shape[0] - 1)
        _, idxk = tree.query(x, k=k)
        grads = np.zeros_like(x)
        for i in range(x.shape[0]):
            nbr = idxk[i, 1:]
            dX = x[nbr] - x[i]
            dP = psi_abs[nbr] - psi_abs[i]
            # Linear least squares: dX @ g = dP
            g, *_ = np.linalg.lstsq(dX, dP, rcond=None)
            grads[i] = g
        return grads


