from io import BytesIO
from typing import Optional, Dict, Union, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import linprog
from scipy.interpolate import CubicSpline
import sympy as sp


class OptimalControlSolverNd:
    """
    Решение линейной (в общем неавтономной) задачи оптимального управления:

        x'(t) = F(t) x(t) + G(t) u(t) + f(t),
        J = ∫_0^T (a^T x(t) + b^T u(t)) dt,
        B u(t) ≤ q.

    Дополнительно:
    - сплайн-аппроксимация фундаментальной матрицы Φ(t)
    - вычисление невязки Φ'(t) − F(t)Φ(t)
    """

    def __init__(self,
                 T: float = 6.0,
                 M: float = 1.0,
                 N: float = 5.0,
                 n: int = 2,
                 m: int = 2,
                 x0: Optional[np.ndarray] = None,
                 F_func: Optional[List[List[str]]] = None,
                 G_func: Optional[List[List[str]]] = None,
                 a: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None,
                 B: Optional[np.ndarray] = None,
                 q: Optional[np.ndarray] = None,
                 ft_func: Optional[List[str]] = None):

        self.T = float(T)
        self.M = float(M)
        self.N = float(N)
        self.n = int(n)
        self.m = int(m)

        self.x0 = np.array([5.0, 12.0], dtype=float) if x0 is None else np.array(x0, dtype=float)
        self.a = np.array([-1.0, 0.0], dtype=float) if a is None else np.array(a, dtype=float)
        self.b = np.array([0.0, 5.0], dtype=float) if b is None else np.array(b, dtype=float)

        self.B = np.array(
            [[-1, 0],
             [0, -1],
             [2, 0],
             [0, 8],
             [2, -7]], dtype=float
        ) if B is None else np.array(B, dtype=float)

        self.q = np.array([0.0, 0.0, 5.0, 20.0, 0.0], dtype=float) if q is None else np.array(q, dtype=float)

        # f(t)
        if ft_func is None:
            def default_ft(t):
                ft = np.zeros(self.n)
                ft[0] = t
                if self.n > 1:
                    ft[1] = 1.0
                return ft
            self.ft = default_ft
        else:
            self.ft = self._create_ft_func(ft_func)

        # G(t)
        if G_func is None:
            def default_gt(t):
                G = np.zeros((self.n, self.m))
                for i in range(self.n):
                    G[i, 0] = t
                return G
            self.G = default_gt
        else:
            self.G = self._create_matrix_func(G_func)

        # F(t)
        if F_func is None:
            def default_Ft(t):
                F = np.zeros((self.n, self.n))
                for i in range(self.n):
                    F[i, i] = t
                return F
            self.F = default_Ft
        else:
            self.F = self._create_matrix_func(F_func)

        self.times = None
        self.U = None
        self.x_traj = None
        self.p_traj = None
        self.ob_value = None

        self.Phi = None
        self.Phi_spline = None
        self.Phi_residual = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _create_ft_func(self, expressions):
        t = sp.symbols('t')
        funcs = [sp.lambdify(t, sp.sympify(e), 'numpy') for e in expressions]

        def ft(tv):
            v = np.zeros(self.n)
            for i in range(min(self.n, len(funcs))):
                v[i] = funcs[i](tv)
            return v

        return ft

    def _create_matrix_func(self, expressions):
        t = sp.symbols('t')
        funcs = [[sp.lambdify(t, sp.sympify(e), 'numpy') for e in row] for row in expressions]

        def mat(tv):
            A = np.zeros((len(funcs), len(funcs[0])))
            for i in range(len(funcs)):
                for j in range(len(funcs[0])):
                    A[i, j] = funcs[i][j](tv)
            return A

        return mat

    # ------------------------------------------------------------------
    # фундаментальная матрица
    # ------------------------------------------------------------------

    def _compute_fundamental(self, K):
        self.times = np.linspace(0.0, self.T, K + 1)
        n = self.n

        def ode_phi(t, y):
            Phi = y.reshape(n, n)
            return (self.F(t) @ Phi).reshape(-1)

        y0 = np.eye(n).reshape(-1)

        sol = solve_ivp(
            ode_phi,
            (0.0, self.T),
            y0,
            t_eval=self.times,
            rtol=1e-8,
            atol=1e-10
        )

        if not sol.success:
            raise RuntimeError("Не удалось построить Φ(t)")

        self.Phi = sol.y.T.reshape(K + 1, n, n)

        Y = self.Phi.reshape(K + 1, n * n)
        self.Phi_spline = CubicSpline(self.times, Y, axis=0, bc_type='natural')

    def Phi_at(self, t):
        return self.Phi_spline(t).reshape(self.n, self.n)

    def dPhi_at(self, t):
        return self.Phi_spline.derivative()(t).reshape(self.n, self.n)

    # ------------------------------------------------------------------
    # невязка Φ'(t) − F(t)Φ(t)
    # ------------------------------------------------------------------

    def compute_phi_residuals(self):
        ts = self.times
        norms = np.zeros(ts.size)

        for i, t in enumerate(ts):
            r = self.dPhi_at(t) - self.F(t) @ self.Phi_at(t)
            norms[i] = np.linalg.norm(r, ord='fro')

        self.Phi_residual = norms
        # plt.plot(self.Phi_residual)
        # plt.show()
        return self.Phi_residual

    # ------------------------------------------------------------------
    # сопряжённая система
    # ------------------------------------------------------------------

    def _solve_costate(self, K):
        self._compute_fundamental(K)

        Phi = self.Phi
        t = self.times
        n = self.n

        A = np.array([Phi[i].T @ self.a for i in range(K + 1)])

        I = np.zeros_like(A)
        for i in range(K - 1, -1, -1):
            dt = t[i + 1] - t[i]
            I[i] = I[i + 1] + 0.5 * dt * (A[i] + A[i + 1])

        self.p_traj = np.zeros((n, K + 1))
        for i in range(K + 1):
            self.p_traj[:, i] = np.linalg.solve(Phi[i].T, I[i])

    # ------------------------------------------------------------------
    # оптимальное управление
    # ------------------------------------------------------------------

    def solve_optimal_control(self, K=50):
        self._solve_costate(K)
        self.U = np.zeros((self.m, K + 1))

        for i, t in enumerate(self.times):
            c = (self.G(t).T @ self.p_traj[:, i] - self.b)
            res = linprog(c, A_ub=self.B, b_ub=self.q, bounds=(None, None))
            self.U[:, i] = res.x if res.success else 0.0

        self.compute_phi_residuals()

    # ------------------------------------------------------------------
    # траектория
    # ------------------------------------------------------------------

    def compute_trajectory(self):
        n = self.n
        K = len(self.times) - 1

        r = np.zeros((K + 1, n))
        for i, t in enumerate(self.times):
            r[i] = self.G(t) @ self.U[:, i] + self.ft(t)

        z = np.array([np.linalg.solve(self.Phi[i], r[i]) for i in range(K + 1)])

        J = np.zeros_like(z)
        for i in range(1, K + 1):
            dt = self.times[i] - self.times[i - 1]
            J[i] = J[i - 1] + 0.5 * dt * (z[i - 1] + z[i])

        self.x_traj = np.array(
            [self.Phi[i] @ (self.x0 + J[i]) for i in range(K + 1)]
        ).T

    # ------------------------------------------------------------------
    # функционал
    # ------------------------------------------------------------------

    def compute_objective(self):
        self.compute_trajectory()

        Ob1 = quad(
            lambda t: self.a @ np.array([
                np.interp(t, self.times, self.x_traj[i]) for i in range(self.n)
            ]),
            0.0, self.T
        )[0]

        Ob2 = quad(
            lambda t: self.b @ np.array([
                np.interp(t, self.times, self.U[i]) for i in range(self.m)
            ]),
            0.0, self.T
        )[0]

        self.ob_value = Ob1 + Ob2
        return self.ob_value

    def solve(self, K=50):
        self.solve_optimal_control(K)
        self.compute_objective()
        return {
            "controls": self.U,
            "trajectory": self.x_traj,
            "objective": self.ob_value,
            "phi_residual": self.Phi_residual
        }

# -------------------------------------------------------------------------
# БЛОК MAIN — ОСТАВЛЕН
# -------------------------------------------------------------------------
if __name__ == "__main__":
    solver = OptimalControlSolverNd(
        T=1.0,
        M=1.0,
        N=1.0,
        n=5,
        m=2,

        x0=[0.0, 1.0, 0.0, -1.0, 2.0],

        # Неавтономная матрица F(t)
        F_func=[
            ["t",        "0",       "0",        "0",          "0"],
            ["0",      "1+t",       "0",        "0",          "0"],
            ["0",        "0",     "-t",          "0",          "0"],
            ["0",        "0",       "0",   "2/(t+1)",          "0"],
            ["0",        "0",       "0",        "0",     "sin(t)"]
        ],

        # Неавтономная матрица G(t)
        G_func=[
            ["1",     "0"],
            ["t",     "1"],
            ["0",   "t+1"],
            ["1",     "1"],
            ["t",    "-1"]
        ],

        # Коэффициенты функционала
        a=np.array([-1.0, 0.5, 0.0, 1.0, -0.2]),
        b=np.array([0.0, 0.0]),

        # Ограничения на управление
        B=np.array([
            [-1.0,  0.0],
            [ 1.0,  0.0],
            [ 0.0, -1.0],
            [ 0.0,  1.0]
        ]),
        q=np.array([0.0, 1.0, 0.0, 1.0]),

        # Неавтономная свободная часть f(t)
        ft_func=[
            "0",
            "t",
            "0",
            "1",
            "t**2"
        ]
    )

    results = solver.solve(K=50)

    print("Objective value:", results["objective"])
    print("Phi residuals:", results["phi_residual"])

    # solver.plot_controls()
    # solver.plot_trajectories()
