from io import BytesIO
from typing import Optional, Dict, Union, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import linprog
import sympy as sp


class OptimalControlSolverNd:
    """
    Решение линейной (в общем неавтономной) задачи оптимального управления:

        x'(t) = F(t) x(t) + G(t) u(t) + f(t),
        J = ∫_0^T (a^T x(t) + b^T u(t)) dt,
        B u(t) ≤ q.

    В этой версии:
    - фундаментальная матрица Φ(t) строится численно: Φ'(t)=F(t)Φ(t), Φ(0)=I.
    - сопряженная переменная p(t) и состояние x(t) вычисляются через Φ(t).
    - управление u(t_i) ищется по условию максимума в узлах сетки через LP (linprog).
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
            def default_ft(t_val: float) -> np.ndarray:
                ft = np.zeros(self.n)
                ft[0] = t_val
                if self.n > 1:
                    ft[1] = 1.0
                return ft
            self.ft = default_ft
        else:
            self.ft = self._create_ft_func(ft_func)

        # G(t)
        if G_func is None:
            def default_gt(t_val: float) -> np.ndarray:
                gt_matrix = np.zeros((self.n, self.m))
                for i in range(self.n):
                    gt_matrix[i, 0] = t_val
                return gt_matrix
            self.G = default_gt
        else:
            self.G = self._create_matrix_func(G_func)

        # F(t)
        if F_func is None:
            def default_Ft(t_val: float) -> np.ndarray:
                ft_matrix = np.zeros((self.n, self.n))
                for i in range(self.n):
                    ft_matrix[i, i] = t_val
                return ft_matrix
            self.F = default_Ft
        else:
            self.F = self._create_matrix_func(F_func)

        # Результаты
        self.times: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.x_traj: Optional[np.ndarray] = None
        self.p_traj: Optional[np.ndarray] = None
        self.ob_value: Optional[float] = None

        # Фундаментальная матрица Φ(t_i)
        self.Phi: Optional[np.ndarray] = None  # shape (K+1, n, n)

    # -------------------------------------------------------------------------
    # Создание функций F(t), G(t), f(t) из строк
    # -------------------------------------------------------------------------

    def _create_ft_func(self, expressions: List[str]):
        t = sp.symbols('t')
        funcs = [sp.lambdify(t, sp.sympify(expr), 'numpy') for expr in expressions]

        def ft(t_value: float) -> np.ndarray:
            ft_vec = np.zeros(self.n)
            for i in range(min(len(funcs), self.n)):
                ft_vec[i] = funcs[i](t_value)
            return ft_vec

        return ft

    def _create_matrix_func(self, expressions: List[List[str]]):
        t = sp.symbols('t')

        if not expressions or not all(expressions):
            raise ValueError("Список expressions должен содержать хотя бы одну строку и один столбец")

        rows = len(expressions)
        cols = len(expressions[0])

        if not all(len(row) == cols for row in expressions):
            raise ValueError("Все строки в expressions должны иметь одинаковую длину")

        func_matrix = []
        for row_exprs in expressions:
            row_funcs = [sp.lambdify(t, sp.sympify(expr), 'numpy') for expr in row_exprs]
            func_matrix.append(row_funcs)

        def mt(t_value: float) -> np.ndarray:
            mat = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    try:
                        mat[i, j] = func_matrix[i][j](t_value)
                    except Exception:
                        mat[i, j] = 0.0
            return mat

        return mt

    # -------------------------------------------------------------------------
    # Проверка размеров
    # -------------------------------------------------------------------------

    def _validate_shapes(self):
        assert self.x0.shape == (self.n,), f"x0 must be of shape ({self.n},)"
        assert self.a.shape == (self.n,), f"a must be of shape ({self.n},)"
        assert self.b.shape == (self.m,), f"b must be of shape ({self.m},)"
        assert self.B.shape[1] == self.m, f"B must have {self.m} columns"
        assert self.q.shape[0] == self.B.shape[0], "B and q must have same number of rows"

        F0 = self.F(0.0)
        G0 = self.G(0.0)
        assert F0.shape == (self.n, self.n), f"F(t) must be ({self.n}, {self.n})"
        assert G0.shape == (self.n, self.m), f"G(t) must be ({self.n}, {self.m})"

        f0 = self.ft(0.0)
        assert f0.shape == (self.n,), f"f(t) must be vector of length {self.n}"

    # -------------------------------------------------------------------------
    # Фундаментальная матрица Φ(t)
    # -------------------------------------------------------------------------

    def _compute_fundamental(self, K: int):
        """
        Строит Φ(t_i) на сетке times:
            Φ'(t) = F(t) Φ(t),  Φ(0)=I
        """
        self.times = np.linspace(0.0, self.T, K + 1)
        n = self.n

        def ode_phi(t, y_flat):
            Phi = y_flat.reshape(n, n)
            dPhi = self.F(t) @ Phi
            return dPhi.reshape(-1)

        y0 = np.eye(n).reshape(-1)

        sol = solve_ivp(
            ode_phi,
            (0.0, self.T),
            y0=y0,
            t_eval=self.times,
            rtol=1e-8,
            atol=1e-10
        )
        if not sol.success:
            raise RuntimeError("Не удалось построить фундаментальную матрицу Φ(t)")

        self.Phi = sol.y.T.reshape(K + 1, n, n)

    # -------------------------------------------------------------------------
    # p(t) через фундаментальную матрицу
    # -------------------------------------------------------------------------

    def _solve_costate(self, K: int):
        """
        p'(t) = -F(t)^T p(t) + a,  p(T)=0

        Решение через Φ(t):
            p(t) = Φ(t)^{-T} ∫_t^T Φ(s)^T a ds
        """
        if self.times is None or self.Phi is None or len(self.times) != K + 1:
            self._compute_fundamental(K)

        n = self.n
        t_grid = self.times
        Phi = self.Phi  # (K+1,n,n)

        # A_i = Φ(t_i)^T a
        A = np.zeros((K + 1, n))
        for i in range(K + 1):
            A[i, :] = (Phi[i].T @ self.a).reshape(-1)

        # I_i = ∫_{t_i}^T Φ(s)^T a ds (трапеции, кумулятивно назад)
        I = np.zeros((K + 1, n))
        I[K, :] = 0.0
        for i in range(K - 1, -1, -1):
            dt = t_grid[i + 1] - t_grid[i]
            I[i, :] = I[i + 1, :] + 0.5 * dt * (A[i + 1, :] + A[i, :])

        # p_i = Φ(t_i)^{-T} I_i  => решаем Φ(t_i)^T p = I
        self.p_traj = np.zeros((n, K + 1))
        for i in range(K + 1):
            self.p_traj[:, i] = np.linalg.solve(Phi[i].T, I[i, :])

    # -------------------------------------------------------------------------
    # Оптимальное управление (LP в узлах)
    # -------------------------------------------------------------------------

    def solve_optimal_control(self, K: int = 50):
        """
        1) Строим Φ(t_i)
        2) Считаем p(t_i) через Φ
        3) В каждом t_i решаем LP:
              max (G(t_i)^T p(t_i) - b)^T u
              s.t. Bu <= q
        """
        self._validate_shapes()
        self._compute_fundamental(K)
        self._solve_costate(K)

        self.U = np.zeros((self.m, K + 1))

        for i, t in enumerate(self.times):
            p_t = self.p_traj[:, i].reshape(-1, 1)     # (n,1)
            Gt = self.G(t)                              # (n,m)
            c_obj = (Gt.T @ p_t - self.b.reshape(-1, 1)).flatten()  # (m,)

            res = linprog(
                c_obj,
                A_ub=self.B,
                b_ub=self.q,
                bounds=(None, None)
            )

            self.U[:, i] = res.x if res.success else 0.0

    # -------------------------------------------------------------------------
    # Интерполяция управления
    # -------------------------------------------------------------------------

    def get_control(self, t: float, i: int) -> float:
        if self.times is None or self.U is None:
            raise RuntimeError("Сначала нужно вызвать solve_optimal_control()")
        return float(np.interp(t, self.times, self.U[i, :]))

    # -------------------------------------------------------------------------
    # x(t) через фундаментальную матрицу
    # -------------------------------------------------------------------------

    def compute_trajectory(self):
        """
        x'(t)=F(t)x+G(t)u+f(t)

        Через Φ(t):
            x(t)=Φ(t) ( x0 + ∫_0^t Φ(s)^{-1}(G(s)u(s)+f(s)) ds )
        Интеграл берём на сетке times (трапеции).
        """
        if self.times is None or self.U is None:
            raise RuntimeError("Сначала решите оптимальное управление (solve_optimal_control).")

        if self.Phi is None:
            K = len(self.times) - 1
            self._compute_fundamental(K)

        n = self.n
        t_grid = self.times
        Phi = self.Phi
        K = len(t_grid) - 1

        # r(t_i) = G(t_i)u(t_i) + f(t_i)
        r = np.zeros((K + 1, n))
        for i in range(K + 1):
            ti = t_grid[i]
            ui = self.U[:, i]
            r[i, :] = (self.G(ti) @ ui + self.ft(ti)).reshape(-1)

        # z(t_i) = Φ(t_i)^{-1} r(t_i)
        z = np.zeros((K + 1, n))
        for i in range(K + 1):
            z[i, :] = np.linalg.solve(Phi[i], r[i, :])

        # J_i = ∫_0^{t_i} z(s) ds (трапеции)
        J = np.zeros((K + 1, n))
        J[0, :] = 0.0
        for i in range(1, K + 1):
            dt = t_grid[i] - t_grid[i - 1]
            J[i, :] = J[i - 1, :] + 0.5 * dt * (z[i - 1, :] + z[i, :])

        # x(t_i) = Φ(t_i) (x0 + J_i)
        self.x_traj = np.zeros((n, K + 1))
        for i in range(K + 1):
            self.x_traj[:, i] = (Phi[i] @ (self.x0 + J[i, :])).reshape(-1)

    # -------------------------------------------------------------------------
    # Функционал качества
    # -------------------------------------------------------------------------

    def compute_objective(self) -> float:
        if self.x_traj is None or self.times is None:
            self.compute_trajectory()

        def integrand_x(t: float) -> float:
            x_interp = np.array([
                np.interp(t, self.times, self.x_traj[i, :])
                for i in range(self.n)
            ])
            return float(self.a @ x_interp)

        def integrand_u(t: float) -> float:
            u_t = np.array([self.get_control(t, i) for i in range(self.m)])
            return float(self.b @ u_t)

        Ob1 = quad(integrand_x, 0.0, self.T)[0]
        Ob2 = quad(integrand_u, 0.0, self.T)[0]

        self.ob_value = Ob1 + Ob2
        return self.ob_value

    # -------------------------------------------------------------------------
    # Визуализация
    # -------------------------------------------------------------------------

    def plot_controls(self):
        if self.U is None or self.times is None:
            raise RuntimeError("Нет данных об управлениях. Сначала вызовите solve_optimal_control().")

        plt.figure(figsize=(12, 6))
        for i in range(self.m):
            plt.plot(self.times, self.U[i, :], label=f'u{i + 1}(t)')
        plt.xlabel('Time')
        plt.ylabel('Control')
        plt.title('Optimal Controls')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_controls_to_bytes(self) -> bytes:
        if self.U is None or self.times is None:
            raise RuntimeError("Нет данных об управлениях. Сначала вызовите solve_optimal_control().")

        plt.figure(figsize=(12, 6))
        for i in range(self.m):
            plt.plot(self.times, self.U[i, :], label=f'u{i + 1}(t)')
        plt.xlabel('Time')
        plt.ylabel('Control')
        plt.title('Optimal Controls')
        plt.legend()
        plt.grid(True)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer.read()

    def plot_trajectories(self):
        if self.x_traj is None or self.times is None:
            raise RuntimeError("Нет данных о траекториях. Сначала вызовите compute_trajectory() или solve().")

        plt.figure(figsize=(12, 6))
        for i in range(self.n):
            plt.plot(self.times, self.x_traj[i, :], label=f'x{i + 1}(t)')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('System Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_trajectories_to_bytes(self) -> bytes:
        if self.x_traj is None or self.times is None:
            raise RuntimeError("Нет данных о траекториях. Сначала вызовите compute_trajectory() или solve().")

        plt.figure(figsize=(12, 6))
        for i in range(self.n):
            plt.plot(self.times, self.x_traj[i, :], label=f'x{i + 1}(t)')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('System Trajectories')
        plt.legend()
        plt.grid(True)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer.read()

    # -------------------------------------------------------------------------
    # Верхнеуровневое решение
    # -------------------------------------------------------------------------

    def solve(self, K: int = 50) -> Dict[str, Union[np.ndarray, float]]:
        self.solve_optimal_control(K)
        self.compute_trajectory()
        self.compute_objective()
        return {
            'controls': self.U,
            'trajectory': self.x_traj,
            'objective': self.ob_value
        }


# -------------------------------------------------------------------------
# Пример использования (тот же, что у тебя, но с фундаментальной матрицей)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    solver = OptimalControlSolverNd(
        T=1.0,
        M=1.0,
        N=1.0,
        n=1,
        m=1,
        x0=[0.0],

        F_func=[["2/(t+1)"]],
        G_func=[["t+1"]],
        a=np.array([-1.0]),
        b=np.array([0.0]),

        B=np.array([[-1.0], [1.0]], dtype=float),
        q=np.array([0.0, 1.0], dtype=float),

        ft_func=["0"]
    )

    results = solver.solve(K=50)
    print(f"Objective value: {results['objective']}")
    solver.plot_controls()
    solver.plot_trajectories()
