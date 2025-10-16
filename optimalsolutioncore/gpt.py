from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict, Union, List

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import linprog

ArrayLike = Union[np.ndarray, List[float], List[List[float]]]

# -----------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ПАРСЕРЫ ДЛЯ ФУНКЦИЙ ОТ ВРЕМЕНИ
# -----------------------------------------------------------------------------

def _as_callable_vector(exprs: Union[Callable[[float], np.ndarray], List[str], np.ndarray, None], n: int,
                        default: Optional[np.ndarray] = None) -> Callable[[float], np.ndarray]:
    """Возвращает f(t) -> (n,) из разных типов входа:
    - callable(t) -> array
    - список строк длины n с выражениями на sympy ('t' допустим)
    - константный np.ndarray формы (n,)
    - None -> default (или нули)
    """
    if callable(exprs):
        return exprs  # type: ignore

    if exprs is None:
        vec = np.zeros(n) if default is None else np.asarray(default, dtype=float).reshape(n)
        return lambda t: vec

    if isinstance(exprs, np.ndarray):
        vec = np.asarray(exprs, dtype=float).reshape(n)
        return lambda t: vec

    # Считаем, что это список символьных выражений длины >= n
    t = sp.symbols('t')
    funcs = [sp.lambdify(t, sp.sympify(s), 'numpy') for s in exprs][:n]  # type: ignore

    def f(tval: float) -> np.ndarray:
        out = np.zeros(n)
        for i, fun in enumerate(funcs):
            out[i] = float(fun(tval))
        return out

    return f


def _as_callable_matrix(exprs: Union[Callable[[float], np.ndarray], List[List[str]], np.ndarray, None],
                        shape: Tuple[int, int], default: Optional[np.ndarray] = None) -> Callable[[float], np.ndarray]:
    """Возвращает M(t) -> (r,c) из:
    - callable(t) -> array
    - матрицы строковых выражений на sympy ('t' допустим)
    - константного np.ndarray
    - None -> default (или нули)
    """
    r, c = shape

    if callable(exprs):
        return exprs  # type: ignore

    if exprs is None:
        mat = np.zeros((r, c)) if default is None else np.asarray(default, dtype=float).reshape(r, c)
        return lambda t: mat

    if isinstance(exprs, np.ndarray):
        mat = np.asarray(exprs, dtype=float).reshape(r, c)
        return lambda t: mat

    # Считаем, что это список списков строковых выражений r x c
    t = sp.symbols('t')
    if len(exprs) != r or any(len(row) != c for row in exprs):  # type: ignore
        raise ValueError("Matrix expression shape mismatch")

    func_mat = [[sp.lambdify(t, sp.sympify(s), 'numpy') for s in row] for row in exprs]  # type: ignore

    def f(tval: float) -> np.ndarray:
        out = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                out[i, j] = float(func_mat[i][j](tval))
        return out

    return f


# -----------------------------------------------------------------------------
# ОСНОВНОЙ КЛАСС РЕШЕНИЯ ЛИНЕЙНОЙ НЕАВТОНОМНОЙ ЗАДАЧИ С ЛИНЕЙНОЙ СТОИМОСТЬЮ
# -----------------------------------------------------------------------------

@dataclass
class OptimalControlSolverNd:
    """
    Линейная неавтономная система с линейным интегральным функционалом и линейными
    ограничениями на управление, решаемая через PMP (сопряжённая система) + LP во времени.

    Динамика:       x˙(t) = F(t) x(t) + G(t) u(t) + f(t),     x(0) = x0.
    Стоимость:      J = ∫_0^T [ a(t)^T x(t) + b(t)^T u(t) ] dt  (без терминального удельного члена).
    Ограничения:    B(t) u(t) ≤ q(t) (по умолчанию константные B, q).

    PMP =>
      p˙(t) = -F(t)^T p(t) - a(t),    p(T) = p_T (обычно 0 при свободном x(T) и без терминальной стоимости)
      u*(t) = argmin_{u: B u≤q}  (b(t) + G(t)^T p(t))^T u  (л.п. в каждый момент)

    Атрибуты можно задавать как:
      - callable(t) -> np.ndarray
      - константные np.ndarray
      - строковые выражения (sympy) для векторов/матриц на переменной t.

    Примечание:
      Если вам нужна терминальная стоимость 1/2 x(T)^T S x(T) или трекинг — это уже LTV-LQR; здесь реализована
      линейная стоимость (удобно для box/политопных ограничений). Для LQR используйте обратную интеграцию
      уравнения Риккати.
    """

    T: float = 6.0
    n: int = 2               # размерность состояния
    m: int = 2               # размерность управления

    # Начальное состояние
    x0: ArrayLike = field(default_factory=lambda: np.array([5.0, 12.0]))

    # Входные описатели (см. конструктор): допускают callable / ndarray / sympy-строки
    F_func: Union[Callable[[float], np.ndarray], List[List[str]], np.ndarray, None] = None  # (n x n)
    G_func: Union[Callable[[float], np.ndarray], List[List[str]], np.ndarray, None] = None  # (n x m)
    a_func: Union[Callable[[float], np.ndarray], List[str], np.ndarray, None] = None       # (n,)
    b_func: Union[Callable[[float], np.ndarray], List[str], np.ndarray, None] = None       # (m,)
    f_func: Union[Callable[[float], np.ndarray], List[str], np.ndarray, None] = None       # (n,)

    # Ограничения Bu <= q (по умолчанию константные) — допускают также time-varying callable
    B_func: Union[Callable[[float], np.ndarray], List[List[str]], np.ndarray, None] = None  # (k x m)
    q_func: Union[Callable[[float], np.ndarray], List[str], np.ndarray, None] = None        # (k,)

    # Терминальное условие для сопряжённой (по умолчанию p(T)=0)
    p_T: Optional[np.ndarray] = None  # (n,)

    # Число узлов дискретизации по времени (K+1 точек, включая 0 и T)
    K: int = 100

    # Внутренние поля, заполняются при solve()
    times: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None          # (m, K+1)
    X: Optional[np.ndarray] = None          # (n, K+1)
    P: Optional[np.ndarray] = None          # (n, K+1)
    objective_value: Optional[float] = None

    # приватные callables после нормализации входов
    _F: Callable[[float], np.ndarray] = field(init=False, repr=False)
    _G: Callable[[float], np.ndarray] = field(init=False, repr=False)
    _a: Callable[[float], np.ndarray] = field(init=False, repr=False)
    _b: Callable[[float], np.ndarray] = field(init=False, repr=False)
    _f: Callable[[float], np.ndarray] = field(init=False, repr=False)
    _B: Callable[[float], np.ndarray] = field(init=False, repr=False)
    _q: Callable[[float], np.ndarray] = field(init=False, repr=False)

    # ------------------------------------------------------------------
    # ИНИЦИАЛИЗАЦИЯ
    # ------------------------------------------------------------------
    def __post_init__(self):
        # Нормализация входов к функциям от t
        self._F = _as_callable_matrix(self.F_func, (self.n, self.n))
        self._G = _as_callable_matrix(self.G_func, (self.n, self.m),
                                      default=np.zeros((self.n, self.m)))
        self._a = _as_callable_vector(self.a_func, self.n, default=np.zeros(self.n))
        self._b = _as_callable_vector(self.b_func, self.m, default=np.zeros(self.m))
        self._f = _as_callable_vector(self.f_func, self.n, default=np.zeros(self.n))

        # Константные ограничения по умолчанию: u ≥ 0, простая коробка (можно переопределить)
        if self.B_func is None and self.q_func is None:
            B = np.vstack([np.eye(self.m), -np.eye(self.m)])  # u ≤ +∞ и -u ≤ 0 => u ≥ 0 при q=[+∞, 0]? Скорректируем ниже
            q = np.hstack([np.full(self.m, np.inf), np.zeros(self.m)])
            # По умолчанию без ограничений лучше пустой полигон не делать; зададим просто отсутствующие ограничения
            B = np.zeros((0, self.m))
            q = np.zeros(0)
            self._B = _as_callable_matrix(B, (0, self.m))
            self._q = _as_callable_vector(q, 0)
        else:
            # Пользовательские B, q (могут быть функциями/константами/символьными)
            # Если передали ndarray/list — размер k x m и k
            # Если None для одного из них — считаем его нулевым размера, чтобы не ограничивать
            # (Если один задан, второй тоже обязан быть задан)
            if (self.B_func is None) ^ (self.q_func is None):
                raise ValueError("If you provide B_func or q_func, you must provide both.")
            # Здесь пока не знаем k, но если это ndarray, то возьмём k из него, иначе попросим у пользователя согласованные формы
            if isinstance(self.B_func, np.ndarray):
                k = self.B_func.shape[0]
            elif callable(self.B_func):
                # вызовем при t=0 чтобы узнать k
                k = np.asarray(self.B_func(0.0)).shape[0]
            else:
                k = len(self.B_func)  # type: ignore
            self._B = _as_callable_matrix(self.B_func, (k, self.m), default=np.zeros((0, self.m)))
            self._q = _as_callable_vector(self.q_func, k, default=np.zeros(0))

        self.x0 = np.asarray(self.x0, dtype=float).reshape(self.n)
        if self.p_T is None:
            self.p_T = np.zeros(self.n)
        else:
            self.p_T = np.asarray(self.p_T, dtype=float).reshape(self.n)

    # ------------------------------------------------------------------
    # ОСНОВНЫЕ ЭТАПЫ РЕШЕНИЯ
    # ------------------------------------------------------------------
    def _integrate_costate_backward(self) -> Tuple[np.ndarray, np.ndarray]:
        """Интеграция сопряжённой системы назад по времени:
            p˙(t) = -F(t)^T p(t) - a(t),   p(T) = p_T.
        Возвращает (t_grid, P) где P.shape = (n, K+1) в возрастающем времени.
        """
        t_grid = np.linspace(0.0, self.T, self.K + 1)
        t_rev = t_grid[::-1]

        def ode(t, p):
            return -self._F(t).T @ p - self._a(t)

        sol = solve_ivp(ode, (self.T, 0.0), self.p_T, t_eval=t_rev, rtol=1e-7, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"Costate integration failed: {sol.message}")

        P = sol.y[:, ::-1]  # вернуть в порядке времени 0..T
        return t_grid, P

    def _solve_instant_lp(self, t: float, c_u: np.ndarray) -> np.ndarray:
        """Решение одномоментной ЛП:  minimize c_u^T u  s.t. B(t) u <= q(t).
        Если ограничений нет (k=0), вернём нули.
        """
        B = self._B(t)
        q = self._q(t)
        k = B.shape[0]
        if k == 0:
            return np.zeros(self.m)

        # SciPy linprog минимизирует c^T u
        res = linprog(c=c_u, A_ub=B, b_ub=q, bounds=(None, None), method='highs')
        if not res.success:
            # В случае неуспеха вернём ближайшую допустимую точку (попытаемся ноль)
            # или поднимем исключение — здесь выберем ноль, чтобы не ронять ход интеграции
            return np.zeros(self.m)
        return res.x

    def _compute_controls_on_grid(self, t_grid: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Строит U[:,i] на сетке t_grid из условия минимума гамильтониана.
        c_u(t) = b(t) + G(t)^T p(t)
        """
        U = np.zeros((self.m, t_grid.size))
        for i, t in enumerate(t_grid):
            c_u = self._b(t) + self._G(t).T @ P[:, i]
            U[:, i] = self._solve_instant_lp(t, c_u)
        return U

    def _u_interp(self, t: float) -> np.ndarray:
        """Линейная интерполяция управления между узлами сетки."""
        if self.times is None or self.U is None:
            raise RuntimeError("Call solve() first")
        u = np.zeros(self.m)
        # По каждой компоненте своя интерполяция
        for j in range(self.m):
            u[j] = np.interp(t, self.times, self.U[j, :])
        return u

    def _integrate_state_forward(self, t_grid: np.ndarray) -> np.ndarray:
        """Интегрируем x˙ = F(t)x + G(t)u(t) + f(t) с интерполированным u(t)."""
        def ode(t, x):
            return self._F(t) @ x + self._G(t) @ self._u_interp(t) + self._f(t)

        sol = solve_ivp(ode, (t_grid[0], t_grid[-1]), self.x0, t_eval=t_grid, rtol=1e-7, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"State integration failed: {sol.message}")
        return sol.y

    # ------------------------------------------------------------------
    # ПОДСЧЁТ ЦЕЛЕВОГО ФУНКЦИОНАЛА
    # ------------------------------------------------------------------
    def _trapz_objective(self, t_grid: np.ndarray, X: np.ndarray, U: np.ndarray) -> float:
        """Интегрирование стоимости J = ∫ (a^T x + b^T u) dt через трапеции."""
        # вычислим значения подынтегральной функции на узлах
        vals = np.zeros_like(t_grid)
        for i, t in enumerate(t_grid):
            vals[i] = float(self._a(t) @ X[:, i] + self._b(t) @ U[:, i])
        # трапецией
        return float(np.trapz(vals, t_grid))

    # ------------------------------------------------------------------
    # ПУБЛИЧНЫЙ ИНТЕРФЕЙС
    # ------------------------------------------------------------------
    def solve(self) -> Dict[str, Union[np.ndarray, float]]:
        """Полный цикл решения: p(t) назад → u(t) на сетке → x(t) вперёд → J."""
        t_grid, P = self._integrate_costate_backward()
        U = self._compute_controls_on_grid(t_grid, P)
        X = self._integrate_state_forward(t_grid)
        J = self._trapz_objective(t_grid, X, U)

        # сохранить результаты
        self.times = t_grid
        self.P = P
        self.U = U
        self.X = X
        self.objective_value = J

        return {
            'times': t_grid,
            'costate': P,
            'controls': U,
            'trajectory': X,
            'objective': J,
        }

    # ------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ
    # ------------------------------------------------------------------
    def plot_controls(self):
        if self.times is None or self.U is None:
            raise RuntimeError("Call solve() first")
        plt.figure(figsize=(10, 4))
        for j in range(self.m):
            plt.plot(self.times, self.U[j, :], label=f"u{j+1}(t)")
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Optimal controls (non-autonomous)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_trajectory(self):
        if self.times is None or self.X is None:
            raise RuntimeError("Call solve() first")
        plt.figure(figsize=(10, 4))
        for i in range(self.n):
            plt.plot(self.times, self.X[i, :], label=f"x{i+1}(t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("State trajectory")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_costate(self):
        if self.times is None or self.P is None:
            raise RuntimeError("Call solve() first")
        plt.figure(figsize=(10, 4))
        for i in range(self.n):
            plt.plot(self.times, self.P[i, :], label=f"p{i+1}(t)")
        plt.xlabel("t")
        plt.ylabel("p")
        plt.title("Costate trajectory")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_controls_to_bytes(self) -> bytes:
        if self.times is None or self.U is None:
            raise RuntimeError("Call solve() first")
        plt.figure(figsize=(10, 4))
        for j in range(self.m):
            plt.plot(self.times, self.U[j, :], label=f"u{j+1}(t)")
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Optimal controls (non-autonomous)")
        plt.grid(True)
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.read()

    def plot_trajectory_to_bytes(self) -> bytes:
        if self.times is None or self.X is None:
            raise RuntimeError("Call solve() first")
        plt.figure(figsize=(10, 4))
        for i in range(self.n):
            plt.plot(self.times, self.X[i, :], label=f"x{i+1}(t)")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("State trajectory")
        plt.grid(True)
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.read()


# -----------------------------------------------------------------------------
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (раскомментировать для запуска)
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Неавтономный пример (2 состояния, 2 управления)
#     n, m, T = 2, 2, 6.0
#
#     F_expr = [["0.5 + 0.1*sin(t)", "-0.02"],
#               ["0.01", "0.4 + 0.05*cos(t)"]]
#     G_expr = [["0.3 + 0.1*sin(t)", "0.2"],
#               ["0.2", "0.2 + 0.1*cos(t)"]]
#     a_expr = ["-1", "0.1*sin(t)"]
#     b_expr = ["0.1*t", "5"]
#     f_expr = ["t", "1"]
#
#     # Политопные ограничения на u: -1 ≤ u1 ≤ 2, 0 ≤ u2 ≤ 3
#     B = np.array([[ 1, 0],   #  u1 ≤ 2
#                   [-1, 0],   # -u1 ≤ 1  => u1 ≥ -1
#                   [ 0, 1],   #  u2 ≤ 3
#                   [ 0,-1]])  # -u2 ≤ 0  => u2 ≥ 0
#     q = np.array([2, 1, 3, 0])
#
#     solver = OptimalControlSolverNd(
#         T=T, n=n, m=m, x0=[5, 12],
#         F_func=F_expr, G_func=G_expr,
#         a_func=a_expr, b_func=b_expr, f_func=f_expr,
#         B_func=B, q_func=q,
#         K=200,
#     )
#
#     res = solver.solve()
#     print("J =", res['objective'])
#     solver.plot_controls()
#     solver.plot_trajectory()
#     solver.plot_costate()
