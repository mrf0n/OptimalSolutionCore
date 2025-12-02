from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import linprog
import sympy as sp
from typing import Optional, Tuple, Dict, Union, List


class OptimalControlSolverNd:
    """
    Класс для решения задачи оптимального управления для неавтономной линейной системы:

        x'(t) = F(t) x(t) + G(t) u(t) + f(t),

    с линейным функционалом качества:

        J = ∫_0^T (a^T x(t) + b^T u(t)) dt

    при линейных ограничениях на управление:

        B u(t) ≤ q.

    Параметры:
    ----------
    T : float
        Конечное время.
    M : float
        Не используется (оставлено для совместимости).
    N : float
        Не используется (оставлено для совместимости).
    n : int
        Размерность состояния.
    m : int
        Размерность управления.
    x0 : np.ndarray, shape (n,)
        Начальное состояние.
    F_func : List[List[str]]
        Матрица системы F(t) (n x n) в виде символьных выражений от t.
    G_func : List[List[str]]
        Матрица управления G(t) (n x m) в виде символьных выражений от t.
    a : np.ndarray, shape (n,)
        Вектор параметров в функционале качества (коэффициент при x).
    b : np.ndarray, shape (m,)
        Вектор параметров в функционале качества (коэффициент при u).
    B : np.ndarray, shape (k, m)
        Матрица ограничений на управление.
    q : np.ndarray, shape (k,)
        Вектор ограничений.
    ft_func : List[str]
        Вектор свободных членов f(t) как список выражений от t.
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

        self.T = T
        self.M = M
        self.N = N
        self.n = n  # размерность состояния
        self.m = m  # размерность управления

        # Начальное состояние
        self.x0 = np.array([5.0, 12.0], dtype=float) if x0 is None else np.array(x0, dtype=float)

        # Вектор параметров в функционале качества
        self.a = np.array([-1.0, 0.0], dtype=float) if a is None else np.array(a, dtype=float)

        # Вектор параметров при управлении
        self.b = np.array([0.0, 5.0], dtype=float) if b is None else np.array(b, dtype=float)

        # Матрица ограничений и вектор ограничений
        self.B = np.array(
            [[-1, 0],
             [0, -1],
             [2, 0],
             [0, 8],
             [2, -7]], dtype=float
        ) if B is None else np.array(B, dtype=float)

        self.q = np.array([0.0, 0.0, 5.0, 20.0, 0.0], dtype=float) if q is None else np.array(q, dtype=float)

        # --- Свободный член f(t) ---
        if ft_func is None:
            # По умолчанию f(t) = (t, 1, 0, ..., 0)
            def default_ft(t_val: float) -> np.ndarray:
                ft = np.zeros(self.n)
                ft[0] = t_val
                if self.n > 1:
                    ft[1] = 1.0
                return ft

            self.ft = default_ft
        else:
            self.ft = self._create_ft_func(ft_func)

        # --- Матрица управления G(t) ---
        if G_func is None:
            # По умолчанию все строки одинаковые: первый столбец = t, остальные нули
            def default_gt(t_val: float) -> np.ndarray:
                gt_matrix = np.zeros((self.n, self.m))
                for i in range(self.n):
                    gt_matrix[i, 0] = t_val
                return gt_matrix

            self.G = default_gt
        else:
            self.G = self._create_matrix_func(G_func)

        # --- Матрица системы F(t) ---
        if F_func is None:
            # По умолчанию диагональная матрица с F_ii = t
            def default_Ft(t_val: float) -> np.ndarray:
                ft_matrix = np.zeros((self.n, self.n))
                for i in range(self.n):
                    ft_matrix[i, i] = t_val
                return ft_matrix

            self.F = default_Ft
        else:
            self.F = self._create_matrix_func(F_func)

        # Результаты
        self.times: Optional[np.ndarray] = None  # сетка по времени
        self.U: Optional[np.ndarray] = None      # оптимальные управления (m, K+1)
        self.x_traj: Optional[np.ndarray] = None # траектория состояния (n, K+1)
        self.p_traj: Optional[np.ndarray] = None # траектория сопряжённых переменных (n, K+1)
        self.ob_value: Optional[float] = None    # значение функционала

    # -------------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СОЗДАНИЯ F(t), G(t), f(t)
    # -------------------------------------------------------------------------

    def _create_ft_func(self, expressions: List[str]):
        """
        Создаёт вектор-функцию f(t) из списка выражений.
        """
        t = sp.symbols('t')
        funcs = [sp.lambdify(t, sp.sympify(expr), 'numpy') for expr in expressions]

        def ft(t_value: float) -> np.ndarray:
            ft_vec = np.zeros(self.n)
            for i in range(min(len(funcs), self.n)):
                ft_vec[i] = funcs[i](t_value)
            return ft_vec

        return ft

    def _create_matrix_func(self, expressions: List[List[str]]):
        """
        Создаёт матричную функцию M(t) из списка символьных выражений.

        expressions: список списков строк (rows x cols)
        """

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
    # ПРОВЕРКА РАЗМЕРНОСТЕЙ
    # -------------------------------------------------------------------------

    def _validate_shapes(self):
        """
        Проверка согласованности размеров.
        """
        assert self.x0.shape == (self.n,), f"x0 must be of shape ({self.n},)"
        assert self.a.shape == (self.n,), f"a must be of shape ({self.n},)"
        assert self.b.shape == (self.m,), f"b must be of shape ({self.m},)"
        assert self.B.shape[1] == self.m, f"B must have {self.m} columns"
        assert self.q.shape[0] == self.B.shape[0], "B and q must have same number of rows"

        # Проверяем матрицы F(t), G(t) на одном значении t
        F0 = self.F(0.0)
        G0 = self.G(0.0)

        assert F0.shape == (self.n, self.n), f"F(t) must be ({self.n}, {self.n})"
        assert G0.shape == (self.n, self.m), f"G(t) must be ({self.n}, {self.m})"

        # Проверка свободного члена
        f0 = self.ft(0.0)
        assert f0.shape == (self.n,), f"f(t) must be vector of length {self.n}"

    # -------------------------------------------------------------------------
    # ВЫЧИСЛЕНИЕ СОПРЯЖЁННОЙ ПЕРЕМЕННОЙ p(t) И ОПТИМАЛЬНОГО УПРАВЛЕНИЯ
    # -------------------------------------------------------------------------

    def _solve_costate(self, K: int):
        """
        Решение p'(t) = -F(t)^T p + a,  p(T)=0
        через замену τ = T - t → dp/dτ = F(T-τ)^T p + a.
        """

        self.times = np.linspace(0.0, self.T, K + 1)

        def costate_ode(tau, p_vec):
            p = np.array(p_vec)
            t = self.T - tau
            Ft = self.F(t)
            return Ft.T @ p - self.a

        taus_eval = np.linspace(0.0, self.T, K + 1)

        sol = solve_ivp(
            costate_ode,
            (0.0, self.T),
            y0=np.zeros(self.n),
            t_eval=taus_eval,
            rtol=1e-6,
            atol=1e-8
        )

        if not sol.success:
            raise RuntimeError("Ошибка при решении уравнения для p(t)")

        taus = sol.t  # (K+1,)
        p_tau = sol.y  # (n, K+1)

        # Теперь восстанавливаем p(t) на сетке t_i, используя τ=T−t
        self.p_traj = np.zeros((self.n, K + 1))

        for i, t in enumerate(self.times):
            tau = self.T - t
            # интерполируем каждую компоненту отдельно
            for j in range(self.n):
                self.p_traj[j, i] = np.interp(tau, taus, p_tau[j, :])

    def solve_optimal_control(self, K: int = 50):
        """
        Решение задачи оптимального управления:

        1. Решаем уравнение для p(t).
        2. Для каждого момента времени t_i решаем линейную задачу:
                maximize   (G(t_i)^T p(t_i) - b)^T u
                subject to B u <= q

           реализованную через scipy.optimize.linprog (минимизацию -c^T u).
        """
        self._validate_shapes()

        # 1. Решаем для p(t)
        self._solve_costate(K)

        # 2. Дискретизируем управление
        self.U = np.zeros((self.m, K + 1))

        for i, t in enumerate(self.times):
            p_t = self.p_traj[:, i].reshape(-1, 1)  # (n, 1)
            Gt = self.G(t)                          # (n, m)
            # Коэффициент при u в Гамильтониане (как в исходном коде):
            # G^T p - b
            c_obj = (Gt.T @ p_t - self.b.reshape(-1, 1)).flatten()  # (m,)

            # Максимизируем c_obj^T u  ⇔  минимизируем (-c_obj)^T u
            res = linprog(-c_obj, A_ub=self.B, b_ub=self.q, bounds=(None, None))

            if res.success:
                self.U[:, i] = res.x
            else:
                # Если LP не решилась — ставим нулевое управление
                self.U[:, i] = 0.0

    # -------------------------------------------------------------------------
    # ТРАЕКТОРИЯ СИСТЕМЫ
    # -------------------------------------------------------------------------

    def get_control(self, t: float, i: int) -> float:
        """
        Интерполяция i-го управления в момент времени t по дискретной сетке.
        """
        if self.times is None or self.U is None:
            raise RuntimeError("Сначала нужно вызвать solve_optimal_control()")

        return float(np.interp(t, self.times, self.U[i, :]))

    def compute_trajectory(self):
        """
        Вычисление траектории системы:

            x'(t) = F(t) x(t) + G(t) u(t) + f(t)
        """

        if self.times is None or self.U is None:
            raise RuntimeError("Сначала нужно решить задачу оптимального управления (solve_optimal_control).")

        def ode_func(t, x_vec):
            x = np.array(x_vec)
            u = np.array([self.get_control(t, i) for i in range(self.m)])  # (m,)
            Ft = self.F(t)
            Gt = self.G(t)
            ft_val = self.ft(t)
            dx = Ft @ x + Gt @ u + ft_val
            return dx

        sol_x = solve_ivp(
            ode_func,
            (0.0, self.T),
            y0=self.x0,
            t_eval=self.times,
            rtol=1e-6,
            atol=1e-8
        )

        if not sol_x.success:
            raise RuntimeError("Не удалось решить систему для x(t)")

        self.x_traj = sol_x.y  # (n, K+1)

    # -------------------------------------------------------------------------
    # ВЫЧИСЛЕНИЕ ФУНКЦИОНАЛА КАЧЕСТВА
    # -------------------------------------------------------------------------

    def compute_objective(self) -> float:
        """
        Вычисление целевого функционала:

            J = ∫_0^T (a^T x(t) + b^T u(t)) dt
        """

        if self.x_traj is None or self.times is None:
            self.compute_trajectory()

        # Интеграл по x(t)
        def integrand_x(t: float) -> float:
            x_interp = np.array([
                np.interp(t, self.times, self.x_traj[i, :])
                for i in range(self.n)
            ])
            return float(self.a @ x_interp)

        # Интеграл по u(t)
        def integrand_u(t: float) -> float:
            u_t = np.array([self.get_control(t, i) for i in range(self.m)])
            return float(self.b @ u_t)

        Ob1 = quad(integrand_x, 0.0, self.T)[0]
        Ob2 = quad(integrand_u, 0.0, self.T)[0]

        self.ob_value = Ob1 + Ob2
        return self.ob_value

    # -------------------------------------------------------------------------
    # ВИЗУАЛИЗАЦИЯ
    # -------------------------------------------------------------------------

    def plot_controls(self):
        """
        Визуализация оптимальных управлений u_i(t).
        """
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
        """
        Возвращает график управлений как PNG-байты.
        """
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
        """
        Визуализация траекторий состояния x_i(t).
        """
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
        """
        Возвращает график траекторий состояния как PNG-байты.
        """
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
    # ВЕРХНЕУРОВНЕВЫЙ МЕТОД
    # -------------------------------------------------------------------------

    def solve(self, K: int = 50) -> Dict[str, Union[np.ndarray, float]]:
        """
        Полное решение задачи:

            1. Поиск оптимального управления.
            2. Вычисление траекторий.
            3. Вычисление функционала качества.
        """
        self.solve_optimal_control(K)
        self.compute_trajectory()
        self.compute_objective()

        return {
            'controls': self.U,
            'trajectory': self.x_traj,
            'objective': self.ob_value
        }


# Пример использования
if __name__ == "__main__":
    # Пример: 2-мерное состояние, 2 управления, F(t), G(t) могут быть неавтономными
    solver = OptimalControlSolverNd(
        T=6.0,
        M=1.0,
        N=5.0,
        n=2,
        m=2,
        x0=[5.0, 12.0],
        # Здесь F(t) автономна (константна), но можно задать зав-т от t, например: "0.5 + 0.1*sin(t)"
        F_func=[
            ["0.5", "-0.02"],
            ["-0.02", "0.4"]
        ],
        # G(t) тоже может быть неавтономной, например "0.3 + 0.1*t"
        G_func=[
            ["0.3", "0.3"],
            ["0.2", "0.2"]
        ],
        a=np.array([-1.0, 0.0]),
        b=np.array([0.0, 5.0]),
        B=np.array([
            [-1, 0],
            [0, -1],
            [2, 0],
            [0, 8],
            [2, -7]
        ], dtype=float),
        q=np.array([0.0, 0.0, 5.0, 20.0, 0.0], dtype=float),
        ft_func=["t", "1"]
    )

    results = solver.solve(K=50)
    print(f"Objective value: {results['objective']}")
    solver.plot_controls()
    solver.plot_trajectories()
