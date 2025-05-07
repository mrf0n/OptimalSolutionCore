from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad, solve_ivp
from scipy.optimize import linprog
import sympy as sp
from typing import Optional, Tuple, Dict, Union, List


class OptimalControlSolverNd:
    """
    Класс для решения задачи оптимального управления с матрицами произвольного размера

    Параметры:
    T: float - конечное время
    M: float - параметр M (опционально)
    N: float - параметр N (опционально)
    n: int - размерность состояния
    m: int - размерность управления
    x0: np.array - начальное состояние (n,)
    F_func: List[List[str]] - матрица системы (n x n)
    G_func: List[List[str]] - матрица управления (n x m)
    a: np.array - вектор параметров (n,)
    b: np.array - вектор параметров (m,)
    B: np.array - матрица ограничений управления (k x m)
    q: np.array - вектор ограничений управления (k,)
    """

    def __init__(self,
                 T: float = 6.0,
                 M: float = 1.0,
                 N: float = 5.0,
                 n: int = 2,
                 m: int = 2,
                 x0: Optional[np.ndarray] = None,
                 F_func: List[List[str]] = None,
                 G_func: List[List[str]] = None,
                 a: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None,
                 B: Optional[np.ndarray] = None,
                 q: Optional[np.ndarray] = None,
                 ft_func: Optional[List[str]] = None):

        self.T = T
        self.M = M
        self.N = N
        self.n = n  # Размерность состояния
        self.m = m  # Размерность управления
        self.ft = None
        # Инициализация параметров по умолчанию
        self.x0 = np.array([5.0, 12.0]) if x0 is None else np.array(x0, dtype=float)

        # Матрица системы (n x n)
        self.F = None

        # Матрица управления (n x m)
        self.G = None

        # Вектор параметров (n,)
        self.a = np.array([-1.0, 0.0]) if a is None else np.array(a, dtype=float)

        # Вектор параметров (m,)
        self.b = np.array([0.0, 5.0]) if b is None else np.array(b, dtype=float)

        # Матрица ограничений (k x m)
        self.B = np.array([[-1, 0], [0, -1], [2, 0], [0, 8], [2, -7]], dtype=float) if B is None else np.array(B,
                                                                                                               dtype=float)

        # Вектор ограничений (k,)
        self.q = np.array([0.0, 0.0, 5.0, 20.0, 0.0], dtype=float) if q is None else np.array(q, dtype=float)

        # Функция для вычисления ft (по умолчанию [t, 1, 0, ...])
        if ft_func is None:
            def default_ft(t):
                ft = np.zeros(self.n)
                ft[0] = t
                if self.n > 1:
                    ft[1] = 1.0
                return ft

            self.ft = default_ft
        else:
            self.ft = self.create_ft_func(ft_func)

        # Функция для вычисления G(t) (по умолчанию [t, 0, 0] [t, 0, 0] [t, 0, 0])
        if G_func is None:
            def default_gt(t):
                # Создаём нулевую матрицу n x m
                gt_matrix = np.zeros((self.n, self.m))
                for i in range(self.n):
                    gt_matrix[i, 0] = t
                return gt_matrix

            self.G = default_gt
        else:
            self.G = self.create_Gt_Ft_func(G_func)

        # Функция для вычисления F(t) (по умолчанию [t, 0, 0] [0, t, 0] [0, 0, t])
        if F_func is None:
            def default_ft(t):
                # Создаём нулевую матрицу n x n
                ft_matrix = np.zeros((self.n, self.n))
                # Заполняем диагональ
                for i in range(self.n):
                    ft_matrix[i, i] = t  # Главная диагональ = t
                return ft_matrix

            self.F = default_ft
        else:
            self.F = self.create_Gt_Ft_func(F_func)

        # Кэши
        self.pt_cache = {}
        self.Yt_cache = {}

        # Результаты
        self.times = None
        self.U = None  # Теперь будет (m, K+1)
        self.x_traj = None  # (n, K+1)
        self.ob_value = None

    def create_ft_func(self, expressions: List[str]):
        # Создаем переменную t для использования в выражениях
        t = sp.symbols('t')

        # Парсим и создаем функции для каждого выражения
        funcs = [sp.lambdify(t, sp.sympify(expr), 'numpy') for expr in expressions]

        def ft(t_value):
            ft = np.zeros(self.n)
            for i in range(min(len(funcs), self.n)):
                ft[i] = funcs[i](t_value)
            return ft

        return ft

    def create_Gt_Ft_func(self, expressions: List[List[str]]):
        """
        Создает матричную функцию из списка символьных выражений.
        Размерность матрицы определяется по входному списку.

        Args:
            expressions: Список списков строк с математическими выражениями.
                        Например: [["t", "0"], ["1", "sin(t)"]] создаст матрицу 2x2.

        Returns:
            Функция gt(t), возвращающая матрицу размерности len(expressions) x len(expressions[0])
        """
        t = sp.symbols('t')

        # Проверка корректности входных данных
        if not expressions or not all(expressions):
            raise ValueError("Expressions list must contain at least one row with at least one expression")

        # Определяем размерность матрицы
        rows = len(expressions)
        cols = len(expressions[0]) if rows > 0 else 0

        # Проверка, что все строки имеют одинаковую длину
        if not all(len(row) == cols for row in expressions):
            raise ValueError("All expression rows must have the same number of columns")

        # Создаем матрицу лямбда-функций
        func_matrix = []
        for row_exprs in expressions:
            row_funcs = [sp.lambdify(t, sp.sympify(expr), 'numpy') for expr in row_exprs]
            func_matrix.append(row_funcs)

        def gt(t_value):
            # Создаем нулевую матрицу вычисленного размера
            gt_mat = np.zeros((rows, cols))

            # Заполняем матрицу значениями
            for i in range(rows):
                for j in range(cols):
                    try:
                        gt_mat[i, j] = func_matrix[i][j](t_value)
                    except (TypeError, ValueError, ZeroDivisionError):
                        gt_mat[i, j] = 0.0  # Запасной вариант при ошибках вычисления
                    except Exception as e:
                        print(f"Warning: Unexpected error computing element ({i},{j}): {str(e)}")
                        gt_mat[i, j] = 0.0
            return gt_mat

        return gt

    def _validate_shapes(self):
        """Проверка согласованности размеров матриц"""
        # assert self.F.shape == (self.n, self.n), f"F must be ({self.n}, {self.n})"
        # assert self.G.shape == (self.n, self.m), f"G must be ({self.n}, {self.m})"
        assert self.x0.shape == (self.n,), f"x0 must be ({self.n},)"
        assert self.a.shape == (self.n,), f"a must be ({self.n},)"
        assert self.b.shape == (self.m,), f"b must be ({self.m},)"
        assert self.B.shape[1] == self.m, f"B must have {self.m} columns"
        assert self.q.shape[0] == self.B.shape[0], "B and q must have same number of rows"

    def compute_Yt(self, t: float) -> np.ndarray:
        """Вычисление матричной экспоненты Y(t) = exp(-F^T t)"""
        return expm(-self.F(t).T * t)

    def compute_A2(self) -> np.ndarray:
        """Вычисление интегральной части A2"""

        def ode_func(t: float, y: np.ndarray) -> np.ndarray:
            Cpts = self.compute_Yt(self.T) @ self.compute_Yt(-t)
            return Cpts @ self.a

        sol = solve_ivp(ode_func, [0, self.T], np.zeros(self.n), rtol=1e-6, atol=1e-8)
        return sol.y[:, -1].reshape(-1, 1)

    def compute_pt(self, t: float) -> np.ndarray:
        """Вычисление pt(t) с кэшированием"""
        if t in self.pt_cache:
            return self.pt_cache[t]

        Yt_val = self.compute_Yt(t)
        term1 = Yt_val @ self.c

        def integrand(s: float, y: np.ndarray) -> np.ndarray:
            Cpts = self.compute_Yt(t) @ self.compute_Yt(-s)
            return (Cpts @ self.a).flatten()

        sol = solve_ivp(integrand, [0, t], np.zeros(self.n), rtol=1e-6, atol=1e-8)
        integral = sol.y[:, -1].reshape(-1, 1)

        pt_val = term1 + integral
        self.pt_cache[t] = pt_val
        return pt_val

    def compute_Gp1(self, t: float) -> np.ndarray:
        """Вычисление Gp1(t) = G^T @ pt(t) - b"""
        pt_val = self.compute_pt(t)
        return self.G(t).T @ pt_val - self.b.reshape(-1, 1)

    def solve_optimal_control(self, K: int = 50):
        """Решение задачи оптимального управления"""
        self._validate_shapes()

        # Вычисление констант
        YT = self.compute_Yt(self.T)
        A1 = np.linalg.inv(YT)
        A2 = self.compute_A2()
        self.c = -A1 @ A2

        # Дискретизация времени
        self.times = np.linspace(0, self.T, K + 1)

        # Оптимизация управления (теперь для m управлений)
        self.U = np.zeros((self.m, K + 1))  # Матрица управлений

        for i, t in enumerate(self.times):
            c_obj = self.compute_Gp1(t).flatten()
            res = linprog(-c_obj, A_ub=self.B, b_ub=self.q, bounds=(None, None))

            if res.success:
                self.U[:, i] = res.x
            else:
                self.U[:, i] = 0.0

    def get_control(self, t: float, i: int) -> float:
        """Получение i-го управления в момент времени t"""
        return np.interp(t, self.times, self.U[i, :])

    def compute_trajectory(self):
        """Вычисление траектории системы"""

        def ode_func(t: float, x: np.ndarray) -> np.ndarray:
            u = np.array([self.get_control(t, i) for i in range(self.m)])
            ft_val = self.ft(t)
            dx = self.F(t) @ x + self.G(t) @ u + ft_val
            return dx

        sol = solve_ivp(ode_func, [0, self.T], self.x0, t_eval=self.times, rtol=1e-6, atol=1e-8)
        self.x_traj = sol.y

    def compute_objective(self) -> float:
        """Вычисление целевого функционала"""
        if self.x_traj is None:
            self.compute_trajectory()

        # Первый интеграл: a^T @ x(t)
        def integrand1(t: float) -> float:
            x_interp = np.array([np.interp(t, self.times, self.x_traj[i, :])
                                 for i in range(self.n)])
            return self.a @ x_interp

        # Второй интеграл: b^T @ u(t)
        def integrand2(t: float) -> float:
            u = np.array([self.get_control(t, i) for i in range(self.m)])
            return self.b @ u

        Ob1 = quad(integrand1, 0, self.T)[0]
        Ob2 = quad(integrand2, 0, self.T)[0]

        self.ob_value = Ob1 + Ob2
        return self.ob_value

    def plot_controls(self):
        """Визуализация оптимальных управлений"""
        if self.U is None:
            self.solve_optimal_control()

        plt.figure(figsize=(12, 6))
        for i in range(self.m):
            plt.plot(self.times, self.U[i, :], label=f'u{i + 1}(t)')
        plt.xlabel('Time')
        plt.ylabel('Control')
        plt.title('Optimal Controls')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_controls_to_bytes(self):
        """Возвращает график как байты (PNG)"""
        if self.U is None:
            self.solve_optimal_control()

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
        """Визуализация траекторий системы"""
        if self.x_traj is None:
            self.compute_trajectory()

        plt.figure(figsize=(12, 6))
        for i in range(self.n):
            plt.plot(self.times, self.x_traj[i, :], label=f'x{i + 1}(t)')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('System Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_trajectories_to_bytes(self):
        """Визуализация траекторий системы"""
        if self.x_traj is None:
            self.compute_trajectory()

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

    def solve(self, K: int = 50) -> Dict[str, Union[np.ndarray, float]]:
        """Полное решение задачи"""
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
    # Пример для 3-мерного состояния и 2 управлений
    solver = OptimalControlSolverNd(
        T=6,
        M=1,
        N=5,
        n=2,
        m=2,
        x0=[5, 12],
        F_func=[["0.5", "-0.02"],["-0.02", "0.4"]],
        G_func=[["0.3", "0.3"],["0.2", "0.2"]],
        a=np.array([-1, 0]),
        b=np.array([0, 5]),
        B=np.array([[-1, 0], [0, -1], [2, 0], [0, 8], [2, -7]]),
        q=np.array([0, 0, 5, 20, 0]),
        ft_func = ["t", "1"]
    )

    results = solver.solve(K=50)
    print(f"Objective value: {results['objective']}")
    solver.plot_controls()
    solver.plot_trajectories()