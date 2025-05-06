import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad, solve_ivp
from scipy.optimize import linprog
import sympy as sp


class OptimalControlSolver:
    """
    Класс для решения задачи оптимального управления

    Параметры:
    T: float - конечное время
    M: float - параметр M
    N: float - параметр N
    x0: np.array - начальное состояние [x1, x2]
    F: np.array - матрица F 2x2
    G: np.array - матрица G 2x2
    B: np.array - матрица ограничений управления
    q: np.array - вектор ограничений управления
    """

    def __init__(self, T=6, M=1, N=5, x0=None, F=None, G=None, B=None, q=None):
        # Параметры по умолчанию
        self.T = T
        self.M = M
        self.N = N

        # Инициализация параметров
        self.x0 = np.array([5.0, 12.0]) if x0 is None else np.array(x0, dtype=float)
        self.F = np.array([[0.5, -0.02], [-0.02, 0.4]], dtype=float) if F is None else np.array(F, dtype=float)
        self.G = np.array([[0.3, 0.3], [0.2, 0.2]], dtype=float) if G is None else np.array(G, dtype=float)
        self.B = np.array([[-1, 0], [0, -1], [2, 0], [0, 8], [2, -7]], dtype=float) if B is None else np.array(B,
                                                                                                               dtype=float)
        self.q = np.array([0.0, 0.0, 5.0, 20.0, 0.0], dtype=float) if q is None else np.array(q, dtype=float)

        # Дополнительные параметры
        self.a = np.array([-self.M * 1.0, 0.0])
        self.b = np.array([0.0, self.N * 1.0])

        # Символьная переменная для ft
        self.t_sym = sp.symbols('t')
        self.ft_sym = sp.Matrix([self.t_sym, 1])

        # Кэши
        self.pt_cache = {}
        self.Yt_cache = {}

        # Результаты
        self.times = None
        self.R1 = None
        self.R2 = None
        self.x_traj = None
        self.ob_value = None

    def compute_Yt(self, t):
        """Вычисление матричной экспоненты Y(t)"""
        return expm(-self.F.T * t)

    def compute_Xt(self, t):
        """Вычисление матричной экспоненты X(t)"""
        return expm(self.F * t)

    def compute_A2(self):
        """Вычисление интегральной части A2"""

        def ode_func(t, y):
            Cpts = self.compute_Yt(self.T) @ self.compute_Yt(-t)
            return Cpts @ self.a

        sol = solve_ivp(ode_func, [0, self.T], np.zeros(2), rtol=1e-6, atol=1e-8)
        return sol.y[:, -1].reshape(-1, 1)

    def get_Yt(self, t):
        """Получение Y(t) с кэшированием"""
        if t not in self.Yt_cache:
            self.Yt_cache[t] = self.compute_Yt(t)
        return self.Yt_cache[t]

    def compute_pt(self, t):
        """Вычисление pt(t) с кэшированием"""
        if t in self.pt_cache:
            return self.pt_cache[t]

        Yt_val = self.get_Yt(t)
        term1 = Yt_val @ self.c

        def integrand(s, y):
            Cpts = self.get_Yt(t) @ self.get_Yt(-s)
            return (Cpts @ self.a).flatten()

        sol = solve_ivp(integrand, [0, t], np.zeros(2), rtol=1e-6, atol=1e-8)
        integral = sol.y[:, -1].reshape(-1, 1)

        pt_val = term1 + integral
        self.pt_cache[t] = pt_val
        return pt_val

    def compute_Gp1(self, t):
        """Вычисление Gp1(t) = G^T @ pt(t) - b"""
        pt_val = self.compute_pt(t)
        return self.G.T @ pt_val - self.b.reshape(-1, 1)

    def solve_optimal_control(self, K=50):
        """Решение задачи оптимального управления"""
        # Вычисление констант
        YT = self.compute_Yt(self.T)
        A1 = np.linalg.inv(YT)
        A2 = self.compute_A2()
        self.c = -A1 @ A2

        # Дискретизация времени
        self.times = np.linspace(0, self.T, K + 1)

        # Оптимизация управления
        self.R1 = np.zeros(K + 1)
        self.R2 = np.zeros(K + 1)

        for i, t in enumerate(self.times):
            c_obj = self.compute_Gp1(t).flatten()
            res = linprog(-c_obj, A_ub=self.B, b_ub=self.q, bounds=(None, None))

            if res.success:
                self.R1[i], self.R2[i] = res.x
            else:
                self.R1[i], self.R2[i] = 0.0, 0.0

    def us1(self, t):
        """Функция управления u1(t)"""
        return np.interp(t, self.times, self.R1)

    def us2(self, t):
        """Функция управления u2(t)"""
        return np.interp(t, self.times, self.R2)

    def compute_trajectory(self):
        """Вычисление траектории системы"""

        def ode_func(t, x):
            u = np.array([self.us1(t), self.us2(t)])
            ft_array = np.array(self.ft_sym.subs(self.t_sym, t), dtype=float).flatten()
            dx = self.F @ x + self.G @ u + ft_array
            return dx

        sol = solve_ivp(ode_func, [0, self.T], self.x0, t_eval=self.times, rtol=1e-6, atol=1e-8)
        self.x_traj = sol.y

    def compute_objective(self):
        """Вычисление целевого функционала"""
        if self.x_traj is None:
            self.compute_trajectory()

        # Первый интеграл: a^T @ x(t)
        def integrand1(t):
            x1 = np.interp(t, self.times, self.x_traj[0, :])
            x2 = np.interp(t, self.times, self.x_traj[1, :])
            return self.a[0] * x1 + self.a[1] * x2

        # Второй интеграл: b^T @ u(t)
        def integrand2(t):
            return self.b[0] * self.us1(t) + self.b[1] * self.us2(t)

        Ob1 = quad(integrand1, 0, self.T)[0]
        Ob2 = quad(integrand2, 0, self.T)[0]

        self.ob_value = Ob1 + Ob2
        return self.ob_value

    def plot_controls(self):
        """Визуализация оптимальных управлений"""
        if self.R1 is None:
            self.solve_optimal_control()

        plt.figure(figsize=(12, 6))
        plt.plot(self.times, self.R1, label='u1(t)')
        plt.plot(self.times, self.R2, label='u2(t)')
        plt.xlabel('Time')
        plt.ylabel('Control')
        plt.title('Optimal Controls')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_trajectories(self):
        """Визуализация траекторий системы"""
        if self.x_traj is None:
            self.compute_trajectory()

        plt.figure(figsize=(12, 6))
        plt.plot(self.times, self.x_traj[0, :], label='x1(t)')
        plt.plot(self.times, self.x_traj[1, :], label='x2(t)')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('System Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    def solve(self, K=50):
        """Полное решение задачи"""
        self.solve_optimal_control(K)
        self.compute_trajectory()
        self.compute_objective()
        return {
            'controls': (self.R1, self.R2),
            'trajectory': self.x_traj,
            'objective': self.ob_value
        }


# # Пример использования
# if __name__ == "__main__":
#     solver = OptimalControlSolver()
#     results = solver.solve()
#
#     print(f"Objective value: {results['objective']}")
#     solver.plot_controls()
#     solver.plot_trajectories()