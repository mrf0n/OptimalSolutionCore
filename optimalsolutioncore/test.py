import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import linprog
import sympy as sp
from typing import List, Optional, Callable
from scipy.interpolate import InterpolatedUnivariateSpline

class NonStationaryOptimalControl:
    """
    Solves an optimal control problem with non-stationary matrices of arbitrary size.
    Implements a similar approach to the Maple code but with more general handling.
    """

    def __init__(self,
                 T: float = None,
                 M: float = None,
                 N: float = None,
                 n: int = None,
                 m: int = None,
                 x0: Optional[np.ndarray] = None,
                 F_func: List[List[str]] = None,
                 G_func: List[List[str]] = None,
                 a: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None,
                 B: Optional[np.ndarray] = None,
                 q: Optional[np.ndarray] = None,
                 ft_func: List[List[str]] = None):
        """
        Initialize the optimal control solver with parameters.

        Args:
            T: Final time
            M: Parameter M from Maple
            N: Parameter N from Maple
            n: State dimension
            m: Control dimension
            F: Function F(t) returning n x n matrix
            G: Function G(t) returning n x m matrix
            ft: Function ft(t) returning n x 1 vector
            x0: Initial state (n x 1)
            a: Vector a (n x 1) for objective
            b: Vector b (m x 1) for objective
            B: Constraint matrix (k x m)
            q: Constraint vector (k x 1)
        """
        self.T = T
        self.M = M
        self.N = N
        self.n = n
        self.m = m
        self.x0 = x0

        # Initialize variables with defaults similar to Maple code
        if ft_func is None:
            def default_ft(t):
                ft = np.zeros(self.n)
                ft[0] = t
                if self.n > 1:
                    ft[1] = 1.0
                return ft

            self.ft = default_ft
        else:
            self.ft = self.create_matrix_function(ft_func)

        # System matrices (default to Maple example)
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
            self.G = self.create_matrix_function(G_func)

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
            self.F = self.create_matrix_function(F_func)

        # Objective parameters
        self.a = a
        self.b = b

        # Constraints B*u <= q
        self.B = B
        self.q = q

        # Caches for intermediate results
        self.Yt_cache = {}
        self.Xt_cache = {}

        # Results storage
        self.times = None
        self.optimal_control = None
        self.state_trajectory = None
        self.objective_value = None

    def compute_Yt(self, t: float) -> np.ndarray:
        """Compute Y(t) = exp(-F^T*t) by solving the matrix ODE dY/dt = -F(t)^T Y, Y(0)=I"""
        if t in self.Yt_cache:
            return self.Yt_cache[t]

        # Flatten the matrix for ODE solver
        y0 = np.eye(self.n).flatten()

        def y_ode(t: float, y_flat: np.ndarray) -> np.ndarray:
            Y = y_flat.reshape((self.n, self.n))
            F_val = self.F(t)
            dYdt = -F_val.T @ Y
            return dYdt.flatten()

        # Solve the ODE
        sol = solve_ivp(y_ode, [0, t], y0, method='LSODA', rtol=1e-6, atol=1e-8)
        Yt = sol.y[:, -1].reshape((self.n, self.n))
        self.Yt_cache[t] = Yt
        return Yt

    def compute_Xt(self, t: float) -> np.ndarray:
        """Compute X(t) = exp(F*t) by solving the matrix ODE dX/dt = F(t) X, X(0)=I"""
        if t in self.Xt_cache:
            return self.Xt_cache[t]

        # Flatten the matrix for ODE solver
        x0 = np.eye(self.n).flatten()

        def x_ode(t: float, x_flat: np.ndarray) -> np.ndarray:
            X = x_flat.reshape((self.n, self.n))
            F_val = self.F(t)
            dXdt = F_val @ X
            return dXdt.flatten()

        # Solve the ODE
        sol = solve_ivp(x_ode, [0, t], x0, method='LSODA', rtol=1e-6, atol=1e-8)
        Xt = sol.y[:, -1].reshape((self.n, self.n))
        self.Xt_cache[t] = Xt
        return Xt

    def compute_state_trajectory(self, times: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute the state trajectory using exact integration with control and forcing term"""
        state = np.zeros((self.n, len(times)))
        state[:, 0] = self.x0.flatten()

        # Create interpolation function for control
        from scipy.interpolate import interp1d
        u_interp = [interp1d(times, control[i, :], kind='linear',
                             fill_value="extrapolate") for i in range(self.m)]

        def integrand(t, x, tau):
            """Integrand for the state equation"""
            # Current control value
            u = np.array([u_interp[i](tau) for i in range(self.m)]).reshape(-1, 1)
            # System matrices at time tau
            F_val = self.F(tau)
            G_val = self.G(tau)
            ft_val = self.ft(tau)
            # The integrand: X(t)X⁻¹(τ)[G(τ)u(τ) + f(τ)]
            Xt = self.compute_Xt(t)
            Xtau_inv = np.linalg.inv(self.compute_Xt(tau))
            return Xt @ Xtau_inv @ (G_val @ u + ft_val)

        for i in range(1, len(times)):
            t_prev = times[i - 1]
            t_curr = times[i]

            # Homogeneous solution (state transition)
            X_curr = self.compute_Xt(t_curr)
            X_prev_inv = np.linalg.inv(self.compute_Xt(t_prev))
            homogeneous_part = X_curr @ X_prev_inv @ state[:, i - 1].reshape(-1, 1)

            # Particular solution (integral of control and forcing term)
            # We'll use numerical integration for this part
            def particular_integrand(tau):
                return integrand(t_curr, state[:, i - 1], tau).flatten()

            # Use quad for each component separately
            particular = np.zeros((self.n, 1))
            for j in range(self.n):
                res, _ = quad(lambda tau: particular_integrand(tau)[j],
                              t_prev, t_curr,
                              limit=100, epsabs=1e-6, epsrel=1e-6)
                particular[j] = res

            # Combine homogeneous and particular solutions
            state[:, i] = (homogeneous_part + particular).flatten()

        return state

    def compute_pt(self, t: float) -> np.ndarray:
        """Compute p(t) = Y(t)c + integral_0^t Y(t)Y(-s)a ds"""
        Yt = self.compute_Yt(t)

        # Compute the integral term
        def integrand(s: float) -> np.ndarray:
            Y1s = self.compute_Yt(-s)
            return (Yt @ Y1s @ self.a).flatten()

        integral = np.zeros((self.n, 1))
        for i in range(self.n):
            integral[i], _ = quad(lambda s: integrand(s)[i], 0, t,
                                  limit=100, epsabs=1e-6, epsrel=1e-6)

        return Yt @ self.c + integral

    def solve(self, K: int = 50) -> dict:
        """Solve the optimal control problem with K time steps"""
        # Step 1: Compute Y(T) and A1, A2 to find c
        YT = self.compute_Yt(self.T)
        A1 = np.linalg.inv(YT)

        # Compute A2 = integral_0^T Y(T)Y(-s)a ds
        def integrand_A2(s: float) -> np.ndarray:
            Y1s = self.compute_Yt(-s)
            return (YT @ Y1s @ self.a).flatten()

        A2 = np.zeros((self.n, 1))
        for i in range(self.n):
            A2[i], _ = quad(lambda s: integrand_A2(s)[i], 0, self.T,
                            limit=100, epsabs=1e-6, epsrel=1e-6)

        self.c = -A1 @ A2

        # Step 2: Discretize time and find optimal control at each point
        self.times = np.linspace(0, self.T, K + 1)
        self.optimal_control = np.zeros((self.m, K + 1))

        for i, t in enumerate(self.times):
            # Compute p(t)
            pt = self.compute_pt(t)

            # Compute G(t)^T p(t) - b
            G_val = self.G(t)
            Gp = G_val.T @ pt - self.b.reshape(-1, 1)

            # Solve linear programming problem
            res = linprog(-Gp.flatten(), A_ub=self.B, b_ub=self.q,
                          bounds=(None, None), method='highs')

            if res.success:
                self.optimal_control[:, i] = res.x
            else:
                self.optimal_control[:, i] = 0.0

        # Step 3: Compute the state trajectory using X(t)
        self.state_trajectory = self.compute_state_trajectory(self.times, self.optimal_control)

        # Step 4: Compute objective value
        self.objective_value = self.compute_objective()

        return {
            'times': self.times,
            'control': self.optimal_control,
            'state': self.state_trajectory,
            'objective': self.objective_value
        }

    def compute_objective(self) -> float:
        """Compute the objective function value"""
        if self.state_trajectory is None or self.optimal_control is None:
            raise ValueError("Run solve() first")

        # First term: integral of a^T x(t)
        def integrand1(t: float) -> float:
            x = np.array([np.interp(t, self.times, self.state_trajectory[i, :])
                          for i in range(self.n)]).reshape(-1, 1)
            return (self.a.T @ x).item()

        Ob1, _ = quad(integrand1, 0, self.T, limit=100, epsabs=1e-6, epsrel=1e-6)

        # Second term: integral of b^T u(t)
        def integrand2(t: float) -> float:
            u = np.array([np.interp(t, self.times, self.optimal_control[i, :])
                          for i in range(self.m)]).reshape(-1, 1)
            return (self.b.T @ u).item()

        Ob2, _ = quad(integrand2, 0, self.T, limit=100, epsabs=1e-6, epsrel=1e-6)

        return Ob1 + Ob2

    def plot_results(self) -> None:
        """Plot the results similar to Maple output"""
        if self.optimal_control is None or self.state_trajectory is None:
            raise ValueError("Run solve() first")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # # Plot control
        # ax1.plot(self.times, self.optimal_control[0, :], 'b-', linewidth=2, label='u1(t)')
        # ax1.plot(self.times, self.optimal_control[1, :], 'g-', linewidth=3, label='u2(t)')
        # ax1.set_xlabel('Time')
        # ax1.set_ylabel('Control')
        # ax1.legend()
        # ax1.grid(True)
        # ax1.set_title('Optimal Control')
        #
        # # Plot state
        # ax2.plot(self.times, self.state_trajectory[0, :], 'r-', linewidth=2, label='x1(t)')
        # ax2.plot(self.times, self.state_trajectory[1, :], 'g-', linewidth=2, label='x2(t)')
        # ax2.set_xlabel('Time')
        # ax2.set_ylabel('State')
        # ax2.legend()
        # ax2.grid(True)
        # ax2.set_title('State Trajectory')
        # Plot control
        ax1.plot(list(self.Xt_cache.keys()), [x[0] for x in self.Xt_cache.values()], 'b-', label='test1(t)')
        ax1.plot(list(self.Xt_cache.keys()), [x[1] for x in self.Xt_cache.values()], 'g-', label='test2(t)')
        items = sorted(self.Xt_cache.items(), key=lambda kv: kv[0])
        t_vals = np.array([t for t, X in items])
        test1_vals = np.array([X[0, 1] for t, X in items])
        spl1 = InterpolatedUnivariateSpline(t_vals, test1_vals, k=3)
        # Задаём «тонкую» сетку для отображения
        t_fine = np.linspace(t_vals.min(), t_vals.max(), 200)
        # Вычисляем значения сплайна на этой сетке
        y_spline = spl1(t_fine)
        # Рисуем
        plt.figure(figsize=(8, 4))
        # Отрисуем исходные точки
        plt.scatter(t_vals, test1_vals, color='red', label='исходные точки')
        # Отрисуем сплайн
        plt.plot(t_fine, y_spline, '-', label='сплайн (k=3)')
        plt.xlabel('t')
        plt.ylabel('test1')
        plt.title('Интерполяция сплайном')
        plt.legend()
        plt.grid(True)
        plt.show()
        # СПЛАЙН
        # ax1.set_xlabel('Time')
        # ax1.set_ylabel('Control')
        # ax1.legend()
        # ax1.grid(True)
        # ax1.set_title('XT')
        #
        # plt.tight_layout()
        # plt.show()

    def create_matrix_function(self, matrix_str):
        """
        Создает лямбда-функцию для матрицы из массива строк
        matrix_str: List[List[str]] - массив строк с выражениями элементов матрицы
        Возвращает: Callable[[float], np.ndarray] - функция, принимающая t и возвращающая матрицу
        """
        # Создаем шаблонную лямбда-функцию, если матрица постоянная
        try:
            # Пробуем преобразовать все элементы в числа
            const_matrix = np.array([[float(elem) for elem in row] for row in matrix_str])
            return lambda t: const_matrix
        except ValueError:
            # Если есть элементы, которые нельзя преобразовать в float (содержат 't' или другие переменные)
            # В этом случае нужно использовать eval (осторожно с безопасностью!)
            return lambda t: np.array([[eval(elem, {'t': t}) for elem in row] for row in matrix_str])


# Example usage
if __name__ == "__main__":
    # Create and solve the problem with default parameters (Maple example)
    solver = NonStationaryOptimalControl(
        T=2,
        M=1,
        N=5,
        n=2,
        m=2,
        x0=np.array([5, 12]),
        F_func=[["t", "-0.1"],["-0.2", "0.25*t*t"]],
        G_func=[["0.3", "0.3"],["0.2", "0.2"]],
        a=np.array([-1, 0]),
        b=np.array([0, 5]),
        B=np.array([[-1, 0], [0, -1], [2, 0], [0, 8], [2, -7]]),
        q=np.array([0, 0, 5, 20, 0]),
        ft_func = [["t"], ["1"]]
    )

    results = solver.solve(K=50)

    # Print and plot results
    print(f"Objective value: {results['objective']}")
    solver.plot_results()