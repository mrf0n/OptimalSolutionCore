import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad, solve_ivp
from scipy.optimize import linprog
import sympy as sp

# Параметры задачи
T = 6
M = 1
N = 5

# Векторы и матрицы
x0 = np.array([5, 12])
F = np.array([[0.5, -0.02], [-0.02, 0.4]])
G = np.array([[0.3, 0.3], [0.2, 0.2]])
a = np.array([-M * 1, 0])
b = np.array([0, N * 1])
t_mas = sp.symbols('t')
ft = sp.Matrix([t_mas, 1])
B = np.array([[-1, 0],
              [0, -1],
              [2, 0],
              [0, 8],
              [2, -7]])
q = np.array([0, 0, 5, 20, 0])


# Предварительное вычисление матричных экспонент
def compute_Yt(t):
    return expm(-F.T * t)


def compute_Xt(t):
    return expm(F * t)


# Вычисление констант c1, c2
YT = compute_Yt(T)
A1 = np.linalg.inv(YT)


# Оптимизированное вычисление A2 с использованием solve_ivp
def compute_A2():
    def ode_func(t, y):
        Cpts = compute_Yt(T) @ compute_Yt(-t)
        return Cpts @ a

    sol = solve_ivp(ode_func, [0, T], np.zeros(2), rtol=1e-6, atol=1e-8)
    return sol.y[:, -1].reshape(-1, 1)


A2 = compute_A2()
c = -A1 @ A2

# Кэширование для pt(t)
pt_cache = {}
Yt_cache = {}


def get_Yt(t):
    if t not in Yt_cache:
        Yt_cache[t] = compute_Yt(t)
    return Yt_cache[t]


# Оптимизированное вычисление pt(t)
def compute_pt(t):
    if t in pt_cache:
        return pt_cache[t]

    Yt_val = get_Yt(t)
    term1 = Yt_val @ c

    # Используем solve_ivp для вычисления интеграла
    def integrand(s, y):
        Cpts = get_Yt(t) @ get_Yt(-s)
        return (Cpts @ a).flatten()

    sol = solve_ivp(integrand, [0, t], np.zeros(2), rtol=1e-6, atol=1e-8)
    integral = sol.y[:, -1].reshape(-1, 1)

    pt_val = term1 + integral
    pt_cache[t] = pt_val
    return pt_val


# Оптимизированные вычисления для управления
def compute_Gp1(t):
    pt_val = compute_pt(t)
    return G.T @ pt_val - b.reshape(-1, 1)


# Оптимизация для нахождения управления us
K = 50  # Количество точек дискретизации
times = np.linspace(0, T, K + 1)

# Предварительно вычисляем все управления
R1 = np.zeros(K + 1)
R2 = np.zeros(K + 1)

for i, t in enumerate(times):
    c_obj = compute_Gp1(t).flatten()

    # Решение задачи линейного программирования
    res = linprog(-c_obj, A_ub=B, b_ub=q, bounds=(None, None))

    if res.success:
        R1[i], R2[i] = res.x
    else:
        R1[i], R2[i] = 0, 0


# Оптимизированные функции управления (линейная интерполяция)
def us1(t):
    return np.interp(t, times, R1)


def us2(t):
    return np.interp(t, times, R2)


# Решение системы для x(t) с использованием solve_ivp
def compute_trajectory():
    def ode_func(t, x):
        u = np.array([us1(t), us2(t)])
        ft_array = np.array(ft.subs(t_mas, t), dtype=float).flatten()
        dx = F @ x + G @ u + ft_array
        return dx

    sol = solve_ivp(ode_func, [0, T], x0, t_eval=times, rtol=1e-6, atol=1e-8)
    return sol.y


x_traj = compute_trajectory()
x1_plot = x_traj[0, :]
x2_plot = x_traj[1, :]

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(times, R1, label='u1(t)')
plt.plot(times, R2, label='u2(t)')
plt.xlabel('Time')
plt.ylabel('Control')
plt.title('Optimal Controls')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(times, x1_plot, label='x1(t)')
plt.plot(times, x2_plot, label='x2(t)')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('System Trajectories')
plt.legend()
plt.grid(True)
plt.show()


# Вычисление целевого функционала (численное интегрирование)
def objective():
    # Разделяем траектории на компоненты
    x1_traj = x_traj[0, :]
    x2_traj = x_traj[1, :]

    # Первый интеграл: a^T @ x(t) = a1*x1(t) + a2*x2(t)
    def integrand1(t):
        x1 = np.interp(t, times, x1_traj)
        x2 = np.interp(t, times, x2_traj)
        return a[0] * x1 + a[1] * x2

    Ob1 = quad(integrand1, 0, T)[0]

    # Второй интеграл: b^T @ u(t) = b1*u1(t) + b2*u2(t)
    def integrand2(t):
        u1_val = us1(t)
        u2_val = us2(t)
        return b[0] * u1_val + b[1] * u2_val

    Ob2 = quad(integrand2, 0, T)[0]

    return Ob1 + Ob2


Ob = objective()
print(f"Objective value: {Ob}")