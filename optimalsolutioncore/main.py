import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import linprog
from scipy.integrate import quad_vec

# Параметры задачи
T = 6
M = 1
N = 5

# Векторы и матрицы
ft = np.array([[1], [0]])  # [t, 1], но t будет подставляться позже
x0 = np.array([[5], [12]])
F = np.array([[0.5, -0.02], [-0.02, 0.4]])
G = np.array([[0.3, 0.3], [0.2, 0.2]])
a = np.array([[-M * 1], [0]])
b = np.array([[0], [N * 1]])

B = np.array([[-1, 0],
              [0, -1],
              [2, 0],
              [0, 8],
              [2, -7]])
q = np.array([[0], [0], [5], [20], [0]])


# Функции матричных экспонент
def Yt(t):
    return expm(-F.T * t)


def Y1s(s):
    return expm(-F.T * (-s))

def Xt(t):
    return expm(F * t)

def X1s(s):
    return expm(-F * s)

# Вычисление констант c1, c2
YT = Yt(T)
A1 = np.linalg.inv(YT)

# Вычисление A2 (интегральная часть)
def Cptsa(t, s):
    Cpts = Yt(t) @ Y1s(s)
    return Cpts @ a

# Приближенное вычисление интегралов для A2
def compute_A2():
    h = 0.01
    integral1 = 0
    integral2 = 0
    for s in np.arange(0, T, h):
        val = Cptsa(T, s)[0][0]
        integral1 += val * h
        val = Cptsa(T, s)[1][0]
        integral2 += val * h
    return np.array([[integral1], [integral2]])

A2 = compute_A2()
c = -A1 @ A2

# Вычисление pt(t)
def pt(t):
    term1 = Yt(t) @ c
    integral1 = quad(lambda s: Cptsa(t, s)[0][0], 0, t)[0]
    integral2 = quad(lambda s: Cptsa(t, s)[1][0], 0, t)[0]
    return term1 + np.array([[integral1], [integral2]])

# Вычисление Gp(t) = G^T @ pt(t)
def Gp(t):
    pt_val = pt(t)
    return G.T @ pt_val

# Вычисление Gp1(t) = Gp(t) - b
def Gp1(t):
    return Gp(t) - b

# Оптимизация для нахождения управления us
K = 50
h = T / K
times = np.linspace(0, T, K + 1)

# Результаты оптимизации
R1 = []
R2 = []

for t in times:
    # Целевая функция (коэффициенты из Gp1(t))
    gp1_val = Gp1(t)
    c_obj = [gp1_val[0][0], gp1_val[1][0]]

    # Ограничения Bx <= q
    A_ub = B
    b_ub = q.flatten()

    # Решение задачи линейного программирования
    res = linprog(-np.array(c_obj), A_ub=A_ub, b_ub=b_ub, bounds=(None, None))

    if res.success:
        R1.append(res.x[0])
        R2.append(res.x[1])
    else:
        R1.append(0)
        R2.append(0)

# Построение кусочно-постоянного управления
def piecewise_control(t, moments, values):
    # Если t меньше первого момента, возвращаем первое значение
    if t < moments[0]:
        return values[0]

    # Ищем интервал, в который попадает t
    for i in range(len(moments) - 1):
        if moments[i] <= t < moments[i + 1]:
            return values[i]

    # Если t больше или равен последнему моменту, возвращаем последнее значение
    return values[-1]

# Находим моменты изменения управления
def find_switches(values, times):
    if not values:
        return [], []

    switches = [times[i] for i in range(len(values) - 1) if values[i] != values[i + 1]]
    switch_values = [values[i] for i in range(len(values) - 1) if values[i] != values[i + 1]] + [values[-1]]

    return switches, switch_values

switches1, switch_values1 = find_switches(R1, times)
switches2, switch_values2 = find_switches(R2, times)

# Функции управления
def us1(t):
    return piecewise_control(t, [0] + switches1 + [T], switch_values1)

def us2(t):
    return piecewise_control(t, [0] + switches2 + [T], switch_values2)

# Визуализация управлений
t_plot = np.linspace(0, T, K+1)
u1_plot = [us1(t) for t in t_plot]
u2_plot = [us2(t) for t in t_plot]
u1_test = R1
u2_test = R2

plt.figure(figsize=(12, 6))
plt.plot(t_plot, u1_plot, label='u1(t)', linewidth=2)
plt.plot(t_plot, u2_plot, label='u2(t)', linewidth=2)
plt.plot(t_plot, u1_test, label='u1_test(t)', linewidth=2)
plt.plot(t_plot, u2_test, label='u2_test(t)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Control')
plt.title('Optimal Controls')
plt.legend()
plt.grid(True)
plt.show()

# Решение системы для x(t)
def x_solution(t):
    # Первый член: Xt(t) @ x0
    term1 = Xt(t) @ x0

    # Второй член: интеграл от Cts(t,s) @ G @ us(s) ds
    def integrand1(s):
        Cts = Xt(t) @ X1s(s)
        us = np.array([[us1(s)], [us2(s)]])
        return (Cts @ G @ us).flatten()

    integral1 = np.zeros((2, 1))
    for i in range(2):
        integral1[i] = quad(lambda s: integrand1(s)[i], 0, t)[0]

    # Третий член: интеграл от Cts(t,s) @ fs(s) ds
    def integrand2(s):
        Cts = Xt(t) @ X1s(s)
        fs = np.array([[s], [1]])
        return (Cts @ fs).flatten()

    integral2 = np.zeros((2, 1))
    for i in range(2):
        integral2[i] = quad(lambda s: integrand2(s)[i], 0, t)[0]

    return term1 + integral1 + integral2

# Вычисление траекторий
x1_plot = []
x2_plot = []
for t in t_plot:
    x = x_solution(t)
    x1_plot.append(x[0][0])
    x2_plot.append(x[1][0])

# Визуализация траекторий
plt.figure(figsize=(12, 6))
plt.plot(t_plot, x1_plot, label='x1(t)', linewidth=2)
plt.plot(t_plot, x2_plot, label='x2(t)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('System Trajectories')
plt.legend()
plt.grid(True)
plt.show()

# Вычисление целевого функционала
def objective():
    # Первый интеграл: a^T @ x(t)
    def integrand1(t):
        x = x_solution(t)
        return (a.T @ x)[0][0]

    Ob1 = quad(integrand1, 0, T)[0]

    # Второй интеграл: b^T @ u(t)
    def integrand2(t):
        u = np.array([[us1(t)], [us2(t)]])
        return (b.T @ u)[0][0]

    Ob2 = quad(integrand2, 0, T)[0]

    return Ob1 + Ob2

Ob = objective()
print(f"Objective value: {Ob}")