import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.linalg import inv
from scipy.optimize import linprog


def fundamental_matrix(F_func, T, n, transposed=False):
    """
    Compute fundamental matrix X(T)=exp(\int_0^T F(t) dt)
    or Y(T)=exp(-\int_0^T F(t).T dt) by solving ODE:
      if not transposed: dX/dt = F(t) X, X(0)=I
      else:               dY/dt = -F(t).T Y, Y(0)=I
    Returns (Phi_T, Phi_sol) where Phi_sol(t) gives matrix at t.
    """
    def ode(t, y):
        M = y.reshape(n, n)
        if not transposed:
            dM = F_func(t) @ M
        else:
            dM = -F_func(t).T @ M
        return dM.flatten()

    y0 = np.eye(n).flatten()
    sol = solve_ivp(ode, [0, T], y0, dense_output=True)
    Phi_T = sol.sol(T).reshape(n, n)
    return Phi_T, sol.sol


def compute_control(F_func, G_func, f_func, x0, a_vec, B, q, T, K):
    """
    Compute optimal piecewise-constant control u over [0,T] for system:
      dx/dt = F(t)x + f(t) + G(t)u(t)
    with linear cost via switching method and LP at K+1 nodes.
    F_func: t->(n,n), G_func: t->(n,m), f_func: t->(n,1)
    a_vec: (n,1) vector for adjoint integration
    B,q: constraints B u <= q
    Returns: u (m x K array), switching moments list
    """
    n = x0.shape[0]
    m = B.shape[1]

    # 1) Compute adjoint fundamental: Y(T) and solver
    YT, Y_sol = fundamental_matrix(F_func, T, n, transposed=True)
    A1 = inv(YT)

    # 2) Compute A2 = \int_0^T Y(T) Y(s)^{-1} a ds
    def integrand_A2(s, i):
        Ys = Y_sol(s).reshape(n, n)
        YTs = YT @ inv(Ys)
        return float((YTs @ a_vec)[i,0])

    A2 = np.array([quad(integrand_A2, 0, T, args=(i,))[0] for i in range(n)])[:,None]
    c_vec = -A1 @ A2

    # 3) Partition and solve LP at each node
    times = np.linspace(0, T, K+1)
    u = np.zeros((m, K))

    for k in range(K):
        t_k = times[k]
        # Y(t_k)
        Yt, _ = fundamental_matrix(F_func, t_k, n, transposed=True)
        # integral term \int_0^t_k Y(t_k)Y(s)^{-1} a ds
        def integrand_p(s, i):
            Ys = Y_sol(s).reshape(n, n)
            return float((Yt @ inv(Ys) @ a_vec)[i,0])
        integral = np.array([quad(integrand_p, 0, t_k, args=(i,))[0] for i in range(n)])[:,None]
        p_t = Yt @ c_vec + integral
        # objective coefficients: G.T p_t - b
        coef = (G_func(t_k).T @ p_t).flatten()
        # solve max coef^T u s.t. B u <= q, u>=0 -> min -coef^T u
        res = linprog(-coef, A_ub=B, b_ub=q, bounds=[(0,None)]*m)
        if not res.success:
            raise RuntimeError(f"LP failed at t={t_k}: {res.message}")
        u[:,k] = res.x

    # 4) Detect switching times
    switches = []  # list of (control_index, time, old, new)
    h = T/K
    for i in range(m):
        last = u[i,0]
        for k in range(1,K):
            val = u[i,k]
            if not np.isclose(val, last):
                switches.append((i, k*h, last, val))
                last = val
    return u, switches


def simulate(F_func, G_func, f_func, x0, u, T):
    """
    Simulate state trajectory under piecewise-constant u.
    Returns solve_ivp solution object.
    """
    n = x0.shape[0]
    m, K = u.shape
    h = T / K

    def u_func(t):
        idx = min(int(t // h), K-1)
        return u[:,idx:idx+1]

    def ode(t, x):
        x = x.reshape(n,1)
        return (F_func(t) @ x + f_func(t) + G_func(t) @ u_func(t)).flatten()

    sol = solve_ivp(ode, [0, T], x0.flatten(), dense_output=True)
    return sol


def compute_objective(aT, bT, sol, u, switches, T):
    """
    Compute cost: Ob = \int_0^T a^T x(t) dt + \int_0^T b^T u(t) dt
    aT: 1xn row vector, bT: 1xm row vector
    sol: state solution, u: control array
    """
    # integrate a^T x
    def integrand_state(t):
        x = sol.sol(t).reshape(-1,1)
        return float(aT @ x)
    Jx = quad(integrand_state, 0, T)[0]
    # integrate b^T u piecewise
    m,K = u.shape
    h = T/K
    J_u = 0.0
    for k in range(K):
        J_u += float(bT @ u[:,k:k+1]) * h
    return Jx + J_u


if __name__ == '__main__':
    # Numeric example from Maple code
    T, K = 6.0, 50
    F_func = lambda t: np.array([[0.5, -0.02],[-0.02,0.4]])
    G_func = lambda t: np.array([[0.3,0.3],[0.2,0.2]])
    f_func = lambda t: np.array([[t],[1.0]])
    x0 = np.array([[5.0],[12.0]])
    a_vec = np.array([[-1.0],[0.0]])
    B = np.array([[-1,0],[0,-1],[2,0],[0,8],[2,-7]])
    q = np.array([0.0,0.0,5.0,20.0,0.0])
    u, switches = compute_control(F_func, G_func, f_func, x0, a_vec, B, q, T, K)
    print("Switching moments:", switches)

    sol = simulate(F_func, G_func, f_func, x0, u, T)

    aT = np.array([[-1.0, 0.0]])
    bT = np.array([[0.0, 5.0]])
    Obj = compute_objective(aT, bT, sol, u, switches, T)
    print("Objective value:", Obj)
