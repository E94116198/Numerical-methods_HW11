import numpy as np 
from scipy.integrate import solve_ivp, quad
from numpy.linalg import solve

# ===================
# 共用設定
# ===================
h = 0.1
x_vals = np.linspace(0, 1, 11)

# ===================
# (a) Shooting Method
# ===================
def ode_sys(x, Y):
    y, dy = Y
    ddy = -(1 + x)*dy + 2*y + (1 - x**2)*np.exp(-x)
    return [dy, ddy]

def shooting_method(y0, y1_target, tol=1e-8):
    def integrate(s):
        sol = solve_ivp(ode_sys, (0, 1), [y0, s], t_eval=x_vals)
        return sol.y[0][-1], sol.y[0]

    s0, s1 = 0.0, 2.0
    yb0, _ = integrate(s0)
    yb1, y_values = integrate(s1)

    for _ in range(50):
        s2 = s1 + (y1_target - yb1) * (s1 - s0) / (yb1 - yb0)
        yb2, y_values = integrate(s2)
        if abs(yb2 - y1_target) < tol:
            return y_values
        s0, yb0 = s1, yb1
        s1, yb1 = s2, yb2
    raise ValueError("Shooting method failed to converge.")

y_shooting = shooting_method(1, 2)

# ===================
# (b) Finite Difference Method (改為第2段程式碼方式)
# ===================
N = 9
A = np.zeros((N, N))
b = np.zeros(N)

for i in range(N):
    xi = x_vals[i + 1]
    pi = -(xi + 1)
    qi = 2
    ri = (1 - xi**2) * np.exp(-xi)

    A[i, i] = 2 + h**2 * qi

    if i > 0:
        A[i, i - 1] = -1 - 0.5 * h * pi
    else:
        b[i] += (1 + 0.5 * h * pi) * 1

    if i < N - 1:
        A[i, i + 1] = -1 + 0.5 * h * pi
    else:
        b[i] += (1 - 0.5 * h * pi) * 2

    b[i] += h**2 * ri

y_fd_internal = np.linalg.solve(A, b)
y_fd = np.zeros(11)
y_fd[0] = 1
y_fd[1:-1] = y_fd_internal
y_fd[-1] = 2

# ===================
# (c) Variation Approach (使用 sin 基底)
# ===================
def phi(i, x):
    return np.sin(i * np.pi * x)

def dphi(i, x):
    return i * np.pi * np.cos(i * np.pi * x)

N_var = 4
A_var = np.zeros((N_var, N_var))
b_var = np.zeros(N_var)

for i in range(N_var):
    for j in range(N_var):
        A_var[i, j] = quad(lambda x: dphi(i+1, x)*dphi(j+1, x) + 2 * phi(i+1, x) * phi(j+1, x), 0, 1)[0]
    b_var[i] = quad(lambda x: (1 - x**2)*np.exp(-x) * phi(i+1, x), 0, 1)[0]

c = solve(A_var, b_var)

def y_variational_fn(x):
    base = 1 + (2 - 1) * x  # y_base(x)
    correction = sum(c[i]*phi(i+1, x) for i in range(N_var))
    return base + correction

y_variational = np.array([y_variational_fn(xi) for xi in x_vals])

# ===================
# 表格輸出
# ===================
print(f"{'x':<4}|{'Shooting Method':^20}|{'Finite Difference Method':^26}|{'Variation Approach Method':^27}")
print("-"*50)
for i in range(len(x_vals)):
    print(f"{x_vals[i]:<4.1f}|{y_shooting[i]:^20.6f}|{y_fd[i]:^26.6f}|{y_variational[i]:^27.6f}")
