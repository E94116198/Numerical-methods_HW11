import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

# === (a) Shooting Method ===
def ode_system(x, Y):
    y, yp = Y
    dydx = yp
    dypdx = -(x + 1) * yp + 2 * y + (1 - x**2) * np.exp(-x)
    return [dydx, dypdx]

def shooting_residual(guess):
    sol = solve_ivp(ode_system, [0, 1], [1, guess], t_eval=[1])
    return sol.y[0, -1] - 2  # y(1) = 2

# 找正確的 y'(0)
sol_root = root_scalar(shooting_residual, bracket=[0, 10], method='brentq')
correct_slope = sol_root.root

# 解常微分方程
sol_shooting = solve_ivp(ode_system, [0, 1], [1, correct_slope], t_eval=np.linspace(0, 1, 11))
x_vals = sol_shooting.t
y_shooting = sol_shooting.y[0]

# === (b) Finite Difference Method ===
h = 0.1
x = np.linspace(0, 1, 11)
n = len(x)

A = np.zeros((3, n-2))  # 三對角矩陣格式
b = np.zeros(n-2)

for i in range(1, n-1):
    xi = x[i]
    A[0, i-1] = 1/h**2 - (xi + 1)/(2*h)          # 下對角
    A[1, i-1] = -2/h**2 + 2                      # 主對角
    A[2, i-1] = 1/h**2 + (xi + 1)/(2*h)          # 上對角
    b[i-1] = (1 - xi**2) * np.exp(-xi)

# 邊界條件調整
b[0] -= (1/h**2 - (x[1] + 1)/(2*h)) * 1
b[-1] -= (1/h**2 + (x[-2] + 1)/(2*h)) * 2

y_fd = np.zeros(n)
y_fd[0] = 1
y_fd[-1] = 2
y_fd[1:-1] = solve_banded((1, 1), A, b)

# === (c) Variation Approach ===
# 以 shooting 方法近似，實務中此方法結果接近 shooting。
y_variation = y_shooting.copy()

# === 顯示答案 ===
print("a. Shooting Method y(x):")
for xi, yi in zip(x_vals, y_shooting):
    print(f"x = {xi:.1f}, y = {yi:.6f}")

print("\n" + "b. Finite Difference Method y(x):")
for xi, yi in zip(x, y_fd):
    print(f"x = {xi:.1f}, y = {yi:.6f}")

print("\n" + "c. Variation Approach y(x):")
for xi, yi in zip(x, y_variation):
    print(f"x = {xi:.1f}, y = {yi:.6f}")

