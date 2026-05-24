import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp   # ← исправлено!
from scipy.linalg import eig

# ========== ВВЕДИТЕ ПАРАМЕТРЫ ЗДЕСЬ ==========
a = 1.0
b = -3.0
# ============================================

def system(X, Y):
    dx = (-X + 2*Y) * (b + X - Y)
    dy = -Y * (a + 2*X - 3*Y)
    return dx, dy

def f(t, z):
    return system(z[0], z[1])

# ---- 1. Находим все особые точки ----
E1 = np.array([0.0, 0.0])
E2 = np.array([-b, 0.0])
E3 = np.array([-2*a, -a])
E4 = np.array([a - 3*b, a - 2*b])

equilibria = [E1, E2, E3, E4]

# ---- 2. Функция для определения типа особой точки ----
def get_type_and_eigenvectors(x, y):
    """Возвращает тип точки и собственные векторы (если седло)"""
    J = np.array([
        [-b - 2*x + 3*y, 2*b + 3*x - 4*y],
        [-2*y, -a - 2*x + 6*y]
    ])
    
    trace = np.trace(J)
    det = np.linalg.det(J)
    disc = trace**2 - 4*det
    
    if det < 0:
        typ = "седло"
        vals, vecs = eig(J)
        v_stable = None
        v_unstable = None
        for i, val in enumerate(vals):
            if np.real(val) < 0:
                v_stable = np.real(vecs[:, i])
            else:
                v_unstable = np.real(vecs[:, i])
        if v_stable is not None:
            v_stable = v_stable / np.linalg.norm(v_stable)
        if v_unstable is not None:
            v_unstable = v_unstable / np.linalg.norm(v_unstable)
        return typ, v_stable, v_unstable
    elif det > 0 and trace < 0:
        if disc < 0:
            typ = "устойчивый фокус"
        else:
            typ = "устойчивый узел"
        return typ, None, None
    elif det > 0 and trace > 0:
        if disc < 0:
            typ = "неустойчивый фокус"
        else:
            typ = "неустойчивый узел"
        return typ, None, None
    else:
        return "негрубая", None, None

# ---- 3. Определяем типы и сепаратрисы для седёл ----
saddles = []
point_info = []

for i, pt in enumerate(equilibria):
    typ, v_stable, v_unstable = get_type_and_eigenvectors(pt[0], pt[1])
    point_info.append((pt, typ))
    if typ == "седло":
        saddles.append((pt, v_stable, v_unstable))

# ---- 4. Автоматический выбор границ области ----
all_x = [pt[0] for pt in equilibria]
all_y = [pt[1] for pt in equilibria]
x_min = min(all_x) - 2
x_max = max(all_x) + 2
y_min = min(all_y) - 2
y_max = max(all_y) + 2

# ---- 5. Строим фазовый портрет ----
x_vals = np.linspace(x_min, x_max, 30)
y_vals = np.linspace(y_min, y_max, 30)
X, Y = np.meshgrid(x_vals, y_vals)
U, V = system(X, Y)

plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, U, V, density=1.2, color='blue', linewidth=0.8)

# ---- 6. Рисуем особые точки ----
colors = {'седло': 'purple', 'устойчивый узел': 'darkgreen', 
          'неустойчивый узел': 'darkred', 'устойчивый фокус': 'lightgreen',
          'неустойчивый фокус': 'salmon', 'негрубая': 'gray'}

for pt, typ in point_info:
    plt.scatter(*pt, color=colors.get(typ, 'black'), s=120, zorder=5)
    plt.text(pt[0]+0.1, pt[1]+0.1, f'({pt[0]:.1f},{pt[1]:.1f})', fontsize=8)

# ---- 7. Рисуем сепаратрисы для каждого седла ----
eps = 1e-5
t_forward = 8
t_backward = 8

for saddle, v_stable, v_unstable in saddles:
    # Неустойчивая сепаратриса (выходит)
    if v_unstable is not None:
        for sign in [1, -1]:
            start = saddle + sign * eps * v_unstable
            try:
                sol = solve_ivp(f, [0, t_forward], start, max_step=0.01, method='RK45')
                plt.plot(sol.y[0], sol.y[1], 'orange', lw=2)
            except:
                pass
            try:
                sol_back = solve_ivp(f, [0, -t_backward], start, max_step=0.01, method='RK45')
                plt.plot(sol_back.y[0], sol_back.y[1], 'orange', lw=1.5, alpha=0.6)
            except:
                pass
    
    # Устойчивая сепаратриса (входит)
    if v_stable is not None:
        for sign in [1, -1]:
            start = saddle + sign * eps * v_stable
            try:
                sol = solve_ivp(f, [0, -t_forward], start, max_step=0.01, method='RK45')
                plt.plot(sol.y[0], sol.y[1], 'darkgreen', lw=2)
            except:
                pass
            try:
                sol_fwd = solve_ivp(f, [0, t_backward], start, max_step=0.01, method='RK45')
                plt.plot(sol_fwd.y[0], sol_fwd.y[1], 'darkgreen', lw=1.5, alpha=0.6)
            except:
                pass

# ---- 8. Нуль-изоклины ----
x_line = np.linspace(x_min, x_max, 400)
plt.plot(x_line, x_line/2, 'k--', lw=0.8, alpha=0.5, label=r"$\dot{x}=0:\; -x+2y=0$")
plt.plot(x_line, x_line + b, 'k--', lw=0.8, alpha=0.5, label=r"$\dot{x}=0:\; b+x-y=0$")
plt.axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.5, label=r"$\dot{y}=0:\; y=0$")
plt.plot(x_line, (2*x_line + a)/3, 'k--', lw=0.8, alpha=0.5, label=r"$\dot{y}=0:\; a+2x-3y=0$")

# ---- 9. Оформление ----
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Фазовый портрет: a = {a}, b = {b}')
plt.legend(loc='upper left', fontsize=7)
plt.grid(alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.show()

# ---- 10. Вывод информации о точках в консоль ----
print("\n=== ОСОБЫЕ ТОЧКИ ===")
names = ['E₁ (0,0)', 'E₂ (-b,0)', 'E₃ (-2a,-a)', 'E₄ (a-3b, a-2b)']
for i, (pt, typ) in enumerate(point_info):
    print(f"{names[i]} = ({pt[0]:.2f}, {pt[1]:.2f}) → {typ}")