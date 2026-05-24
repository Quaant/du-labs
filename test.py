import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig

# ========== Введите параметры здесь ==========
a = -3.0
b = -2.0
# ============================================

def system(X, Y):
    dx = (-X + 2*Y) * (b + X - Y)
    dy = -Y * (a + 2*X - 3*Y)
    return dx, dy

def f(t, z):
    return system(z[0], z[1])

# ---- 1. Особые точки ----
E1 = np.array([0.0, 0.0])
E2 = np.array([-b, 0.0])
E3 = np.array([-2*a, -a])
E4 = np.array([a - 3*b, a - 2*b])

equilibria = [E1, E2, E3, E4]
names = ['E₁ (0,0)', 'E₂ (-b,0)', 'E₃ (-2a,-a)', 'E₄ (a-3b, a-2b)']

# ---- 2. Определение типа точки и собственных векторов ----
def get_type_and_vectors(x, y):
    J = np.array([
        [-b - 2*x + 3*y, 2*b + 3*x - 4*y],
        [-2*y, -a - 2*x + 6*y]
    ])
    trace = np.trace(J)
    det = np.linalg.det(J)
    disc = trace**2 - 4*det
    vals, vecs = eig(J)
    
    # Сортируем собственные значения по модулю
    idx = np.argsort(np.abs(vals))
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    if det < 0:
        typ = "седло"
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
        return typ, v_stable, v_unstable, None
    elif det > 0 and trace < 0:
        if disc < 0:
            typ = "устойчивый фокус"
            return typ, None, None, None
        else:
            typ = "устойчивый узел"
            # Ведущее направление — собственный вектор с меньшим |λ|
            v_leading = np.real(vecs[:, 0])  # меньшее по модулю
            v_leading = v_leading / np.linalg.norm(v_leading)
            return typ, None, None, v_leading
    elif det > 0 and trace > 0:
        if disc < 0:
            typ = "неустойчивый фокус"
            return typ, None, None, None
        else:
            typ = "неустойчивый узел"
            v_leading = np.real(vecs[:, 0])  # меньшее по модулю
            v_leading = v_leading / np.linalg.norm(v_leading)
            return typ, None, None, v_leading
    else:
        return "негрубая", None, None, None

# ---- 3. Функция для рисования сепаратрисы ----
def draw_separatrix(start, t_direction, color, alpha=1.0, max_time=50):
    try:
        sol = solve_ivp(f, [0, t_direction * max_time], start, 
                        max_step=0.01, method='RK45')
        x_traj, y_traj = sol.y[0], sol.y[1]
        mask = (x_traj >= x_min) & (x_traj <= x_max) & (y_traj >= y_min) & (y_traj <= y_max)
        if np.any(mask):
            last_valid = np.where(mask)[0]
            if len(last_valid) > 0:
                idx = last_valid[-1]
                plt.plot(x_traj[:idx+1], y_traj[:idx+1], color, lw=2.5, alpha=alpha)
    except:
        pass

# ---- 4. Функция для рисования ведущего направления узла ----
def draw_leading_direction(pt, v_leading, length=0.8):
    """Рисует красную стрелку вдоль ведущего направления"""
    if v_leading is not None:
        # Стрелка в обе стороны от точки
        plt.arrow(pt[0] - length * v_leading[0], pt[1] - length * v_leading[1],
                  2 * length * v_leading[0], 2 * length * v_leading[1],
                  head_width=0.15, head_length=0.2, fc='red', ec='red', alpha=0.7)

# ---- 5. Собираем информацию о точках ----
saddles = []
nodes = []  # для узлов с ведущим направлением
point_info = []
for i, pt in enumerate(equilibria):
    typ, v_stable, v_unstable, v_leading = get_type_and_vectors(pt[0], pt[1])
    point_info.append((pt, typ, names[i]))
    if typ == "седло":
        saddles.append((pt, v_stable, v_unstable))
    elif typ in ["устойчивый узел", "неустойчивый узел"] and v_leading is not None:
        nodes.append((pt, v_leading))

# ---- 6. Границы области рисования ----
all_x = [pt[0] for pt in equilibria]
all_y = [pt[1] for pt in equilibria]
x_min = min(all_x) - 2
x_max = max(all_x) + 2
y_min = min(all_y) - 2
y_max = max(all_y) + 2

# Для a=4,b=3 расширим область
if a == 4 and b == 3:
    x_min = -10
    x_max = 2
    y_min = -6
    y_max = 4

# ---- 7. Рисуем фазовый портрет ----
x_vals = np.linspace(x_min, x_max, 30)
y_vals = np.linspace(y_min, y_max, 30)
X, Y = np.meshgrid(x_vals, y_vals)
U, V = system(X, Y)

plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, U, V, density=1.2, color='blue', linewidth=0.8)

# ---- 8. Особые точки ----
colors = {
    'седло': 'purple',
    'устойчивый узел': 'darkgreen',
    'неустойчивый узел': 'darkred',
    'устойчивый фокус': 'lightgreen',
    'неустойчивый фокус': 'salmon',
    'негрубая': 'gray'
}

for pt, typ, name in point_info:
    plt.scatter(*pt, color=colors.get(typ, 'black'), s=120, zorder=5)
    plt.text(pt[0]+0.1, pt[1]+0.2, f'{name}\n({pt[0]:.1f},{pt[1]:.1f})',
             fontsize=8, ha='left', va='bottom')

# ---- 9. Ведущие направления узлов (красным) ----
for pt, v_leading in nodes:
    draw_leading_direction(pt, v_leading)

# ---- 10. Сепаратрисы седёл (зелёным) ----
eps = 1e-5
for saddle, v_stable, v_unstable in saddles:
    if v_unstable is not None:
        draw_separatrix(saddle + eps * v_unstable, 1, 'green', alpha=0.9)
        draw_separatrix(saddle - eps * v_unstable, 1, 'green', alpha=0.9)
    if v_stable is not None:
        draw_separatrix(saddle + eps * v_stable, -1, 'green', alpha=0.9)
        draw_separatrix(saddle - eps * v_stable, -1, 'green', alpha=0.9)

# ---- 11. Нуль-изоклины ----
x_line = np.linspace(x_min, x_max, 400)
plt.plot(x_line, x_line/2, 'k--', lw=0.8, alpha=0.6, label=r"$\dot{x}=0:\; y=x/2$")
plt.plot(x_line, x_line + b, 'k--', lw=0.8, alpha=0.6, label=r"$\dot{x}=0:\; y=x+b$")
plt.axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.6, label=r"$\dot{y}=0:\; y=0$")
plt.plot(x_line, (2*x_line + a)/3, 'k--', lw=0.8, alpha=0.6, label=r"$\dot{y}=0:\; y=(2x+a)/3$")

# ---- 12. Оформление ----
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Фазовый портрет: a = {a}, b = {b}\nзелёные — сепаратрисы, красные стрелки — ведущие направления узлов')
plt.legend(loc='upper left', fontsize=7)
plt.grid(alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.show()

# ---- 13. Вывод в консоль ----
print("\n=== ОСОБЫЕ ТОЧКИ ===")
for pt, typ, name in point_info:
    print(f"{name} = ({pt[0]:.2f}, {pt[1]:.2f}) → {typ}")