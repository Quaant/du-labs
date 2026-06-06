import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig

# ========== Введите параметры здесь ==========
a = 0.0
b = 0.0
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
def is_jordan_cell(J):
    vals, vecs = eig(J)
    if abs(vals[0] - vals[1]) > 1e-8:
        return False, None
    lam = vals[0]
    M = J - lam * np.eye(2)
    if np.linalg.matrix_rank(M, tol=1e-8) == 1:
        return True, np.real(lam)
    return False, None

def get_jordan_direction(J, lam):
    M = J - lam * np.eye(2)
    if abs(M[0,0]) > 1e-8:
        v = np.array([M[0,1], -M[0,0]])
    else:
        v = np.array([M[1,1], -M[1,0]])
    v = v / np.linalg.norm(v)
    return v

def get_type_and_vectors(x, y):
    J = np.array([
        [-b - 2*x + 3*y, 2*b + 3*x - 4*y],
        [-2*y, -a - 2*x + 6*y]
    ])
    trace = np.trace(J)
    det = np.linalg.det(J)
    disc = trace**2 - 4*det
    vals, vecs = eig(J)
    
    idx = np.argsort(np.abs(vals))
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    is_jordan, lam = is_jordan_cell(J)
    if is_jordan:
        if lam < 0:
            return "жорданов узел (уст)", None, None, None, None, lam
        else:
            return "жорданов узел (неуст)", None, None, None, None, lam
    
    if abs(trace) < 1e-8 and det > 0:
        return "центр", None, None, None, None, None
    
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
        return typ, v_stable, v_unstable, None, None, None
    elif det > 0 and trace < 0:
        if disc < 0:
            typ = "устойчивый фокус"
            return typ, None, None, None, None, None
        else:
            typ = "устойчивый узел"
            v_leading = np.real(vecs[:, 0])
            v_leading = v_leading / np.linalg.norm(v_leading)
            return typ, None, None, v_leading, None, None
    elif det > 0 and trace > 0:
        if disc < 0:
            typ = "неустойчивый фокус"
            return typ, None, None, None, None, None
        else:
            typ = "неустойчивый узел"
            v_leading = np.real(vecs[:, 0])
            v_leading = v_leading / np.linalg.norm(v_leading)
            return typ, None, None, v_leading, None, None
    else:
        return "негрубая", None, None, None, None, None

def get_saddle_node_info(x, y):
    J = np.array([
        [-b - 2*x + 3*y, 2*b + 3*x - 4*y],
        [-2*y, -a - 2*x + 6*y]
    ])
    det = np.linalg.det(J)
    trace = np.trace(J)
    vals, vecs = eig(J)
    
    if abs(det) > 1e-8 or abs(trace) < 1e-8:
        return None, None, None, None
    
    zero_idx = None
    non_zero_idx = None
    for i, val in enumerate(vals):
        if abs(val) < 1e-10:
            zero_idx = i
        else:
            non_zero_idx = i
    
    if zero_idx is not None and non_zero_idx is not None:
        v_zero = np.real(vecs[:, zero_idx])
        v_zero = v_zero / np.linalg.norm(v_zero)
        v_nonzero = np.real(vecs[:, non_zero_idx])
        v_nonzero = v_nonzero / np.linalg.norm(v_nonzero)
        if np.real(vals[non_zero_idx]) < 0:
            return "седлоузел", v_zero, v_nonzero, "уст"
        else:
            return "седлоузел", v_zero, v_nonzero, "неуст"
    return None, None, None, None

def draw_trajectory(start, t_direction, color, alpha=1.0, max_time=50):
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

def draw_leading_direction(pt, v_leading, length=0.8):
    if v_leading is not None:
        plt.arrow(pt[0] - length * v_leading[0], pt[1] - length * v_leading[1],
                  2 * length * v_leading[0], 2 * length * v_leading[1],
                  head_width=0.15, head_length=0.2, fc='red', ec='red', alpha=0.7)

def draw_center(pt, num_trajectories=6, radius=0.5):
    angles = np.linspace(0, 2*np.pi, num_trajectories + 1)[:-1]
    for angle in angles:
        start = pt + radius * np.array([np.cos(angle), np.sin(angle)])
        try:
            sol = solve_ivp(f, [0, 20], start, max_step=0.01, method='RK45')
            plt.plot(sol.y[0], sol.y[1], 'cyan', lw=1.5, alpha=0.7)
        except:
            pass

def draw_jordan_node(pt, lam, length=1.0):
    J = np.array([
        [-b - 2*pt[0] + 3*pt[1], 2*b + 3*pt[0] - 4*pt[1]],
        [-2*pt[1], -a - 2*pt[0] + 6*pt[1]]
    ])
    v = get_jordan_direction(J, lam)
    plt.arrow(pt[0] - length * v[0], pt[1] - length * v[1],
              2 * length * v[0], 2 * length * v[1],
              head_width=0.15, head_length=0.2, fc='red', ec='red', alpha=0.8)
    
    eps = 1e-5
    offsets = [0.3, 0.6, 0.9]
    for offset in offsets:
        perp = np.array([-v[1], v[0]])
        start = pt + offset * perp
        if lam < 0:
            draw_trajectory(start, 1, 'lightgreen', alpha=0.7, max_time=30)
            draw_trajectory(start, -1, 'lightgreen', alpha=0.5, max_time=30)
        else:
            draw_trajectory(start, -1, 'salmon', alpha=0.7, max_time=30)
            draw_trajectory(start, 1, 'salmon', alpha=0.5, max_time=30)

# ---- 7. Собираем информацию о точках ----
saddles = []
saddle_nodes = []
nodes = []
centers = []
jordan_nodes = []
point_info = []

for i, pt in enumerate(equilibria):
    typ, _, _, _, _, _ = get_type_and_vectors(pt[0], pt[1])
    if typ == "центр":
        point_info.append((pt, "центр", names[i]))
        centers.append(pt)
        continue
    
    typ2, v_zero, v_nonzero, dir_nonzero = get_saddle_node_info(pt[0], pt[1])
    if typ2 == "седлоузел":
        point_info.append((pt, "седлоузел", names[i]))
        saddle_nodes.append((pt, v_zero, v_nonzero, dir_nonzero))
        continue
    
    typ, _, _, _, _, lam = get_type_and_vectors(pt[0], pt[1])
    if "жорданов" in typ:
        point_info.append((pt, typ, names[i]))
        jordan_nodes.append((pt, lam))
        continue
    
    typ, v_stable, v_unstable, v_leading, _, _ = get_type_and_vectors(pt[0], pt[1])
    point_info.append((pt, typ, names[i]))
    if typ == "седло":
        saddles.append((pt, v_stable, v_unstable))
    elif typ in ["устойчивый узел", "неустойчивый узел"] and v_leading is not None:
        nodes.append((pt, v_leading))

# ---- 8. Границы области рисования ----
all_x = [pt[0] for pt in equilibria]
all_y = [pt[1] for pt in equilibria]
x_min = min(all_x) - 2
x_max = max(all_x) + 2
y_min = min(all_y) - 2
y_max = max(all_y) + 2

if a == 4 and b == 3:
    x_min, x_max, y_min, y_max = -10, 2, -6, 4
if a == 0 and b == 1:
    x_min, x_max, y_min, y_max = -4, 2, -3, 3

# ---- 9. Рисуем фазовый портрет ----
x_vals = np.linspace(x_min, x_max, 30)
y_vals = np.linspace(y_min, y_max, 30)
X, Y = np.meshgrid(x_vals, y_vals)
U, V = system(X, Y)

plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, U, V, density=1.2, color='blue', linewidth=0.8)

# ---- 10. Особые точки ----
colors = {
    'седло': 'purple',
    'седлоузел': 'orange',
    'центр': 'magenta',
    'устойчивый узел': 'darkgreen',
    'неустойчивый узел': 'darkred',
    'устойчивый фокус': 'lightgreen',
    'неустойчивый фокус': 'salmon',
    'жорданов узел (уст)': 'darkgreen',
    'жорданов узел (неуст)': 'darkred',
    'негрубая': 'gray'
}
markers = {
    'седло': 's',
    'седлоузел': 'D',
    'центр': '*',
    'устойчивый узел': 'o',
    'неустойчивый узел': 'o',
    'устойчивый фокус': 'o',
    'неустойчивый фокус': 'o',
    'жорданов узел (уст)': 'd',
    'жорданов узел (неуст)': 'd',
    'негрубая': 'o'
}

for pt, typ, name in point_info:
    plt.scatter(*pt, color=colors.get(typ, 'black'), marker=markers.get(typ, 'o'), 
                s=120, zorder=5)
    plt.text(pt[0]+0.1, pt[1]+0.2, f'{name}\n({pt[0]:.1f},{pt[1]:.1f})\n{typ}',
             fontsize=7, ha='left', va='bottom')

# ---- 11. Центры ----
for pt in centers:
    draw_center(pt)

# ---- 12. Жордановы узлы ----
for pt, lam in jordan_nodes:
    draw_jordan_node(pt, lam)

# ---- 13. Ведущие направления узлов ----
for pt, v_leading in nodes:
    draw_leading_direction(pt, v_leading)

# ---- 14. Сепаратрисы седёл ----
eps = 1e-5
for saddle, v_stable, v_unstable in saddles:
    if v_unstable is not None:
        draw_trajectory(saddle + eps * v_unstable, 1, 'green', alpha=0.9)
        draw_trajectory(saddle - eps * v_unstable, 1, 'green', alpha=0.9)
    if v_stable is not None:
        draw_trajectory(saddle + eps * v_stable, -1, 'green', alpha=0.9)
        draw_trajectory(saddle - eps * v_stable, -1, 'green', alpha=0.9)

# ---- 15. Сепаратрисы седлоузлов (улучшенная визуализация с направлениями) ----
for saddle_pt, v_zero, v_nonzero, dir_nonzero in saddle_nodes:
    # Главная траектория (вдоль нулевого направления) — синяя
    for sign in [1, -1]:
        start = saddle_pt + sign * eps * v_zero
        draw_trajectory(start, 1, 'blue', alpha=0.9, max_time=30)
        draw_trajectory(start, -1, 'blue', alpha=0.6, max_time=30)
        # Добавляем стрелки в обе стороны вдоль синей линии
        # Берём точку на небольшом удалении от седлоузла
        far_start = saddle_pt + sign * 0.5 * v_zero
        # Стрелка от седлоузла
        plt.arrow(saddle_pt[0], saddle_pt[1], 
                  far_start[0] - saddle_pt[0], far_start[1] - saddle_pt[1],
                  head_width=0.1, head_length=0.15, fc='red', ec='red', alpha=0.6)
        # Стрелка к седлоузлу (с противоположной стороны)
        plt.arrow(far_start[0], far_start[1], 
                  saddle_pt[0] - far_start[0], saddle_pt[1] - far_start[1],
                  head_width=0.1, head_length=0.15, fc='red', ec='red', alpha=0.6)
    
    # Сепаратриса вдоль ненулевого направления — зелёная + красная стрелка (одна)
    if dir_nonzero == "уст":
        for sign in [1, -1]:
            start = saddle_pt + sign * eps * v_nonzero
            draw_trajectory(start, -1, 'green', alpha=0.9, max_time=30)
            # Стрелка в сторону седлоузла
            far_start = saddle_pt + sign * 0.5 * v_nonzero
            plt.arrow(far_start[0], far_start[1], 
                      saddle_pt[0] - far_start[0], saddle_pt[1] - far_start[1],
                      head_width=0.12, head_length=0.18, fc='red', ec='red', alpha=0.8)
    else:
        for sign in [1, -1]:
            start = saddle_pt + sign * eps * v_nonzero
            draw_trajectory(start, 1, 'green', alpha=0.9, max_time=30)
            # Стрелка от седлоузла
            far_start = saddle_pt + sign * 0.5 * v_nonzero
            plt.arrow(saddle_pt[0], saddle_pt[1], 
                      far_start[0] - saddle_pt[0], far_start[1] - saddle_pt[1],
                      head_width=0.12, head_length=0.18, fc='red', ec='red', alpha=0.8)
# ---- 16. Нуль-изоклины ----
x_line = np.linspace(x_min, x_max, 400)
plt.plot(x_line, x_line/2, 'k--', lw=0.8, alpha=0.6, label=r"$\dot{x}=0:\; y=x/2$")
plt.plot(x_line, x_line + b, 'k--', lw=0.8, alpha=0.6, label=r"$\dot{x}=0:\; y=x+b$")
plt.axhline(0, color='k', linestyle='--', lw=0.8, alpha=0.6, label=r"$\dot{y}=0:\; y=0$")
plt.plot(x_line, (2*x_line + a)/3, 'k--', lw=0.8, alpha=0.6, label=r"$\dot{y}=0:\; y=(2x+a)/3$")

# ---- 17. Оформление ----
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Фазовый портрет: a = {a}, b = {b}\n'
          f'зелёные — сепаратрисы, синие — главные траектории седлоузлов,\n'
          f'голубые — центр, красные стрелки — направления')
plt.legend(loc='upper left', fontsize=7)
plt.grid(alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.show()

# ---- 18. Вывод в консоль ----
print("\n=== ОСОБЫЕ ТОЧКИ ===")
for pt, typ, name in point_info:
    print(f"{name} = ({pt[0]:.2f}, {pt[1]:.2f}) → {typ}")