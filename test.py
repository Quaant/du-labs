import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ========== Параметры системы ==========
a = 0.0
b = 0.0

def system(X, Y):
    dx = (-X + 2*Y) * (b + X - Y)
    dy = -Y * (a + 2*X - 3*Y)
    return dx, dy

def f(t, z):
    return system(z[0], z[1])

# ---- Границы области рисования ----
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# ---- Рисуем фазовый портрет (векторное поле) ----
x_vals = np.linspace(x_min, x_max, 25)
y_vals = np.linspace(y_min, y_max, 25)
X, Y = np.meshgrid(x_vals, y_vals)
U, V = system(X, Y)

plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, U, V, density=1.2, color='blue', linewidth=0.8)

# ---- Особая точка ----
plt.scatter(0, 0, color='gray', s=120, zorder=5, label='Негрубая (0,0)')
plt.text(0.1, 0.1, 'E(0,0)\nвырожденная', fontsize=8)

# ---- Добавляем характерные траектории вручную ----

# 1. Ось x (y=0) — инвариантна
t_span = [0, 5]
# Справа от 0 (x>0) — движение влево к 0
sol = solve_ivp(f, t_span, [1.5, 0], max_step=0.01)
plt.plot(sol.y[0], sol.y[1], 'green', lw=2, label='Ось x (вход справа)')
# Слева от 0 (x<0) — движение влево от 0
sol = solve_ivp(f, t_span, [-1.5, 0], max_step=0.01)
plt.plot(sol.y[0], sol.y[1], 'green', lw=2)
# Добавляем стрелки вручную (примерно)
plt.arrow(1.2, 0, -0.4, 0, head_width=0.07, head_length=0.1, fc='green', ec='green')
plt.arrow(-1.2, 0, -0.4, 0, head_width=0.07, head_length=0.1, fc='green', ec='green')

# 2. Диагональные направления y = ±x/√2 (примерно 0.707 и -0.707)
k_pos = 1/np.sqrt(2)   # ≈ 0.7071
k_neg = -1/np.sqrt(2)

# Положительный луч (выход/вход)
start_pos = [0.5, k_pos*0.5]
sol = solve_ivp(f, [0, 5], start_pos, max_step=0.01)
plt.plot(sol.y[0], sol.y[1], 'orange', lw=2, label='Диагональ y ≈ 0.707x')
# Обратное время
sol_back = solve_ivp(f, [0, -5], start_pos, max_step=0.01)
plt.plot(sol_back.y[0], sol_back.y[1], 'orange', lw=2, alpha=0.7)

# Отрицательный луч
start_neg = [0.5, k_neg*0.5]
sol = solve_ivp(f, [0, 5], start_neg, max_step=0.01)
plt.plot(sol.y[0], sol.y[1], 'orange', lw=2)
sol_back = solve_ivp(f, [0, -5], start_neg, max_step=0.01)
plt.plot(sol_back.y[0], sol_back.y[1], 'orange', lw=2, alpha=0.7)

# 3. Вертикаль x=0 (y ≠ 0) — параболические траектории
start_vert = [0, 1.2]
sol = solve_ivp(f, [0, 5], start_vert, max_step=0.01)
plt.plot(sol.y[0], sol.y[1], 'magenta', lw=2, label='x=0 (парабола)')
sol_back = solve_ivp(f, [0, -5], start_vert, max_step=0.01)
plt.plot(sol_back.y[0], sol_back.y[1], 'magenta', lw=2, alpha=0.7)

# Симметричная траектория вниз
start_vert2 = [0, -1.2]
sol = solve_ivp(f, [0, 5], start_vert2, max_step=0.01)
plt.plot(sol.y[0], sol.y[1], 'magenta', lw=2)
sol_back = solve_ivp(f, [0, -5], start_vert2, max_step=0.01)
plt.plot(sol_back.y[0], sol_back.y[1], 'magenta', lw=2, alpha=0.7)

# ---- Оформление ----
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Фазовый портрет: a = {a}, b = {b}\n'
          f'Вырожденная точка (0,0), зелёные — ось x, оранжевые — диагонали, '
          f'малиновые — x=0')
plt.legend(loc='upper left', fontsize=7)
plt.grid(alpha=0.3)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.show()

# ---- Вывод информации ----
print("\n=== ОСОБАЯ ТОЧКА ===")
print(f"E(0,0) — полностью вырожденная (матрица Якоби = 0)")
print("Поведение: ось x инвариантна (вход справа, выход налево)")
print("Два диагональных направления (y = ±x/√2) — сепаратрисы")
print("Вертикаль x=0 — параболические траектории")