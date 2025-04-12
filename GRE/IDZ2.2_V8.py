import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


print('---------------- 1 часть ----------------')

# Определение символа x
x = sp.symbols('x')

# Интегральная функция распределения F(x)
F = sp.Piecewise(
    (0, x <= 0),
    (x**2 / 100, (x > 0) & (x <= 10)),
    (1, x > 10)
)

# а) Найти дифференциальную функцию (плотность вероятности)
f = sp.diff(F, x)
f_simplified = sp.simplify(f)

# б) Найти математическое ожидание M(X)
# M(X) = ∫ x * f(x) dx в пределах от 0 до 10
M_X = sp.integrate(x * f_simplified, (x, 0, 10))

# Найти дисперсию D(X)
# D(X) = M(X^2) - [M(X)]^2
# M(X^2) = ∫ x^2 * f(x) dx в пределах от 0 до 10
M_X_squared = sp.integrate(x**2 * f_simplified, (x, 0, 10))
D_X = M_X_squared - M_X**2

# Вывод результатов
print("а) Дифференциальная функция (плотность вероятности):")
print(f"f(x) = {f_simplified} - так выглядит система функций, через запятую указаны условия") #(0, True) - условие
# по умолчанию, если не выполнилось ни одно

print("\nб) Математическое ожидание и дисперсия:")
print(f"M(X) = {M_X}")
print(f"D(X) = {D_X}")

# Построение графиков
# Преобразуем символьные выражения в числовые функции
F_func = sp.lambdify(x, F, 'numpy')
f_func = sp.lambdify(x, f_simplified, 'numpy')

# Диапазон значений x
x_vals = np.linspace(-1, 12, 1000)

# Вычисление значений функций
F_vals = F_func(x_vals)
f_vals = f_func(x_vals)

# Построение графиков
plt.figure(figsize=(12, 6))

# График интегральной функции
plt.subplot(1, 2, 1)
plt.plot(x_vals, F_vals, label='$F(x)$', color='blue')
plt.title('Интегральная функция распределения $F(x)$')
plt.xlabel('$x$')
plt.ylabel('$F(x)$')
plt.legend()
plt.grid()

# График дифференциальной функции
plt.subplot(1, 2, 2)
plt.plot(x_vals, f_vals, label='$f(x)$', color='orange')
plt.title('Дифференциальная функция (плотность вероятности) $f(x)$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Задача 2:
print('---------------- 2 часть ----------------')
# Плотность вероятности f(x)
f_x = sp.Piecewise((0, x <= 0), (2*x, (x > 0) & (x <= 1)), (0, True)) # (0, True) - условие по умолчанию, если не выполнилось ни одно

# Вывод выражения плотности вероятности
print("Плотность вероятности:")
print(f"f(x) = {f_x} - так выглядит система функций, через запятую указаны условия")

# Математическое ожидание M(X)
M_X = sp.integrate(x * f_x, (x, -sp.oo, sp.oo))
print("\nМатематическое ожидание:")
print(f"M(X) = {M_X}")

# Дисперсия D(X)
# Сначала вычислим M(X^2)
M_X_squared = sp.integrate(x**2 * f_x, (x, -sp.oo, sp.oo))
# Теперь найдем дисперсию
D_X = M_X_squared - M_X**2
print("\nДисперсия:")
print(f"D(X) = {D_X}")