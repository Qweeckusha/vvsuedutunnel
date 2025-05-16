import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
import sympy as sp

def task(n):
    print(f"\n--------------------- {n} ---------------------")


task(1)
# Заданные значения математических ожиданий X и Y
M_X = 3
M_Y = 2

# Коэффициенты и константа из Z = 8X - 5Y + 7
a = 8
b = -5
c = 7

# Вычисление математического ожидания Z
M_Z = a * M_X + b * M_Y + c

# Вывод результата
print("Математическое ожидание Z:", M_Z)

print("Математическое ожидание M(Z):", M_Z)

task(2)
# Заданные значения дисперсий
D_X = 1.5
D_Y = 1

# Коэффициенты
a = 8
b = -5

# Вычисление дисперсии Z
D_Z = (a ** 2) * D_X + (b ** 2) * D_Y

# Вывод результата
print("Дисперсия Z:", D_Z)

task(3)

# Объявляем символы для неизвестных вероятностей
p4, p5 = symbols('p4 p5')

# Уравнения из условия задачи
# 1. Сумма вероятностей равна 1: p(-2) + p(-1) + p(0) + p4 + p5 = 1
eq1 = Eq(0.2 + 0.1 + 0.2 + p4 + p5, 1)

# 2. Математическое ожидание M(X) = 0.1
# M(X) = (-2)*0.2 + (-1)*0.1 + 0*0.2 + 1*p4 + 2*p5
eq2 = Eq(-2*0.2 + -1*0.1 + 0*0.2 + 1*p4 + 2*p5, 0.1)

# Решаем систему уравнений
solution = solve((eq1, eq2), (p4, p5))
p4_value = solution[p4]
p5_value = solution[p5]

# Вычисляем дисперсию D(X)
# D(X) = M(X^2) - [M(X)]^2
# M(X^2) = (-2)^2 * 0.2 + (-1)^2 * 0.1 + 0^2 * 0.2 + 1^2 * p4 + 2^2 * p5
M_X_squared = (-2)**2 * 0.2 + (-1)**2 * 0.1 + 0**2 * 0.2 + 1**2 * p4_value + 2**2 * p5_value
M_X = 0.1  # По условию
D_X = M_X_squared - M_X**2

# Вывод результатов
print(f"p4 = {p4_value}")
print(f"p5 = {p5_value}")
print(f"D(X) = {D_X}")


task(4)
# Шаг 1: Найти значение параметра p
p = symbols('p')
# Уравнение для суммы вероятностей
equation = Eq(4*p + 0.2 + 0.3 + p + 0.4, 1)
# Решаем уравнение
p_value = solve(equation, p)[0]
# Обновленные вероятности
p1 = 4 * p_value  # 4p
p2 = 0.2
p3 = 0.3
p4 = p_value      # p
p5 = 0.4

# Значения случайной величины X
x_values = [-2, -1, 3, 8, 9]
p_values = [p1, p2, p3, p4, p5]

# Шаг 2: Вычислить математическое ожидание M(X)
M_X = sum(x * p for x, p in zip(x_values, p_values))

# Шаг 3: Вычислить дисперсию D(X)
# Вычисляем M(X^2)
M_X2 = sum(x**2 * p for x, p in zip(x_values, p_values))
# Дисперсия
D_X = M_X2 - M_X**2

# Шаг 4: Построить функцию распределения F(X)
def F(x):
    cumulative_prob = 0
    for xi, pi in zip(x_values, p_values):
        if xi <= x:
            cumulative_prob += pi
    return cumulative_prob

# Вывод результатов
print(f"Значение параметра p: {round(p_value, ndigits=2)}")
print(f"Математическое ожидание M(X): {round(M_X, ndigits=1)}")
print(f"Дисперсия D(X): {round(D_X, ndigits=2)}")

# Пример значений функции распределения F(X)
test_points = [-2, -1, 3, 7, 8, 9]
print("\nФункция распределения F(X):")
for point in test_points:
    print(f"F({round(point, ndigits=2)}) = {round(F(point), ndigits=2)}")

task(5)

# Объявляем символы
x1, x2, p1, p2 = symbols('x1 x2 p1 p2')
M_X = 3.2
D_X = 0.16
# Условия задачи
eq1 = Eq(p1 + p2, 0.8)          # p1 + p2 = 0.8
eq2 = Eq(x1 * p1 + x2 * p2, M_X) # x1*p1 + x2*p2 = 3.2
eq3 = Eq(x1**2 * p1 + x2**2 * p2, 10.4) # x1^2*p1 + x2^2*p2 = 10.4

# Перебираем возможные значения x1 и x2
solution_found = False
for i in range(0, 10):  # Расширяем диапазон значений x1
    for j in range(0, 10):  # Расширяем диапазон значений x2
        if i != j:  # Значения x1 и x2 должны быть разными
            # Подставляем x1 и x2 в уравнения
            eq2_sub = eq2.subs({x1: i, x2: j})
            eq3_sub = eq3.subs({x1: i, x2: j})

            # Решаем систему уравнений
            solution = solve((eq1, eq2_sub, eq3_sub), (p1, p2))
            # print(i, j, solution)

            # Проверяем, что решение найдено
            if solution:
                print(i, j, solution)
                # Извлекаем значения p1 и p2
                p1_val = solution[p1]
                p2_val = solution[p2]

                # Проверяем, что вероятности неотрицательны
                if p1_val >= 0 and p2_val >= 0:
                    print(f"Решение найдено:")
                    print(f"x1 = {i}, x2 = {j}")
                    print(f"p1 = {p1_val:.4f}, p2 = {p2_val:.4f}")
                    solution_found = True
                    break
    if solution_found:
        break

if not solution_found:
    print("Решение не найдено для всех значений x1 и x2.")

task(6)

# Законы распределения случайных величин X и Y
x_values = [2, 4, 6, 8]
x_probs = [0.4, 0.2, 0.1, 0.3]

y_values = [0, 1, 2]
y_probs = [0.5, 0.2, 0.3]

# Функция для вычисления математического ожидания
def calculate_mean(values, probs):
    return sum(val * prob for val, prob in zip(values, probs))

# Функция для вычисления дисперсии
def calculate_variance(values, probs, mean):
    mean_squared = mean ** 2
    expected_square = sum(val**2 * prob for val, prob in zip(values, probs))
    return expected_square - mean_squared

# Вычисление M(X) и D(X)
mean_X = calculate_mean(x_values, x_probs)
variance_X = calculate_variance(x_values, x_probs, mean_X)

# Вычисление M(Y) и D(Y)
mean_Y = calculate_mean(y_values, y_probs)
variance_Y = calculate_variance(y_values, y_probs, mean_Y)

# Линейная комбинация Z = 2X + 3Y
a = 2
b = 3

# Математическое ожидание Z
mean_Z = a * mean_X + b * mean_Y

# Дисперсия Z
variance_Z = a**2 * variance_X + b**2 * variance_Y

# Вывод результатов
print(f"Математическое ожидание M(X) = {mean_X}")
print(f"Дисперсия D(X) = {round(variance_X, ndigits=2)}")
print(f"Математическое ожидание M(Y) = {mean_Y}")
print(f"Дисперсия D(Y) = {round(variance_Y, ndigits=2)}")
print(f"Математическое ожидание M(Z) = {mean_Z}")
print(f"Дисперсия D(Z) = {round(variance_Z, ndigits=3)}")

task(7)

# Значения X и их вероятности
x_values = [3, 4, 5, 6, 7]
probabilities = [0.1, 0.2, 0.4, 0.1, 0.2]

# a) Функция распределения вероятностей
cumulative_probabilities = np.cumsum(probabilities)

# Построение графика
plt.step(x_values, cumulative_probabilities, where='post')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Функция распределения вероятностей')
plt.grid(True)
plt.show()

# b) Вероятность дополнительных расходов
probability_additional_costs = probabilities[3] + probabilities[4]
print(f"Вероятность дополнительных расходов: {round(probability_additional_costs, ndigits=2)}")

# c) Математическое ожидание и дисперсия
mean_X = sum(x * p for x, p in zip(x_values, probabilities))
variance_X = sum(x**2 * p for x, p in zip(x_values, probabilities)) - mean_X**2

print(f"Математическое ожидание M(X): {round(mean_X, ndigits=2)}")
print(f"Дисперсия D(X): {round(variance_X, ndigits=2)}")

task(8)
# Определение переменной
x = sp.symbols('x')

# Плотность вероятности
f_x = x / 2

# Математическое ожидание M(X)
M_X = sp.integrate(x * f_x, (x, 0, 2))
print(f"Математическое ожидание M(X): {M_X}")

# Вычисление M(X^2)
M_X2 = sp.integrate(x**2 * f_x, (x, 0, 2))
print(f"M(X^2): {M_X2}")

# Дисперсия D(X)
D_X = M_X2 - M_X**2
print(f"Дисперсия D(X): {D_X}")

task(9)
import numpy as np
import matplotlib.pyplot as plt

# Определение функций
def F(x):
    """ Интегральная функция """
    return np.piecewise(
        x,
        [x <= 0, (x > 0) & (x <= 6), x > 6],
        [0, lambda x: x**2 / 36, 1]
    )

def f(x):
    """ Дифференциальная функция """
    return np.piecewise(
        x,
        [x <= 0, (x > 0) & (x <= 6), x > 6],
        [0, lambda x: x / 18, 0]
    )

# Диапазон значений x
x = np.linspace(-1, 8, 1000)

# Вычисление значений функций
F_values = F(x)
f_values = f(x)

# График интегральной функции
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(x, F_values, label='$F(x)$')
plt.title('График интегральной функции $F(x)$')
plt.xlabel('$x$')
plt.ylabel('$F(x)$')
plt.legend()
plt.grid(True)

# График дифференциальной функции
plt.subplot(1, 2, 2)
plt.plot(x, f_values, label='$f(x)$', color='orange')
plt.title('График дифференциальной функции $f(x)$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод результатов
print(f"Математическое ожидание M(X): {M_X}")
print(f"Дисперсия D(X): {D_X}")

task(10)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Функция распределения F(x)
def F(x):
    if x <= 0:
        return 0
    elif x <= 2:
        return x**2 / 4
    else:
        return 1

# Плотность вероятности f(x)
def f(x):
    if x <= 0 or x > 2:
        return 0
    else:
        return x / 2

# Построение графиков F(x) и f(x)
x_values = np.linspace(-1, 3, 1000)
F_values = np.array([F(x) for x in x_values])
f_values = np.array([f(x) for x in x_values])

plt.figure(figsize=(12, 6))

# График F(x)
plt.subplot(1, 2, 1)
plt.plot(x_values, F_values, label='$F(x)$')
plt.title('График функции распределения $F(x)$')
plt.xlabel('$x$')
plt.ylabel('$F(x)$')
plt.legend()
plt.grid(True)

# График f(x)
plt.subplot(1, 2, 2)
plt.plot(x_values, f_values, label='$f(x)$')
plt.title('График плотности вероятности $f(x)$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Вычисление вероятностей
P_X_equals_1 = f(1)
P_X_less_1 = F(1)
P_1_leq_X_less_2 = F(2) - F(1)

# Вычисление математического ожидания M(X)
M_X = quad(lambda x: x * f(x), 0, 2)[0]

# Вычисление дисперсии D(X)
M_X2 = quad(lambda x: x**2 * f(x), 0, 2)[0]
D_X = M_X2 - M_X**2

# Вывод результатов
print(f"P(X = 1) = {P_X_equals_1}")
print(f"P(X < 1) = {P_X_less_1}")
print(f"P(1 <= X < 2) = {P_1_leq_X_less_2}")
print(f"M(X) = {M_X}")
print(f"D(X) = {D_X}")

task(11)

# Определение переменной
x = sp.symbols('x')

# Плотность вероятности
A = sp.symbols('A')
f_x = sp.Piecewise((0, x <= 1), (A / x**4, x > 1))

# a) Найти значение A
integral_A = sp.integrate(f_x, (x, 1, sp.oo))
A_value = sp.solve(integral_A - 1, A)[0]

# b) Функция распределения F(x)
F_x = sp.Piecewise((0, x <= 1), (1 - 1/x**3, x > 1))

# c) Математическое ожидание M(X)
M_X = sp.integrate(x * f_x.subs(A, A_value), (x, 1, sp.oo))

# Дисперсия D(X)
M_X2 = sp.integrate(x**2 * f_x.subs(A, A_value), (x, 1, sp.oo))
D_X = M_X2 - M_X**2

# Вывод результатов
print(f"Значение A: {A_value}")
print(f"Функция распределения F(x):")
print(F_x)
print(f"Математическое ожидание M(X): {M_X}")
print(f"Дисперсия D(X): {D_X}")