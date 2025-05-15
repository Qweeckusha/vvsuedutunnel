from math import comb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

print("---------------- 1 ----------------")
minimum = 6
total = 45
other = total-minimum # 36
p3 = (comb(minimum, 3) * comb(other, minimum-3))/(comb(total, minimum))
print(f"Вероятность P(X=3): {p3:.9f}")
p4 = (comb(minimum, 4) * comb(other, minimum-4))/(comb(total, minimum))
print(f"Вероятность P(X=4): {p4:.9f}")
p5 = (comb(minimum, 5) * comb(other, minimum-5))/(comb(total, minimum))
print(f"Вероятность P(X=5): {p5:.9f}")
p6 = (comb(minimum, 6) * comb(other, minimum-6))/(comb(total, minimum))
print(f"Вероятность P(X=6): {p6:.9f}\n")

print("---------------- 2 ----------------")
calls = 5
p_accept = 0.4
q = 1 - p_accept
p1 = q**(calls-(calls+1-1)) * 0.4
print(f"Вероятность P(X=1): {p1}")
p2 = q**(calls-(calls+1-2)) * 0.4
print(f"Вероятность P(X=2): {p2}")
p3 = q**(calls-(calls+1-3)) * 0.4
print(f"Вероятность P(X=3): {p3}")
p4 = q**(calls-(calls+1-4)) * 0.4
print(f"Вероятность P(X=4): {round(p4, ndigits=4)}")
p5 = 1 - (p1 + p2 + p3 + p4)
print(f"Вероятность P(X=5): {round(p5, ndigits=4)}\n")

print("---------------- 3 ----------------")
p_A = 0.7
p_B = 0.9
q_A = 1 - p_A
q_B = 1 - p_B
p0 = q_A * q_B
print(f"Вероятность P(X=0): {p0}")
p1 = p_A * q_B + q_A * p_B
print(f"Вероятность P(X=1): {round(p1, ndigits=2)}")
p2 = p_A * p_B
print(f"Вероятность P(X=2): {round(p2, ndigits=2)}")

print("---------------- 4 ----------------")
# Законы распределения X и Y
x_values = [2, 4, 6, 8]
p_x = [0.4, 0.2, 0.1, 0.3]

y_values = [0, 1, 2]
p_y = [0.5, 0.2, 0.3]

# Создаем пустой словарь для хранения значений Z и их вероятностей
z_distribution = {}

# Перебираем все возможные комбинации (x, y)
for x in x_values:
    for y in y_values:
        z = 2 * x + 3 * y  # Вычисляем Z
        prob_z = p_x[x_values.index(x)] * p_y[y_values.index(y)]  # Вероятность для данной пары (x, y)

        # Обновляем словарь с вероятностями
        if z in z_distribution:
            z_distribution[z] += prob_z
        else:
            z_distribution[z] = prob_z

# Сортируем значения Z и их вероятности
z_sorted = sorted(z_distribution.items())

# Выводим результат
print("Ряд распределения Z:")
for z, prob in z_sorted:
    print(f"Z = {z}: P(Z = {z}) = {round(prob, ndigits=4)}")

# Построение многоугольника распределения
z_values, probabilities = zip(*z_sorted)
plt.plot(z_values, probabilities, marker='o')
plt.xlabel('Значения Z')
plt.ylabel('Вероятность P(Z)')
plt.title('#4 Распределение Z')
plt.grid(True)
plt.show()

print("---------------- 5 ----------------")
# Законы распределения X и Y
x_values = [0, 2, 4]
p_x = [0.5, 0.2, 0.3]

y_values = [-2, 0, 2]
p_y = [0.1, 0.6, 0.3]

# Создаем пустой словарь для хранения значений Z и их вероятностей
z_distribution = {}

# Перебираем все возможные комбинации (x, y)
for x in x_values:
    for y in y_values:
        z = x * y  # Вычисляем Z
        prob_z = p_x[x_values.index(x)] * p_y[y_values.index(y)]  # Вероятность для данной пары (x, y)

        # Обновляем словарь с вероятностями
        if z in z_distribution:
            z_distribution[z] += prob_z
        else:
            z_distribution[z] = prob_z

# Сортируем значения Z и их вероятности
z_sorted = sorted(z_distribution.items())

# Выводим результат
print("Ряд распределения Z = X * Y:")
for z, prob in z_sorted:
    print(f"Z = {z}: P(Z = {z}) = {round(prob, ndigits=4)}")

# Построение многоугольника распределения
z_values, probabilities = zip(*z_sorted)
plt.plot(z_values, probabilities, marker='o')
plt.xlabel('Значения Z')
plt.ylabel('Вероятность P(Z)')
plt.title('#5 Распределение Z = X * Y')
plt.grid(True)
plt.show()

print("---------------- 6 ----------------")
# Законы распределения X и Y
x_values = [-4, 0, 4]
p_x = [0.1, 0.5, 0.4]

y_values = [2, 4]
p_y = [0.5, 0.5]

# Создаем пустой словарь для хранения значений Z и их вероятностей
z_distribution = {}

# Перебираем все возможные комбинации (x, y)
for x in x_values:
    for y in y_values:
        z = (x + y) / 2  # Вычисляем Z
        prob_z = p_x[x_values.index(x)] * p_y[y_values.index(y)]  # Вероятность для данной пары (x, y)

        # Обновляем словарь с вероятностями
        if z in z_distribution:
            z_distribution[z] += prob_z
        else:
            z_distribution[z] = prob_z

# Сортируем значения Z и их вероятности
z_sorted = sorted(z_distribution.items())

# Выводим результат
print("Ряд распределения Z = (X + Y) / 2:")
for z, prob in z_sorted:
    print(f"Z = {z}: P(Z = {z}) = {prob}")

# Построение многоугольника распределения
z_values, probabilities = zip(*z_sorted)
plt.plot(z_values, probabilities, marker='o')
plt.xlabel('Значения Z')
plt.ylabel('Вероятность P(Z)')
plt.title('#6 Распределение Z = (X + Y) / 2')
plt.grid(True)
plt.show()

print("---------------- 7 ----------------")

# Параметры биномиального распределения
n = 3  # Число испытаний
p = 0.84  # Вероятность успеха

# Составляем ряд распределения
x_values = range(n + 1)
probabilities = [binom.pmf(k, n, p) for k in x_values]

# Выводим результат
print("Ряд распределения X:")
for x, prob in zip(x_values, probabilities):
    print(f"X = {x}: P(X = {x}) = {prob:.4f}")

# Построение многоугольника распределения
plt.bar(x_values, probabilities, tick_label=x_values)
plt.xlabel('Значения X')
plt.ylabel('Вероятность P(X)')
plt.title('#7 Распределение банок высшего качества')
plt.grid(axis='y')
plt.show()

print("---------------- 8 ----------------")
# Значения X и их вероятности (примерные данные из графика)
x_values = [0, 1, 2, 3, 4]
probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]

# Выводим закон распределения
print("Ряд распределения X:")
for x, prob in zip(x_values, probabilities):
    print(f"X = {x}: P(X = {x}) = {prob}")

# Построение многоугольника распределения
plt.bar(x_values, probabilities, tick_label=x_values)
plt.xlabel('Значения X')
plt.ylabel('Вероятность P(X)')
plt.title('#8 Распределение X')
plt.grid(True)
plt.show()

print("---------------- 9 ----------------")

# Значения X и их вероятности
x_values = [-2, 0, 3, 7]
p_values = [0.4, 0.1, 0.3, 0.2]

# Строим функцию распределения F(x)
F = []
cumulative_prob = 0
for p in p_values:
    cumulative_prob += p
    F.append(cumulative_prob)

# Выводим функцию распределения
print("Функция распределения F(x):")
for x, f in zip(x_values, F):
    print(f"F({x}) = {f}")

# Вычисляем требуемые вероятности
P_minus1_to_5 = F[2] - F[0]  # P(-1 < X < 5) = F(3) - F(-2)
P_4_to_10 = 1 - F[2]  # P(4 < X <= 10) = 1 - F(3)
P_leq_2 = F[1]  # P(X <= 2) = F(0)
P_3_to_7 = F[3] - F[1]  # P(3 <= X <= 7) = F(7) - F(0)
P_gt_7 = 1 - F[3]  # P(X > 7) = 1 - F(7)

# Выводим результаты
print("\nТребуемые вероятности:")
print(f"P(-1 < X < 5) = {P_minus1_to_5}")
print(f"P(4 < X <= 10) = {round(P_4_to_10, ndigits=4)}")
print(f"P(X <= 2) = {P_leq_2}")
print(f"P(3 <= X <= 7) = {P_3_to_7}")
print(f"P(X > 7) = {P_gt_7}")

# Построение графика функции распределения F(x)
x_points = []
y_points = []

# Добавляем начальную точку (до первого значения x)
x_points.append(x_values[0] - 1)  # Немного меньше первого значения x
y_points.append(0)  # Вероятность до первого значения равна 0

# Добавляем точки для каждого значения x_i
for i, x in enumerate(x_values):
    x_points.append(x)  # Точка перед скачком
    y_points.append(F[i - 1] if i > 0 else 0)  # Значение F(x) перед скачком

    x_points.append(x)  # Точка после скачка
    y_points.append(F[i])  # Значение F(x) после скачка

# Добавляем последнюю точку после максимального значения x
x_points.append(x_values[-1] + 1)  # Немного больше последнего значения x
y_points.append(1)  # Вероятность после последнего значения равна 1

# Проверяем, что массивы одинаковой длины
assert len(x_points) == len(y_points), "Массивы x_points и y_points должны иметь одинаковую длину"

# Построение графика
plt.step(x_points, y_points, where='post', label='$F(x)$')  # Используем 'post' для правильной ступеньки
plt.scatter(x_values, F, color='red', label='Точки разрыва')  # Отмечаем точки разрыва
plt.title('#9 График функции распределения $F(x)$')
plt.xlabel('$x$')
plt.ylabel('$F(x)$')
plt.grid(True)
plt.legend()
plt.show()

print("---------------- 10 ----------------")
# Значения X и их вероятности
x_values = [12, 16, 21, 26, 30]
p_values = [0.2, 0.1, 0.4, 0.2, 0.1]

# Строим функцию распределения F(x)
F = []
cumulative_prob = 0
for p in p_values:
    cumulative_prob += p
    F.append(cumulative_prob)

# Выводим функцию распределения
print("Функция распределения F(x):")
for x, f in zip(x_values, F):
    print(f"F({x}) = {round(f, ndigits=4)}")

# Вычисляем требуемые вероятности
P_15_to_25 = F[3] - F[1]  # P(15 < X < 25) = F(26) - F(16)
P_12_to_20 = F[2] - F[0]  # P(12 < X <= 20) = F(21) - F(12)
P_geq_21 = 1 - F[2]  # P(X >= 21) = 1 - F(21)
P_lt_16 = F[1]  # P(X < 16) = F(16)
P_leq_16 = F[1]  # P(X <= 16) = F(16)

# Выводим результаты
print("\nТребуемые вероятности:")
print(f"P(15 < X < 25) = {round(P_15_to_25, ndigits=4)}")
print(f"P(12 < X <= 20) = {round(P_12_to_20, ndigits=4)}")
print(f"P(X >= 21) = {round(P_geq_21, ndigits=4)}")
print(f"P(X < 16) = {round(P_lt_16, ndigits=4)}")
print(f"P(X <= 16) = {round(P_leq_16, ndigits=4)}")

# Построение графика функции распределения F(x)
x_points = []
y_points = []

# Добавляем начальную точку (до первого значения x)
x_points.append(x_values[0] - 1)  # Немного меньше первого значения x
y_points.append(0)  # Вероятность до первого значения равна 0

# Добавляем точки для каждого значения x_i
for i, x in enumerate(x_values):
    x_points.append(x)  # Точка перед скачком
    y_points.append(F[i - 1] if i > 0 else 0)  # Значение F(x) перед скачком

    x_points.append(x)  # Точка после скачка
    y_points.append(F[i])  # Значение F(x) после скачка

# Добавляем последнюю точку после максимального значения x
x_points.append(x_values[-1] + 1)  # Немного больше последнего значения x
y_points.append(1)  # Вероятность после последнего значения равна 1

# Проверяем, что массивы одинаковой длины
assert len(x_points) == len(y_points), "Массивы x_points и y_points должны иметь одинаковую длину"

# Построение графика
plt.step(x_points, y_points, where='post', label='$F(x)$')  # Используем 'post' для правильной ступеньки
plt.scatter(x_values, F, color='red', label='Точки разрыва')  # Отмечаем точки разрыва
plt.title('#10 График функции распределения $F(x)$')
plt.xlabel('$x$')
plt.ylabel('$F(x)$')
plt.grid(True)
plt.legend()
plt.show()