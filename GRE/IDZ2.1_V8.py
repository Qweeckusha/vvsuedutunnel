import sympy as sp

# Задача 1: Проверка свойства M(x + y) = M(x) + M(y)

# Определение символов
x_values = [1, 2]
x_probs = [0.2, 0.8]

y_values = [0.5, 1]
y_probs = [0.3, 0.7]

# Вычисление M(x)
M_x = sum(x * p for x, p in zip(x_values, x_probs))

# Вычисление M(y)
M_y = sum(y * p for y, p in zip(y_values, y_probs))

# Вычисление M(x + y)
M_x_plus_y = M_x + M_y

# Проверка свойства
property_holds = sp.simplify(M_x_plus_y - (M_x + M_y)) == 0

# Вывод результатов для задачи 1
print("Задача 1:")
print(f"M(x) = {M_x}")
print(f"M(y) = {M_y}")
print(f"M(x + y) = {M_x_plus_y}")
print(f"Свойство M(x + y) = M(x) + M(y) выполняется: {property_holds}\n")

# Задача 2: Ряд распределения и характеристики для случайных величин x и y

# Определение символа p
p = sp.symbols('p')

# Вероятности для числа попаданий k
P_0 = (1 - p)**2
P_1 = 2 * p * (1 - p)
P_2 = p**2

# Случайная величина x: разность между числом попаданий и числом промахов
x_values = [-2, 0, 2]
x_probs = [P_0, P_1, P_2]

# Математическое ожидание M(x)
M_x = sum(x * prob for x, prob in zip(x_values, x_probs))
M_x_simplified = sp.simplify(M_x)

# Дисперсия D(x)
M_x_squared = sum(x**2 * prob for x, prob in zip(x_values, x_probs))
D_x = sp.simplify(M_x_squared - M_x**2)

# Случайная величина y: сумма числа попаданий и числа промахов
y_values = [2]
y_probs = [1]

# Математическое ожидание M(y)
M_y = sum(y * prob for y, prob in zip(y_values, y_probs))

# Дисперсия D(y)
D_y = 0  # y всегда равно 2, дисперсия равна 0

# Вывод результатов для задачи 2
print("Задача 2:")
print(f"Ряд распределения x: {list(zip(x_values, x_probs))}")
print(f"M(x) = {M_x_simplified}")
print(f"D(x) = {D_x}")
print(f"M(y) = {M_y}")
print(f"D(y) = {D_y}")